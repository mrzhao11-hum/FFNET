"""Joint experiments isolate late detail from post-concat fusion."""
import gc
import unittest
import torch
from vehicle_ffnet.model import build_model
from vehicle_ffnet.pml_loss import ProximalMappingLoss

JOINTS=(("b1_detail","c1_channel"),("b1_detail","c2_spatial"),
        ("b3_semantic_gate","c1_channel"),("b3_semantic_gate","c2_spatial"))

class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_identity_fusion_preserves_standalone_detail_and_sampling_rng(self):
        image=torch.randn(1,3,96,128)
        for variant in ("b1_detail","b3_semantic_gate"):
            torch.manual_seed(42)
            standalone=build_model(False,"group_all",vehicle_variant=variant).eval()
            rng=torch.get_rng_state()
            with torch.no_grad():expected=standalone(image)[0]
            core={k:v.clone() for k,v in standalone.state_dict().items()
                  if k.startswith(("backbone.","ccsm","fusion.final","late_detail."))}
            del standalone;gc.collect()
            for mode in ("c1_channel","c2_spatial"):
                torch.manual_seed(42)
                joint=build_model(False,"group_all",fusion_mode=mode,vehicle_variant=variant).eval()
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                with torch.no_grad():actual=joint(image)[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                for key,value in core.items():torch.testing.assert_close(joint.state_dict()[key],value,rtol=0,atol=0)
                self.assertEqual(joint.output_stride,8)
                joint.load_state_dict(joint.state_dict(),strict=True)
                del joint;gc.collect()

    def test_detail_and_fusion_receive_gradients_in_all_four_joints(self):
        for variant,mode in JOINTS:
            torch.manual_seed(42)
            model=build_model(False,"group_all",fusion_mode=mode,vehicle_variant=variant)
            with torch.no_grad():
                model.fusion.final[0].weight.fill_(.01)
                model.fusion.final[0].bias.fill_(1.)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.005)
            image=torch.randn(1,3,96,128)
            points=[torch.tensor([[12.,16.],[68.,40.]])]
            for step in (1,2):
                optimizer.zero_grad(set_to_none=True)
                density,_=model(image)
                loss=ProximalMappingLoss()(density,points)
                self.assertTrue(torch.isfinite(loss));loss.backward()
                parameters={"detail":model.late_detail.local[0].weight,
                            "detail_down":model.late_detail.down.weight,
                            "fusion_output":model.fusion.output_projection.weight}
                if hasattr(model.late_detail,"gate"):parameters["detail_gate"]=model.late_detail.gate[-1].weight
                if step==2:parameters.update(model.fusion.probe_parameters())
                for name,p in parameters.items():
                    self.assertIsNotNone(p.grad,(variant,mode,step,name))
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(float(p.grad.abs().sum()),0,(variant,mode,step,name))
                optimizer.step()
            del model,optimizer;gc.collect()

    def test_disabling_detail_recovers_fusion_only_model(self):
        image=torch.randn(1,3,96,128)
        torch.manual_seed(42)
        fusion=build_model(False,"group_all",fusion_mode="c2_spatial").eval()
        with torch.no_grad():
            fusion.fusion.output_projection.weight.normal_(std=.01)
            expected=fusion(image)[0]
        fusion_state=fusion.fusion.state_dict();del fusion;gc.collect()
        torch.manual_seed(42)
        joint=build_model(False,"group_all",fusion_mode="c2_spatial",vehicle_variant="b3_semantic_gate").eval()
        joint.fusion.load_state_dict(fusion_state,strict=True)
        with torch.no_grad():
            joint.late_detail.scale.logit.fill_(-100)
            actual=joint(image)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_unsupported_combinations_still_rejected(self):
        for variant,mode in (("d4_detail","c2_spatial"),("a3_lde_vftm","c1_channel"),
                             ("b3_semantic_gate","c3_semantic"),("b2_self_gate","c2_spatial")):
            with self.assertRaises(ValueError):build_model(False,"group_all",fusion_mode=mode,vehicle_variant=variant)

if __name__=="__main__":unittest.main()
