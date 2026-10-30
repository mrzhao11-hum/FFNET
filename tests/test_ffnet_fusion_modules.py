"""Verify fusion controls preserve FFNet and activate after identity initialization."""
import copy
import gc
import unittest
import torch
from vehicle_ffnet.model import build_model, upstream_modules
from vehicle_ffnet.fusion_modules import CollaborativeFusion, FUSION_MODES
from vehicle_ffnet.pml_loss import ProximalMappingLoss
from vehicle_ffnet.inference import predict_density_full


class FusionModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def features(self):
        return (torch.randn(1,16,8,12),torch.randn(1,32,4,6),torch.randn(1,64,2,3))

    def base_fusion(self):
        network,_=upstream_modules()
        original=network.Fusion(16,32,64)
        with torch.no_grad():
            original.final[0].weight.fill_(.01)
            original.final[0].bias.fill_(1.)
        return original

    def test_identity_start_preserves_all_scales_and_original_head(self):
        original=self.base_fusion();features=self.features()
        expected=original(*features).detach()
        for mode in FUSION_MODES:
            block=CollaborativeFusion(copy.deepcopy(original),mode)
            torch.testing.assert_close(block(*features),expected,atol=0,rtol=0)
            self.assertEqual(block.final[0].in_channels,112)
            self.assertTrue((block.output_projection.weight==0).all())
            self.assertAlmostEqual(float(block.scale()),.1,places=6)

    def test_second_update_reaches_input_spatial_and_conditional_gate(self):
        points=[torch.tensor([[12.,16.],[68.,40.]])]
        for mode in FUSION_MODES:
            torch.manual_seed(42)
            block=CollaborativeFusion(self.base_fusion(),mode)
            features=self.features()
            optimizer=torch.optim.SGD(block.parameters(),lr=1e-3)
            loss_fn=ProximalMappingLoss()
            for step in (1,2):
                optimizer.zero_grad(set_to_none=True)
                prediction=block(*features)
                total=loss_fn(prediction,points)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                gradients=block.probe_parameters()
                for name,parameter in gradients.items():
                    self.assertIsNotNone(parameter.grad,(mode,step,name))
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    if step==2 or name=='fusion_output':
                        self.assertGreater(float(parameter.grad.abs().sum()),0,(mode,step,name))
                optimizer.step()

    def test_spatial_control_starts_as_identity_and_can_add_neighbor_interaction(self):
        c1=CollaborativeFusion(self.base_fusion(),'c1_channel')
        c2=CollaborativeFusion(self.base_fusion(),'c2_spatial')
        missing,unexpected=c2.load_state_dict(c1.state_dict(),strict=False)
        self.assertEqual(missing,['spatial.weight']);self.assertEqual(unexpected,[])
        with torch.no_grad():
            c1.output_projection.weight.normal_(std=.03)
            c2.output_projection.weight.copy_(c1.output_projection.weight)
        features=self.features()
        torch.testing.assert_close(c1(*features),c2(*features),atol=1e-6,rtol=1e-6)
        with torch.no_grad():c2.spatial.weight[:,:,0,1].add_(.5)
        self.assertGreater(float((c1(*features)-c2(*features)).abs().sum()),1e-5)

    def test_c3_gate_is_content_conditioned_and_has_nonzero_floor(self):
        block=CollaborativeFusion(self.base_fusion(),'c3_semantic')
        context=torch.randn(1,16,8,12)
        t8=torch.randn_like(context)
        gate1=block.gate_weights(t8,context)
        gate2=block.gate_weights(-t8,context)
        self.assertTrue(((gate1>.1)&(gate1<1)).all())
        self.assertGreater(float((gate1-gate2).abs().sum()),1e-6)

    def test_full_model_initial_state_rng_density_and_checkpoint_roundtrip(self):
        image=torch.randn(1,3,96,128)
        torch.manual_seed(42)
        original=build_model(False,'group_all').eval()
        with torch.no_grad():expected=original(image)[0]
        original_state={k:v.clone() for k,v in original.state_dict().items()}
        rng=torch.get_rng_state();del original;gc.collect()
        for mode in FUSION_MODES:
            torch.manual_seed(42)
            model=build_model(False,'group_all',fusion_mode=mode).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            state=model.state_dict()
            for name,value in original_state.items():
                torch.testing.assert_close(state[name],value,rtol=0,atol=0)
            self.assertEqual(model.output_stride,8)
            self.assertEqual(sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),0)
            self.assertFalse(hasattr(model,'late_detail'))
            with torch.no_grad():
                actual=model(image)[0]
                fine=predict_density_full(model,image)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertAlmostEqual(float(fine.sum()),float(actual.sum()),places=4)
            model.load_state_dict(state,strict=True)
            del model,state;gc.collect()

    def test_fusion_only_mode_rejects_combining_vehicle_modules(self):
        with self.assertRaisesRegex(ValueError,'require fusion_mode=original'):
            build_model(False,'group_all',fusion_mode='c1_channel',vehicle_variant='b2_self_gate')


if __name__=='__main__':unittest.main()
