import gc
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from vehicle_ffnet.model import build_model, initialize_density_head
from vehicle_ffnet.data import VehicleDataset
from vehicle_ffnet.detail_resolution import LateDetail
from vehicle_ffnet.inference import predict_density_full
from vehicle_ffnet.diagnose_instance_mass import predict_valid_d8, assign_mass
from vehicle_ffnet.pml_loss import ProximalMappingLoss
from vehicle_ffnet.train import compute_loss


class DetailResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_b_controls_start_identical_and_gate_receives_gradient(self):
        f4, t8 = torch.randn(1,96,16,24), torch.randn(1,16,8,12)
        outputs = []
        for variant in ('b1_detail','b2_self_gate','b3_semantic_gate'):
            torch.manual_seed(42)
            module = LateDetail(variant)
            output = module(f4,t8)
            outputs.append(output.detach())
            output.square().mean().backward()
            self.assertGreater(module.local[0].weight.grad.abs().sum().item(),0)
            if hasattr(module,'gate'):
                self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(),0)
        for output in outputs[1:]:
            torch.testing.assert_close(output,outputs[0],rtol=0,atol=0)

    def test_backbone_ftm_and_original_head_initialization_preserved(self):
        torch.manual_seed(42)
        original = build_model(False,'group_all')
        state = {k:v.clone() for k,v in original.state_dict().items()}
        rng = torch.get_rng_state()
        del original
        gc.collect()
        for variant in ('b1_detail','b2_self_gate','b3_semantic_gate','d4_detail'):
            torch.manual_seed(42)
            model = build_model(False,'group_all',vehicle_variant=variant)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in model.state_dict().items():
                if key.startswith(('backbone.','ccsm','fusion.upsample')) or (variant!='d4_detail' and key.startswith('fusion.')):
                    torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            self.assertEqual(sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),0)
            del model
            gc.collect()

    def test_d4_shape_pml_gradients_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        model = build_model(False,'group_all',vehicle_variant='d4_detail')
        initialize_density_head(model,.004,1e-5,allow_relu=True)
        model.eval()
        image = torch.randn(1,3,96,128)
        points = [torch.tensor([[8.,12.],[122.,77.]])]
        gt = torch.zeros(1,1,21,32); gt[0,0,3,2]=1;gt[0,0,19,30]=1
        criterion = ProximalMappingLoss(stride=4)
        total,_,_,_ = compute_loss(model,criterion,image,points,gt,0,0,valid_shape=(21,32),supervision='pml')
        total.backward()
        for layer in (model.fusion.semantic[0], model.fusion.local[0], model.fusion.gate[-1], model.fusion.final[0]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()),0)
        state = model.state_dict()
        del model
        other = build_model(False,'group_all',vehicle_variant='d4_detail').eval()
        other.load_state_dict(state,strict=True)
        with torch.no_grad():
            density,_ = other(image)
        self.assertEqual(tuple(density.shape),(1,1,24,32))
        with self.assertRaisesRegex(ValueError,'stride mismatch'):
            compute_loss(other,ProximalMappingLoss(stride=8),image,points,torch.zeros_like(density),0,0,supervision='pml')

    def test_disabling_added_detail_recovers_original_forward(self):
        image=torch.randn(1,3,96,128)
        torch.manual_seed(42)
        original=build_model(False,'group_all').eval()
        with torch.no_grad(): expected=original(image)[0]
        del original
        torch.manual_seed(42)
        candidate=build_model(False,'group_all',vehicle_variant='b1_detail').eval()
        with torch.no_grad():
            candidate.late_detail.scale.logit.fill_(-100)
            actual=candidate(image)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_stride4_cell_centers_and_mass_ownership(self):
        # Cell centres (1.5,1.5), (5.5,1.5): exactly one vehicle each.
        density = torch.ones(1,1,1,2,requires_grad=True)
        points = torch.tensor([[1.5,1.5],[5.5,1.5]])
        loss,point,transport = ProximalMappingLoss(4)(density,[points],True)
        self.assertEqual(float(point),0)
        self.assertEqual(float(transport),0)
        loss.backward()
        assigned,_,_ = assign_mass(density.detach()[0,0],points,stride=4)
        torch.testing.assert_close(assigned,torch.ones(2))

    def test_padding_excluded_and_valid_mass_preserved_at_both_strides(self):
        class PaddedMass(torch.nn.Module):
            def __init__(self,stride):
                super().__init__();self.output_stride=stride
            def forward(self,image):
                s=self.output_stride
                density=torch.ones(1,1,image.shape[-2]//s,image.shape[-1]//s)
                density[:,:, (73+s-1)//s:,:]=10000
                density[:,:,:, (103+s-1)//s:]=10000
                return density,density/density.sum()
        image=torch.zeros(1,3,73,103)
        for stride in (4,8):
            model=PaddedMass(stride)
            native=predict_valid_d8(model,image)
            fine=predict_density_full(model,image)
            expected=((73+stride-1)//stride)*((103+stride-1)//stride)
            self.assertEqual(float(native.sum()),expected)
            self.assertAlmostEqual(float(fine.sum()),expected,places=3)
            self.assertEqual(tuple(fine.shape),(1,1,73,103))

    def test_dataset_stride4_boundary_centers_survive_flip_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(103,73)).save(root/'image.png')
            (root/'boxes.txt').write_text('0 0 2 4\n100 70 102 72\n')
            (root/'train.json').write_text(json.dumps({'samples':[dict(id='x',image='image.png',annotation='boxes.txt',width=103,height=73,count=2)]}))
            dataset=VehicleDataset(root,root/'train.json',train=True,full_image=True,flip_probability=1,output_stride=4)
            image,points,target,valid=dataset[0]
            self.assertEqual(tuple(image.shape),(3,96,128))
            self.assertEqual(valid,(19,26))
            self.assertEqual(float(target.sum()),2)
            torch.testing.assert_close(points,torch.tensor([[101.,2.],[1.,71.]]))


if __name__=='__main__':
    unittest.main()
