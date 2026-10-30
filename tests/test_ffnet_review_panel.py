import csv
import tempfile
import unittest
from pathlib import Path

import torch
from vehicle_ffnet.diagnose_review_panel import checked_panel, peak_metrics, roi_mass


class ReviewPanelTests(unittest.TestCase):
    def test_peak_from_neighbor_is_flagged_and_zero_map_has_no_peak(self):
        density = torch.tensor([[0., 1.]])
        distance = torch.tensor([[0., 8.]])
        owner = torch.tensor([[0,1]])
        peak = peak_metrics(density,distance,owner,0)
        self.assertFalse(peak['peak_owned_by_this_vehicle'])
        self.assertEqual(peak['peak_owner_index'],1)
        peak = peak_metrics(density*0,distance,owner,0)
        self.assertIsNone(peak['peak_distance_px'])

    def test_roi_mass_partial_cell_uses_real_valid_area(self):
        density = torch.tensor([[1.,2.]])
        self.assertAlmostEqual(roi_mass(density,(0,0,10,8),8,10),3.)
        self.assertAlmostEqual(roi_mass(density,(8,0,9,8),8,10),1.)

    def test_geometry_checked_before_reusing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'ann.txt').write_text('0 0 8 8 1\n')
            row=dict(id='a',object_index=0,x1=0,y1=0,x2=8,y2=8,image_width=16,
                     image_height=16,visibility='complete',review_protocol='assistant_visual_blind')
            def write():
                with (root/'review_labels.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            write()
            records=[dict(id='a',annotation='ann.txt',width=16,height=16)]
            selected,total=checked_panel(root,records,root)
            self.assertEqual((len(selected),total),(1,1))
            row['x1']=1;write()
            with self.assertRaisesRegex(ValueError,'mismatch'):
                checked_panel(root,records,root)


if __name__ == '__main__':
    unittest.main()
