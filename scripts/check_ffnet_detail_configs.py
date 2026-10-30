"""Verify shared baseline settings and the unchanged NTU162 protocol."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=('b1_detail','b2_self_gate','b3_semantic_gate','d4_detail')

def check():
    baseline=json.loads((ROOT/'configs/carpk_ntu162_ffnet_pml_full_seed42.json').read_text(encoding='utf-8'))
    allowed={'vehicle_variant','output_dir','output_stride','density_activation','density_head_init','density_head_weight_std'}
    for arm in ARMS:
        config=json.loads((ROOT/f'configs/carpk_ntu162_{arm}_seed42.json').read_text(encoding='utf-8'))
        changed={k for k in set(config)|set(baseline) if config.get(k)!=baseline.get(k)}
        if changed-allowed: raise ValueError(f'{arm}: unexpected changes {changed-allowed}')
        assert config['vehicle_variant']==arm
        assert config['output_stride']==(4 if arm=='d4_detail' else 8)
        assert config['pml_count_weight']==0 and config['supervision']=='pml'
    manifest=ROOT/baseline['manifest_dir']
    for split,size in (('train',827),('val',162),('test',459)):
        records=json.loads((manifest/f'{split}.json').read_text(encoding='utf-8'))['samples']
        assert len(records)==size,(split,len(records))
    print('B1/B2/B3 D8 + D4 candidate: shared PML baseline settings; split=827/162/459; no B4 bypass')

if __name__=='__main__':check()
