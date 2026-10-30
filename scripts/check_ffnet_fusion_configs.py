"""Check that C1-C3 change only fusion_mode/output_dir, using the NTU162 split."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
ARMS=("c1_channel","c2_spatial","c3_semantic")

def check():
    baseline=json.loads((ROOT/"configs/carpk_ntu162_ffnet_pml_full_seed42.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        cfg=json.loads((ROOT/f"configs/carpk_ntu162_fusion_{arm}_seed42.json").read_text(encoding="utf-8"))
        changed={k for k in set(cfg)|set(baseline) if cfg.get(k)!=baseline.get(k)}
        if changed != {"fusion_mode","output_dir"}:
            raise ValueError(f"{arm}: unexpected baseline differences {changed}")
        assert cfg["fusion_mode"]==arm
        assert cfg.get("vehicle_variant","baseline")=="baseline"
        assert cfg["pretrained"] and cfg["supervision"]=="pml" and cfg["pml_count_weight"]==0
        assert cfg.get("density_head_init","original")=="original"
        assert cfg.get("output_stride",8)==8
    ids={}
    for split,count in (("train",827),("val",162),("test",459)):
        rows=json.loads((ROOT/baseline["manifest_dir"]/f"{split}.json").read_text(encoding="utf-8"))["samples"]
        ids[split]={r["id"] for r in rows}
        assert len(rows)==len(ids[split])==count,(split,len(rows))
    assert not (ids["train"]&ids["val"] or ids["train"]&ids["test"] or ids["val"]&ids["test"])
    print("C1-C3: fusion-only changes; ImageNet full training, original FTM/head, D8/PML; 827/162/459")

if __name__=="__main__":check()
