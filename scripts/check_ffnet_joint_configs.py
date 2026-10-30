"""Validate the 2x2 B1/B3 x C1/C2 experiment without changing supervision."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
JOINTS={"j1_b1_c1":("b1_detail","c1_channel"),"j2_b1_c2":("b1_detail","c2_spatial"),
        "j3_b3_c1":("b3_semantic_gate","c1_channel"),"j4_b3_c2":("b3_semantic_gate","c2_spatial")}

def check():
    base=json.loads((ROOT/"configs/carpk_ntu162_b3_semantic_gate_seed42.json").read_text(encoding="utf-8"))
    for arm,(variant,mode) in JOINTS.items():
        cfg=json.loads((ROOT/f"configs/carpk_ntu162_joint_{arm}_seed42.json").read_text(encoding="utf-8"))
        changed={k for k in set(base)|set(cfg) if base.get(k)!=cfg.get(k)}
        if changed-{"vehicle_variant","fusion_mode","output_dir"}:
            raise ValueError(f"{arm}: training settings changed: {changed}")
        assert cfg["vehicle_variant"]==variant and cfg["fusion_mode"]==mode
        assert cfg["pretrained"] and cfg["output_stride"]==8
        assert cfg["supervision"]=="pml" and cfg["pml_count_weight"]==0
    ids={}
    for split,size in (("train",827),("val",162),("test",459)):
        rows=json.loads((ROOT/base["manifest_dir"]/f"{split}.json").read_text(encoding="utf-8"))["samples"]
        ids[split]={r["id"] for r in rows};assert len(rows)==len(ids[split])==size
    assert not (ids["train"]&ids["val"] or ids["train"]&ids["test"] or ids["val"]&ids["test"])
    print("J1-J4: B1/B3 x C1/C2, ImageNet full training, unchanged FTM/head, D8/PML, 827/162/459")

if __name__=="__main__":check()
