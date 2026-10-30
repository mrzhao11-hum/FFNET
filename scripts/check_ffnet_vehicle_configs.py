"""Check that the four vehicle arms differ only in model variant and run path."""

from __future__ import annotations

import json
from pathlib import Path


ARMS = ("a1_lde", "a2_vftm", "a3_lde_vftm", "a4_lde_vftm_gate")
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "carpk_ntu162_ffnet_pml_full_seed42.json"


def main() -> None:
    baseline = json.loads(BASE.read_text(encoding="utf-8"))
    for arm in ARMS:
        path = ROOT / "configs" / f"carpk_ntu162_vehicle_{arm}_seed42.json"
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("vehicle_variant") != arm:
            raise ValueError(f"wrong vehicle_variant in {path}")
        expected_run = f"outputs/ffnet_vehicle_modules_ntu162_seed42/{arm}"
        if candidate.get("output_dir") != expected_run:
            raise ValueError(f"wrong output_dir in {path}")
        compared = {key: value for key, value in candidate.items()
                    if key not in {"vehicle_variant", "output_dir"}}
        expected = {key: value for key, value in baseline.items() if key != "output_dir"}
        if compared != expected:
            difference = sorted(key for key in set(compared) | set(expected)
                                if compared.get(key) != expected.get(key))
            raise ValueError(f"{arm} differs from PML baseline in {difference}")
    print("Four vehicle configs match the existing NTU162 PML baseline apart from variant/output_dir")


if __name__ == "__main__":
    main()
