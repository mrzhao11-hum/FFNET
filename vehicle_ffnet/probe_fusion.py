"""Two full-image PML updates verify zero-initialized fusion branches activate."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import torch
from .data import VehicleDataset
from .fusion_modules import FUSION_MODES, CollaborativeFusion
from .model import build_model
from .pml_loss import ProximalMappingLoss
from .train import compute_loss, set_seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--no-pretrained",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("fusion_mode") not in FUSION_MODES or cfg.get("supervision")!="pml" or cfg.get("train_mode")!="full":
        parser.error("requires a C1-C3 full-image PML config")
    set_seed(cfg["seed"])
    ds=VehicleDataset(cfg["data_root"],Path(cfg["manifest_dir"])/"train.json",train=True,
                      crop_size=cfg["crop_size"],full_image=True,flip_probability=0)
    index=max(range(len(ds.records)),key=lambda i:ds.records[i]["count"])
    image,points,target,valid=ds[index]
    device=torch.device(args.device)
    model=build_model(pretrained=cfg["pretrained"] and not args.no_pretrained,
                      attention_norm=cfg["attention_norm"],fusion_mode=cfg["fusion_mode"],
                      vehicle_variant=cfg.get("vehicle_variant","baseline")).to(device).train()
    assert isinstance(model.fusion,CollaborativeFusion) and model.output_stride==8
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    loss_fn=ProximalMappingLoss(8,cfg["pml_radius"],cfg["pml_density_factor"])
    image=image[None].to(device);target=target[None].to(device);points=[points.to(device)]
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        total,terms,pred,_=compute_loss(model,loss_fn,image,points,target,cfg["wot"],cfg["wtv"],
                                      valid_shape=valid,supervision="pml",pml_count_weight=0)
        if not bool(torch.isfinite(total)):raise FloatingPointError("nonfinite fusion loss")
        total.backward()
        for name,parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"nonfinite gradient: {name}")
        parameters=model.fusion.probe_parameters()
        detail=getattr(model,"late_detail",None)
        if detail is not None:
            parameters.update(f4_local=detail.local[0].weight,f4_down=detail.down.weight)
            if hasattr(detail,"gate"):parameters["f4_gate"]=detail.gate[-1].weight
        norms={name:float(p.grad.abs().sum()) if p.grad is not None else 0.
               for name,p in parameters.items()}
        required=norms if step==2 else {k:v for k,v in norms.items()
                                      if k=="fusion_output" or k.startswith("f4_")}
        if any(not math.isfinite(x) or x<=0 for x in required.values()):
            raise RuntimeError(f"inactive fusion gradient at step {step}: {norms}")
        print(f"step={step} id={ds.records[index]['id']} GT={len(points[0])} prediction={float(pred.item()):.3f} "
              f"loss={float(total.detach()):.5f} gradient_L1={norms}",flush=True)
        optimizer.step()
    if device.type=="cuda":torch.cuda.synchronize(device)
    peak=torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=="cuda" else 0.
    print(f"Fusion probe passed: {cfg['fusion_mode']} variant={cfg.get('vehicle_variant','baseline')} D8={valid[0]}x{valid[1]} "
          f"peak_allocated={peak:.2f}GiB two_updates_elapsed={time.perf_counter()-started:.2f}s; "
          "step-1 upstream zeros are expected from the zero output projection",flush=True)

if __name__=="__main__":main()
