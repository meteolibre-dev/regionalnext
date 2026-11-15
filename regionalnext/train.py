"""Training & evaluation recipe for RegionalNext FiLM-UNet 250m downscaling."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

from . import config
from .build_patches import ALL_CHANNELS
from .dataset import RegionalNextFullGridDataset, RegionalNextPatchDataset
from .model import FiLMUNet

log = logging.getLogger("regionalnext.train")

ELEV_IDX = ALL_CHANNELS.index("elevation_m")


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    w_t2m: float = 1.0,
    w_wind: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked Huber/SmoothL1 loss on sparse station pixels [t2m, u10, v10]."""
    mt = mask[:, 0]
    mw = mask[:, 1]  # u10 & v10 share wind validity mask

    loss_t = (
        F.smooth_l1_loss(pred[:, 0][mt], target[:, 0][mt], beta=beta, reduction="mean")
        if mt.any()
        else pred.sum() * 0.0
    )
    if mw.any():
        loss_u = F.smooth_l1_loss(pred[:, 1][mw], target[:, 1][mw], beta=beta, reduction="mean")
        loss_v = F.smooth_l1_loss(pred[:, 2][mw], target[:, 2][mw], beta=beta, reduction="mean")
        loss_w = 0.5 * (loss_u + loss_v)
    else:
        loss_w = pred.sum() * 0.0

    total = w_t2m * loss_t + w_wind * loss_w
    return total, {
        "loss": float(total.detach()),
        "loss_t2m": float(loss_t.detach()),
        "loss_wind": float(loss_w.detach()),
    }


@torch.no_grad()
def evaluate(
    model: FiLMUNet,
    loader: DataLoader,
    device: torch.device,
    elev_mean: float,
    elev_std: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate model vs coarse ERA5-Land across elevation bands and wind."""
    model.eval()
    sums = {
        "n_t": 0, "abs_t": 0.0, "sq_t": 0.0, "c_abs_t": 0.0, "c_sq_t": 0.0,
        "n_mtn": 0, "abs_mtn": 0.0, "c_abs_mtn": 0.0,
        "n_high": 0, "abs_high": 0.0, "c_abs_high": 0.0,
        "n_w": 0, "abs_spd": 0.0, "c_abs_spd": 0.0, "sq_uv": 0.0, "c_sq_uv": 0.0,
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        coarse = batch["coarse"].to(device, non_blocking=True)
        solar = batch["solar"].to(device, non_blocking=True) if "solar" in batch else None

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x, coarse, solar=solar)

        pred = pred.float()
        mt = mask[:, 0]
        if mt.any():
            err_p = pred[:, 0][mt] - y[:, 0][mt]
            err_c = coarse[:, 0][mt] - y[:, 0][mt]
            elev = x[:, ELEV_IDX][mt] * elev_std + elev_mean

            sums["n_t"] += int(mt.sum())
            sums["abs_t"] += float(err_p.abs().sum())
            sums["sq_t"] += float((err_p ** 2).sum())
            sums["c_abs_t"] += float(err_c.abs().sum())
            sums["c_sq_t"] += float((err_c ** 2).sum())

            m_mtn = elev >= 1000.0
            if m_mtn.any():
                sums["n_mtn"] += int(m_mtn.sum())
                sums["abs_mtn"] += float(err_p[m_mtn].abs().sum())
                sums["c_abs_mtn"] += float(err_c[m_mtn].abs().sum())

            m_high = elev >= 2000.0
            if m_high.any():
                sums["n_high"] += int(m_high.sum())
                sums["abs_high"] += float(err_p[m_high].abs().sum())
                sums["c_abs_high"] += float(err_c[m_high].abs().sum())

        mw = mask[:, 1]
        if mw.any():
            pu, pv = pred[:, 1][mw], pred[:, 2][mw]
            cu, cv = coarse[:, 1][mw], coarse[:, 2][mw]
            yu, yv = y[:, 1][mw], y[:, 2][mw]
            pspd = torch.sqrt(pu ** 2 + pv ** 2)
            cspd = torch.sqrt(cu ** 2 + cv ** 2)
            yspd = torch.sqrt(yu ** 2 + yv ** 2)

            sums["n_w"] += int(mw.sum())
            sums["abs_spd"] += float((pspd - yspd).abs().sum())
            sums["c_abs_spd"] += float((cspd - yspd).abs().sum())
            sums["sq_uv"] += float(((pu - yu) ** 2 + (pv - yv) ** 2).sum())
            sums["c_sq_uv"] += float(((cu - yu) ** 2 + (cv - yv) ** 2).sum())

    nt = max(1, sums["n_t"])
    nm = max(1, sums["n_mtn"])
    nh = max(1, sums["n_high"])
    nw = max(1, sums["n_w"])
    return {
        "n_t2m_obs": sums["n_t"],
        "t2m_mae": sums["abs_t"] / nt,
        "t2m_rmse": math.sqrt(sums["sq_t"] / nt),
        "coarse_t2m_mae": sums["c_abs_t"] / nt,
        "coarse_t2m_rmse": math.sqrt(sums["c_sq_t"] / nt),
        "mtn_t2m_mae": sums["abs_mtn"] / nm if sums["n_mtn"] else float("nan"),
        "coarse_mtn_t2m_mae": sums["c_abs_mtn"] / nm if sums["n_mtn"] else float("nan"),
        "high_t2m_mae": sums["abs_high"] / nh if sums["n_high"] else float("nan"),
        "coarse_high_t2m_mae": sums["c_abs_high"] / nh if sums["n_high"] else float("nan"),
        "n_wind_obs": sums["n_w"],
        "wspd_mae": sums["abs_spd"] / nw,
        "coarse_wspd_mae": sums["c_abs_spd"] / nw,
        "wind_vec_rmse": math.sqrt(sums["sq_uv"] / nw),
        "coarse_wind_vec_rmse": math.sqrt(sums["c_sq_uv"] / nw),
    }


def _maybe_download_hf(data_dir: Path, repo_id: str, dataset_type: str) -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["full_grid/**"] if dataset_type == "full_grid" else [
        "patches_dense.zarr/**",
        "patches_sparse.parquet",
        "patches_meta.parquet",
        "norm_stats.json",
    ]
    log.info("Downloading %s (%s) from hf://datasets/%s -> %s", dataset_type, patterns, repo_id, data_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=patterns,
    )
    return data_dir / "full_grid" if dataset_type == "full_grid" else data_dir


def _build_dataset(data_dir: Path, dataset_type: str, split: str, random_crop: bool, eval_heldout_only: bool):
    if dataset_type == "full_grid":
        return RegionalNextFullGridDataset(
            data_dir,
            split=split,
            normalize=True,
            eval_heldout_only=eval_heldout_only,
            random_crop=random_crop,
        )
    return RegionalNextPatchDataset(
        data_dir,
        split=split,
        normalize=True,
        eval_heldout_only=eval_heldout_only,
    )


def _worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2**32))


def _subset_eval_loader(ds, max_samples: int, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    if max_samples > 0 and len(ds) > max_samples:
        idx = np.linspace(0, len(ds) - 1, max_samples, dtype=np.int64)
        ds = Subset(ds, idx.tolist())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    if args.hf_download:
        data_dir = _maybe_download_hf(data_dir, args.repo_id, args.dataset_type)
    elif args.dataset_type == "full_grid" and (data_dir / "full_grid").is_dir():
        data_dir = data_dir / "full_grid"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    num_workers = args.num_workers if args.num_workers >= 0 else (8 if device.type == "cuda" else 0)
    pin_memory = device.type == "cuda"

    log.info("Device=%s | AMP=%s (%s) | workers=%d | data=%s",
             device, use_amp, amp_dtype if use_amp else "fp32", num_workers, data_dir)

    stats = json.loads((data_dir / "norm_stats.json").read_text())
    elev_mean = float(stats["mean"][ELEV_IDX])
    elev_std = float(stats["std"][ELEV_IDX])

    ckpt_wn3 = args.wn3_compat
    ckpt_solar = args.solar_film
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_wn3 = bool(ckpt.get("args", {}).get("wn3_compat", ckpt_wn3))
        ckpt_solar = bool(ckpt.get("args", {}).get("solar_film", ckpt_solar))

    model = FiLMUNet(
        base_ch=args.base_ch,
        cond_emb_dim=args.cond_emb_dim,
        dropout=args.dropout,
        wn3_compat=ckpt_wn3,
        use_solar_film=ckpt_solar,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("FiLMUNet initialized (wn3_compat=%s, solar_film=%s): %.2fM parameters",
             ckpt_wn3, ckpt_solar, n_params / 1e6)

    if args.resume:
        model.load_state_dict(ckpt["model"])
        log.info("Loaded weights from %s", args.resume)

    if args.eval_only:
        ds_test = _build_dataset(data_dir, args.dataset_type, "test", False, False)
        ds_held = _build_dataset(data_dir, args.dataset_type, "test", False, True)
        test_loader = _subset_eval_loader(ds_test, args.val_max_samples, args.batch_size, num_workers, pin_memory)
        held_loader = _subset_eval_loader(ds_held, args.val_max_samples, args.batch_size, num_workers, pin_memory)
        m_all = evaluate(model, test_loader, device, elev_mean, elev_std, use_amp, amp_dtype)
        m_held = evaluate(model, held_loader, device, elev_mean, elev_std, use_amp, amp_dtype)
        log.info("TEST ALL:     %s", json.dumps({k: round(v, 4) for k, v in m_all.items()}))
        log.info("TEST HELDOUT: %s", json.dumps({k: round(v, 4) for k, v in m_held.items()}))
        return

    ds_train = _build_dataset(data_dir, args.dataset_type, "train", args.dataset_type == "full_grid", False)
    ds_val = _build_dataset(data_dir, args.dataset_type, "val", False, False)
    ds_val_held = _build_dataset(data_dir, args.dataset_type, "val", False, True)
    log.info("Dataset sizes: train=%d | val=%d | val_heldout=%d", len(ds_train), len(ds_val), len(ds_val_held))

    if args.train_samples_per_epoch > 0 and len(ds_train) > args.train_samples_per_epoch:
        train_sampler = RandomSampler(ds_train, replacement=False, num_samples=args.train_samples_per_epoch)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = _subset_eval_loader(ds_val, args.val_max_samples, args.batch_size, num_workers, pin_memory)
    val_held_loader = _subset_eval_loader(ds_val_held, args.val_max_samples, args.batch_size, num_workers, pin_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "loss_t2m": 0.0, "loss_wind": 0.0}
        n_steps = 0

        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            coarse = batch["coarse"].to(device, non_blocking=True)
            solar = batch["solar"].to(device, non_blocking=True) if "solar" in batch else None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(x, coarse, solar=solar)
                loss, parts = masked_smooth_l1(
                    pred, y, mask,
                    beta=args.huber_beta,
                    w_t2m=args.w_t2m,
                    w_wind=args.w_wind,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            for k in running:
                running[k] += parts[k]
            n_steps += 1

            if step % args.log_every == 0 or step == len(train_loader):
                log.info(
                    "Epoch %d/%d [%d/%d] lr=%.2e loss=%.4f (t2m=%.4f, wind=%.4f)",
                    epoch, args.epochs, step, len(train_loader),
                    scheduler.get_last_lr()[0],
                    running["loss"] / n_steps,
                    running["loss_t2m"] / n_steps,
                    running["loss_wind"] / n_steps,
                )
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        val_m = evaluate(model, val_loader, device, elev_mean, elev_std, use_amp, amp_dtype)
        held_m = evaluate(model, val_held_loader, device, elev_mean, elev_std, use_amp, amp_dtype)
        elapsed = time.time() - t0

        log.info(
            "Epoch %d done in %.1fs | VAL T2m MAE=%.3f (coarse=%.3f) | MTN>=1km=%.3f (coarse=%.3f) | HELDOUT T2m MAE=%.3f | WSPD MAE=%.3f (coarse=%.3f)",
            epoch, elapsed,
            val_m["t2m_mae"], val_m["coarse_t2m_mae"],
            val_m["mtn_t2m_mae"], val_m["coarse_mtn_t2m_mae"],
            held_m["t2m_mae"],
            val_m["wspd_mae"], val_m["coarse_wspd_mae"],
        )

        record = {
            "epoch": epoch,
            "train_loss": running["loss"] / max(1, n_steps),
            "val": val_m,
            "val_heldout": held_m,
            "elapsed_s": round(elapsed, 1),
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        ckpt_obj = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_t2m_mae": val_m["t2m_mae"],
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        torch.save(ckpt_obj, args.out_dir / "last.pt")
        if val_m["t2m_mae"] < best_val_mae:
            best_val_mae = val_m["t2m_mae"]
            torch.save(ckpt_obj, args.out_dir / "best.pt")
            log.info("Saved new best checkpoint (val_t2m_mae=%.4f) -> %s", best_val_mae, args.out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FiLM-UNet for RegionalNext 250m downscaling")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--dataset-type", choices=["full_grid", "patches"], default="full_grid")
    p.add_argument("--hf-download", action="store_true", help="Download dataset from HF into --data-dir")
    p.add_argument("--repo-id", default="meteolibre-dev/regionalnext")
    p.add_argument("--out-dir", type=Path, default=config.ROOT_DIR / "runs" / "film_unet")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-samples-per-epoch", type=int, default=65536,
                   help="Random samples per epoch for full_grid (0 = full pass)")
    p.add_argument("--val-max-samples", type=int, default=16384,
                   help="Max val/test samples per eval pass (0 = full split)")
    p.add_argument("--max-train-steps", type=int, default=0, help="Debug limit steps/epoch")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--cond-emb-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--w-t2m", type=float, default=1.0)
    p.add_argument("--w-wind", type=float, default=0.5)
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=-1, help="-1 = auto (8 on CUDA, 0 on CPU)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--wn3-compat", action="store_true",
                   help="Drop sp_hpa and snow_depth_m so inputs match WeatherNext 3 1-to-1")
    p.add_argument("--solar-film", action="store_true",
                   help="Add 5D suncalc solar direction features (sin/cos az, sin alt, sin noon_alt, lat) to FiLM")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())


if __name__ == "__main__":
    main()
