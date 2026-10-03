import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

NUM_CLASSES = 100
MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def dataloader_worker_init(worker_id):
    # Prevent each data-loader worker from spawning large BLAS thread pools.
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

class CIFAR100Pickle(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        fn = "train" if split == "train" else "test"
        p = Path(root) / "cifar-100-python" / fn
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        self.x = d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
        self.y = np.asarray(d[b"fine_labels"], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.fromarray(self.x[idx], mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])

def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def eval_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def make_model():
    # No pretrained download. Train from scratch for full provenance.
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def stratified_split(labels, seed):
    # CIFAR-100 has 500 official training examples per fine class.
    # We allocate exactly 450 train + 50 validation per class.
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    tr, va = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        if len(idx) != 500:
            raise RuntimeError(f"class {c} expected 500 train examples, got {len(idx)}")
        va.extend(idx[:50].tolist())
        tr.extend(idx[50:].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va

def epoch_lr(epoch, total_epochs):
    # 5 epoch linear warmup -> 0.1, then cosine decay.
    base = 0.1
    if epoch <= 5:
        return base * epoch / 5.0
    progress = (epoch - 5) / max(1, total_epochs - 5)
    return 0.5 * base * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()
    n = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0

    for x, y in loader:
        out = model(x)
        loss_sum += ce(out, y).item() * y.size(0)
        _, pred5 = out.topk(5, dim=1)
        top1 += (pred5[:,0] == y).sum().item()
        top5 += pred5.eq(y.view(-1,1)).any(dim=1).sum().item()
        n += y.size(0)

    return {
        "samples": n,
        "loss": loss_sum/n,
        "top1": top1/n,
        "top5": top5/n,
    }

def save_ckpt(path, model, optimizer, epoch, best_val_top1, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_top1": best_val_top1,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "cpu_threads": args.cpu_threads,
        "data_workers": args.workers,
        "architecture": "torchvision.models.resnet101(pretrained=False)",
        "num_classes": NUM_CLASSES,
        "input_shape": [1,3,224,224],
        "mean": list(MEAN),
        "std": list(STD),
        "optimizer_name": "SGD",
        "base_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "schedule": "5-epoch linear warmup + cosine decay",
        "train_split": 45000,
        "validation_split": 5000,
        "official_test_split": 10000,
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--best", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=70)
    ap.add_argument("--interop-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass

    print("TORCH_NUM_THREADS=", torch.get_num_threads(), flush=True)
    print("CPU_COUNT_OS=", os.cpu_count(), flush=True)
    print("CUDA_AVAILABLE=", torch.cuda.is_available(), flush=True)

    # Require CPU-only training.
    if torch.cuda.is_available():
        print("NOTE: CUDA is visible but will not be used.", flush=True)

    raw_train = CIFAR100Pickle(args.data_root, "train", None)
    tr_idx, va_idx = stratified_split(raw_train.y, args.seed)

    train_ds = Subset(CIFAR100Pickle(args.data_root, "train", train_transform()), tr_idx)
    val_ds = Subset(CIFAR100Pickle(args.data_root, "train", eval_transform()), va_idx)
    test_ds = CIFAR100Pickle(args.data_root, "test", eval_transform())

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=False,
    )
    if args.workers > 0:
        loader_kwargs["worker_init_fn"] = dataloader_worker_init

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=g,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        **loader_kwargs
    )

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Preflight one batch before the long run.
    print("===== PREFLIGHT ONE BATCH =====", flush=True)
    px, py = next(iter(train_loader))
    print(
        "PREFLIGHT_BATCH_SHAPE=%s LABEL_SHAPE=%s" %
        (tuple(px.shape), tuple(py.shape)),
        flush=True
    )
    model.eval()
    with torch.no_grad():
        po = model(px)
    print("PREFLIGHT_OUTPUT_SHAPE=%s" % (tuple(po.shape),), flush=True)
    if tuple(po.shape) != (px.shape[0], NUM_CLASSES):
        raise RuntimeError("unexpected ResNet101 preflight output shape")
    model.train()
    print("PREFLIGHT=PASS", flush=True)

    start_epoch = 1
    best_val = -1.0

    if args.resume and Path(args.last).is_file() and not args.force:
        ck = torch.load(args.last, map_location="cpu")
        model.load_state_dict(ck["state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        best_val = float(ck.get("best_val_top1", -1.0))
        print(
            f"RESUME=YES LAST_EPOCH={start_epoch-1} BEST_VAL_TOP1={best_val}",
            flush=True
        )

    if args.force:
        print("FORCE_RETRAIN=YES", flush=True)
        start_epoch = 1
        best_val = -1.0

    if start_epoch > args.epochs:
        print("TRAINING_ALREADY_COMPLETE=YES", flush=True)
    else:
        append_csv = Path(args.metrics_csv).is_file() and start_epoch > 1
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)

        with open(args.metrics_csv, "a" if append_csv else "w", newline="") as f:
            wr = csv.writer(f)
            if not append_csv:
                wr.writerow([
                    "epoch","lr",
                    "train_loss","train_top1",
                    "val_loss","val_top1","val_top5",
                    "epoch_seconds"
                ])

            for epoch in range(start_epoch, args.epochs+1):
                lr = epoch_lr(epoch, args.epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                model.train()
                t0 = time.time()
                seen = 0
                train_loss_sum = 0.0
                train_ok = 0

                for step, (x, y) in enumerate(train_loader, 1):
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    b = y.size(0)
                    seen += b
                    train_loss_sum += loss.item() * b
                    train_ok += (out.argmax(1) == y).sum().item()

                    if step % 25 == 0:
                        print(
                            f"EPOCH={epoch}/{args.epochs} "
                            f"STEP={step}/{len(train_loader)} "
                            f"LR={lr:.8f} "
                            f"TRAIN_LOSS={train_loss_sum/seen:.6f} "
                            f"TRAIN_TOP1={train_ok/seen:.6f}",
                            flush=True
                        )

                val = evaluate(model, val_loader)
                epoch_s = time.time() - t0

                train_loss = train_loss_sum / seen
                train_top1 = train_ok / seen

                improved = val["top1"] > best_val
                if improved:
                    best_val = val["top1"]
                    save_ckpt(
                        args.best, model, optimizer, epoch, best_val, args
                    )

                save_ckpt(
                    args.last, model, optimizer, epoch, best_val, args
                )

                wr.writerow([
                    epoch, lr,
                    train_loss, train_top1,
                    val["loss"], val["top1"], val["top5"],
                    epoch_s
                ])
                f.flush()

                print(json.dumps({
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_top1": train_top1,
                    "val_loss": val["loss"],
                    "val_top1": val["top1"],
                    "val_top5": val["top5"],
                    "best_val_top1": best_val,
                    "best_updated": improved,
                    "epoch_seconds": epoch_s,
                }), flush=True)

    if not Path(args.best).is_file():
        raise RuntimeError("best checkpoint missing")

    best = torch.load(args.best, map_location="cpu")
    model = make_model()
    model.load_state_dict(best["state_dict"], strict=True)

    test = evaluate(model, test_loader)

    result = {
        "best_checkpoint": args.best,
        "best_epoch": int(best["epoch"]),
        "best_val_top1": float(best["best_val_top1"]),
        "official_test": test,
        "dataset": "CIFAR-100",
        "train_samples": 45000,
        "validation_samples": 5000,
        "test_samples": 10000,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "cpu_threads": args.cpu_threads,
        "data_workers": args.workers,
        "batch_size": args.batch,
        "input_size": [3,224,224],
        "mean": MEAN,
        "std": STD,
    }

    with open(args.test_json, "w") as f:
        json.dump(result, f, indent=2)

    print("FINAL_TEST=", json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
