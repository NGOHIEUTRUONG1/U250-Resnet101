import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

ROOT = Path("/datastore/icc_fce/resnet_paper")
CONFIG_PATH = ROOT / "train_config.json"
STACK_PATH = ROOT / "STACK_CONTRACT.env"

EXPECTED = {
    "python": (3, 6, 8),
    "torch": "1.8.0+cu111",
    "torchvision": "0.9.0+cu111",
    "numpy": "1.19.5",
    "pillow": "8.4.0",
}

def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def version_gate():
    import PIL
    assert sys.version_info[:3] == EXPECTED["python"], sys.version
    assert torch.__version__ == EXPECTED["torch"], torch.__version__
    assert torchvision.__version__ == EXPECTED["torchvision"], torchvision.__version__
    assert np.__version__ == EXPECTED["numpy"], np.__version__
    assert PIL.__version__ == EXPECTED["pillow"], PIL.__version__
    assert torch.version.cuda == "11.1", torch.version.cuda
    assert torch.cuda.is_available(), "CUDA unavailable"
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # Physical GPU is selected by the cluster helper in job.sh.
    # Do not lock training to A100; keep the exact software stack gate instead.
    print("LIBRARY_GATE=PASS")
    print("GPU=", gpu)
    print("GPU_COMPUTE_CAPABILITY=", cap)
    print("CUDA_RUNTIME=", torch.version.cuda)

class CIFAR100(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        p = Path(root) / "cifar-100-python" / ("train" if split == "train" else "test")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        with open(str(p), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        self.x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.y = np.asarray(d[b"fine_labels"], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x[idx], mode="RGB")
        if self.transform:
            x = self.transform(x)
        return x, int(self.y[idx])

def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

def worker_init(worker_id):
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)

def make_model():
    m = models.resnet101(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, 100)
    return m

def split_indices(labels, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    labels = np.asarray(labels)
    for c in range(100):
        idx = np.where(labels == c)[0].copy()
        if len(idx) != 500:
            raise RuntimeError("class {} count={} != 500".format(c, len(idx)))
        rng.shuffle(idx)
        va.extend(idx[:50].tolist())
        tr.extend(idx[50:].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va

def lr_at(epoch, epochs, warmup):
    base = 0.1
    if epoch <= warmup:
        return base * epoch / warmup
    t = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * t))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    n = top1 = top5 = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += ce(out, y).item() * y.size(0)
        _, p5 = out.topk(5, 1)
        top1 += (p5[:, 0] == y).sum().item()
        top5 += p5.eq(y.view(-1, 1)).any(1).sum().item()
        n += y.size(0)
    return {"samples": n, "loss": loss_sum/n, "top1": top1/n, "top5": top5/n}

def cpu_state_dict(model):
    # Canonical portable artifact: no CUDA tensor is stored in best.pt.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def save_best(path, model, epoch, best_val, cfg, config_sha, stack_sha):
    torch.save({
        "state_dict": cpu_state_dict(model),
        "epoch": epoch,
        "best_val_top1": best_val,
        "architecture": "torchvision.models.resnet101(pretrained=False)",
        "num_classes": 100,
        "input_shape": [1, 3, 224, 224],
        "normalization_mean": cfg["model"]["normalization_mean"],
        "normalization_std": cfg["model"]["normalization_std"],
        "seed": cfg["training"]["seed"],
        "torch_base_version": "1.8.0",
        "torch_build_version": torch.__version__,
        "torchvision_base_version": "0.9.0",
        "torchvision_build_version": torchvision.__version__,
        "python_version": "3.6.8",
        "config_sha256": config_sha,
        "stack_contract_sha256": stack_sha,
        "checkpoint_tensor_device": "cpu",
    }, str(path))

def save_last(path, model, optimizer, epoch, best_val, cfg, config_sha, stack_sha):
    # LAST is only for recovery inside the same A100 training environment.
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_top1": best_val,
        "seed": cfg["training"]["seed"],
        "config_sha256": config_sha,
        "stack_contract_sha256": stack_sha,
    }, str(path))

def main():
    version_gate()

    cfg = json.loads(CONFIG_PATH.read_text())
    config_sha = sha256(CONFIG_PATH)
    stack_sha = sha256(STACK_PATH)

    tcfg = cfg["training"]
    seed = int(tcfg["seed"])
    seed_all(seed)

    out = cfg["outputs"]
    best = Path(out["best_checkpoint"])
    last = Path(out["last_checkpoint"])
    lock = Path(out["lock_file"])
    best_sha_file = Path(out["best_sha256"])
    metrics = Path(out["metrics_csv"])
    test_json = Path(out["test_json"])

    for p in [best.parent, metrics.parent]:
        p.mkdir(parents=True, exist_ok=True)

    if lock.exists():
        if not best.exists() or not best_sha_file.exists():
            raise RuntimeError("TRAIN_ONCE_LOCK exists but protected artifact missing")
        expected = best_sha_file.read_text().split()[0]
        actual = sha256(best)
        if actual != expected:
            raise RuntimeError("locked best.pt SHA256 mismatch")
        print("TRAIN_ONCE=ALREADY_COMPLETE")
        print("BEST_SHA256=", actual)
        return

    droot = cfg["dataset"]["extracted_root"]
    mean = cfg["model"]["normalization_mean"]
    std = cfg["model"]["normalization_std"]

    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ev_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    raw = CIFAR100(droot, "train")
    tr_idx, va_idx = split_indices(raw.y, seed)
    train_ds = Subset(CIFAR100(droot, "train", tr_tf), tr_idx)
    val_ds = Subset(CIFAR100(droot, "train", ev_tf), va_idx)
    test_ds = CIFAR100(droot, "test", ev_tf)

    gen = torch.Generator()
    gen.manual_seed(seed)

    workers = int(tcfg["workers"])
    batch = int(tcfg["batch_size"])
    dl_common = dict(
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, generator=gen, **dl_common
    )
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **dl_common)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, **dl_common)

    device = torch.device("cuda:0")
    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(tcfg["base_lr"]),
        momentum=float(tcfg["momentum"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    px, py = next(iter(train_loader))
    with torch.no_grad():
        po = model(px.to(device, non_blocking=True))
    assert tuple(po.shape) == (px.shape[0], 100), tuple(po.shape)
    print("PREFLIGHT=PASS", tuple(px.shape), tuple(po.shape), flush=True)

    epochs = int(tcfg["epochs"])
    warmup = int(tcfg["warmup_epochs"])
    best_val = -1.0

    with open(str(metrics), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr", "train_loss", "train_top1",
            "val_loss", "val_top1", "val_top5", "seconds"
        ])

        for epoch in range(1, epochs + 1):
            lr = lr_at(epoch, epochs, warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()
            t0 = time.time()
            n = correct = 0
            loss_sum = 0.0

            for step, (x, y) in enumerate(train_loader, 1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                b = y.size(0)
                n += b
                loss_sum += loss.item() * b
                correct += (pred.argmax(1) == y).sum().item()

                if step % 50 == 0:
                    print(
                        "EPOCH={}/{} STEP={}/{} LR={:.8f} LOSS={:.6f} TOP1={:.6f}".format(
                            epoch, epochs, step, len(train_loader),
                            lr, loss_sum/n, correct/n
                        ),
                        flush=True
                    )

            val = evaluate(model, val_loader, device)
            seconds = time.time() - t0
            tr_loss = loss_sum / n
            tr_top1 = correct / n

            improved = val["top1"] > best_val
            if improved:
                best_val = val["top1"]
                save_best(best, model, epoch, best_val, cfg, config_sha, stack_sha)

            save_last(last, model, optimizer, epoch, best_val, cfg, config_sha, stack_sha)

            w.writerow([
                epoch, lr, tr_loss, tr_top1,
                val["loss"], val["top1"], val["top5"], seconds
            ])
            f.flush()

            print(json.dumps({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_top1": tr_top1,
                "val": val,
                "best_val_top1": best_val,
                "best_updated": improved,
                "seconds": seconds
            }), flush=True)

    ckpt = torch.load(str(best), map_location="cpu")
    final_model = make_model()
    final_model.load_state_dict(ckpt["state_dict"], strict=True)
    final_model.to(device)

    test = evaluate(final_model, test_loader, device)
    best_sha = sha256(best)

    best_sha_file.write_text("{}  {}\n".format(best_sha, best.name))
    test_json.write_text(json.dumps({
        "best_checkpoint": str(best),
        "best_sha256": best_sha,
        "best_epoch": ckpt["epoch"],
        "best_val_top1": ckpt["best_val_top1"],
        "official_test": test,
        "config_sha256": config_sha,
        "stack_contract_sha256": stack_sha,
        "python": "3.6.8",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_tensor_device": "cpu",
    }, indent=2))

    lock.write_text(
        "TRAIN_ONCE_LOCK=ACTIVE\n"
        "best_checkpoint={}\n".format(best) +
        "best_sha256={}\n".format(best_sha) +
        "config_sha256={}\n".format(config_sha) +
        "stack_contract_sha256={}\n".format(stack_sha) +
        "best_epoch={}\n".format(ckpt["epoch"]) +
        "python=3.6.8\n"
        "torch_base=1.8.0\n"
        "torchvision_base=0.9.0\n"
        "checkpoint_tensor_device=cpu\n"
        "rule=DO_NOT_RETRAIN_FOR_1CU_OR_3CU\n"
        "rule=LOAD_STATE_DICT_STRICT_TRUE_ON_U250_SIDE\n"
    )

    os.chmod(str(best), 0o444)
    os.chmod(str(best_sha_file), 0o444)
    os.chmod(str(lock), 0o444)

    print("TRAINING=PASS")
    print("BEST_PT=", best)
    print("BEST_SHA256=", best_sha)
    print("TEST=", json.dumps(test))

if __name__ == "__main__":
    main()
