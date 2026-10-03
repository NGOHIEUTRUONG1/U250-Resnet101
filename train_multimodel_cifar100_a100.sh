#!/usr/bin/env bash
# Self-contained A100/MPS/Slurm training campaign for Vitis-AI 2.0 / U250.
#
# Login-node use:
#   sbatch options may be supplied through SLURM_PARTITION / SLURM_ACCOUNT.
#   This is the training engine. Use prepare_data_env_a100.sh first, then
#   submit jobs through job.sh.
#
# Useful overrides:
#   PROJECT_ROOT=/datastore/icc_fce/resnet_paper_multimodel
#   MAX_PARALLEL=1 OFFLINE=0
#   SLURM_PARTITION=<site-partition> SLURM_ACCOUNT=<site-account>
#
# The distributable consists of this file only.  At runtime it materializes its
# embedded Python driver inside PROJECT_ROOT/provenance.
# The existing build_resnet101_xmodel... script remains ResNet-101-only and has
# a historical checkpoint SHA hard-coded; do not use it unchanged for other
# architectures or newly trained checkpoints.

set -Eeuo pipefail
umask 022

readonly SCRIPT_VERSION="2.0.2-a100-vai20"
readonly CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
readonly CIFAR_MD5="eb9058c3a382ffc7106e4002c42a8d85"
readonly PYTHON_VERSION="3.6.8"
readonly PIP_VERSION="21.3.1"
readonly SETUPTOOLS_VERSION="59.6.0"
readonly WHEEL_VERSION="0.37.1"
readonly TORCH_VERSION="1.8.0+cu111"
readonly TORCHVISION_VERSION="0.9.0+cu111"
readonly NUMPY_VERSION="1.19.5"
readonly PILLOW_VERSION="8.4.0"
readonly ZSTANDARD_VERSION="0.17.0"

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PROJECT_ROOT:-/datastore/icc_fce/resnet_paper_multimodel}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OFFLINE="${OFFLINE:-0}"
CONDA_ROOT="${CONDA_ROOT:-/datastore/icc_fce/miniconda3}"
WHEELHOUSE="${WHEELHOUSE:-$PROJECT_ROOT/wheelhouse}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
WORKER_CPUS="${WORKER_CPUS:-4}"
WORKER_MEM="${WORKER_MEM:-32G}"
WORKER_TIME="${WORKER_TIME:-72:00:00}"
SLURM_GRES="${SLURM_GRES:-mps:1}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
REQUIRED_VRAM="${REQUIRED_VRAM:-25000}"
GPU_CHECK_HELPER="${GPU_CHECK_HELPER:-}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-600}"

ENV_PREFIX="$PROJECT_ROOT/runtime/vai20_a100_py368"
DRIVER="$PROJECT_ROOT/provenance/multimodel_driver.py"
DATA_ARCHIVE="$PROJECT_ROOT/cache/datasets/cifar-100-python.tar.gz"
DATA_ROOT="$PROJECT_ROOT/data"
RESULTS_ROOT="$PROJECT_ROOT/results"
LOG_ROOT="$PROJECT_ROOT/logs"
LOCK_ROOT="$PROJECT_ROOT/locks"

export PROJECT_ROOT MAX_PARALLEL OFFLINE CONDA_ROOT WHEELHOUSE TORCH_HOME
export WORKER_CPUS WORKER_MEM WORKER_TIME SLURM_GRES SLURM_PARTITION SLURM_ACCOUNT
export REQUIRED_VRAM GPU_CHECK_HELPER
export LOCK_TIMEOUT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_uint() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got: $2"
}

init_dirs() {
  mkdir -p "$PROJECT_ROOT" "$PROJECT_ROOT/runtime" "$PROJECT_ROOT/provenance" \
    "$PROJECT_ROOT/cache/datasets" "$TORCH_HOME" "$DATA_ROOT" "$RESULTS_ROOT" \
    "$LOG_ROOT" "$LOCK_ROOT" "$PROJECT_ROOT/packages"
}

write_driver() {
  init_dirs
  local tmp
  tmp="$(mktemp "$PROJECT_ROOT/provenance/.multimodel_driver.XXXXXX")"
  cat >"$tmp" <<'PY'
#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import shutil
import socket
import statistics
import sys
import tarfile
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

MODELS = [
    "resnet18",
    "resnet50",
    "resnet101",
    "mobilenet_v2",
    "densenet121",
    "vgg16",
    "shufflenet_v2_x1_0",
    "squeezenet1_1",
]
SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924]
SPLIT_SEED = 20260920
NUM_CLASSES = 100
INPUT_SIZE = 224
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
T_CRIT_DF4_975 = 2.7764451051977987
METRICS_HEADER = [
    "epoch", "lr", "train_loss", "train_top1", "val_loss", "val_top1",
    "val_top5", "seconds", "best_updated",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(torch, value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, str(tmp))
    os.replace(tmp, path)


def task_mapping(task_id):
    if not 0 <= task_id < len(MODELS) * len(SEEDS):
        raise ValueError("task id must be in [0, 39]")
    model = MODELS[task_id // len(SEEDS)]
    seed = SEEDS[task_id % len(SEEDS)]
    return model, seed


def run_dir(root, model, seed):
    return Path(root) / "results" / model / ("seed_{}".format(seed))


def cpu_state_dict(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def imports():
    import numpy as np
    import PIL
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import models, transforms
    return np, PIL, torch, torchvision, nn, DataLoader, Dataset, Subset, models, transforms


def model_spec(models, name):
    # URLs are the torchvision 0.9.0 pretrained artifacts.  They are recorded
    # and hashed after download; no modern WeightsEnum API is used.
    specs = {
        "resnet18": (models.resnet18, "https://download.pytorch.org/models/resnet18-5c106cde.pth", "fc"),
        "resnet50": (models.resnet50, "https://download.pytorch.org/models/resnet50-19c8e357.pth", "fc"),
        "resnet101": (models.resnet101, "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", "fc"),
        "mobilenet_v2": (models.mobilenet_v2, "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", "classifier.1"),
        "densenet121": (models.densenet121, "https://download.pytorch.org/models/densenet121-a639ec97.pth", "classifier"),
        "vgg16": (models.vgg16, "https://download.pytorch.org/models/vgg16-397923af.pth", "classifier.6"),
        "shufflenet_v2_x1_0": (
            models.shufflenet_v2_x1_0,
            "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "fc",
        ),
        "squeezenet1_1": (
            models.squeezenet1_1,
            "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
            "classifier.1.conv",
        ),
    }
    return specs[name]


def build_model(models, nn, name, pretrained):
    constructor, pretrained_url, head = model_spec(models, name)
    model = constructor(pretrained=pretrained, progress=True)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif head == "classifier":
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif head == "classifier.1":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif head == "classifier.6":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    elif head == "classifier.1.conv":
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
        model.num_classes = NUM_CLASSES
    else:
        raise AssertionError(head)
    return model, pretrained_url, head


def set_determinism(np, torch, seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


def worker_init(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataset_class(Dataset, np):
    from PIL import Image

    class CIFAR100Pickle(Dataset):
        def __init__(self, root, split, transform=None):
            self.transform = transform
            source = Path(root) / "cifar-100-python" / ("train" if split == "train" else "test")
            if not source.is_file():
                raise FileNotFoundError(str(source))
            with open(source, "rb") as f:
                obj = pickle.load(f, encoding="bytes")
            self.data = obj[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.targets = np.asarray(obj[b"fine_labels"], dtype=np.int64)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            image = Image.fromarray(self.data[index], mode="RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, int(self.targets[index])

    return CIFAR100Pickle


def stratified_indices(np, labels):
    rng = np.random.RandomState(SPLIT_SEED)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for class_id in range(NUM_CLASSES):
        idx = np.where(labels == class_id)[0].copy()
        if len(idx) != 500:
            raise RuntimeError("class {} has {} samples, expected 500".format(class_id, len(idx)))
        rng.shuffle(idx)
        val_idx.extend(idx[:50].tolist())
        train_idx.extend(idx[50:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if len(train_idx) != 45000 or len(val_idx) != 5000:
        raise AssertionError("invalid split sizes")
    return train_idx, val_idx


def lr_at(epoch, epochs=100, warmup=5, base_lr=0.01):
    if epoch <= warmup:
        return base_lr * epoch / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def capture_rng(np, torch, loader_generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng(np, torch, state, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])


def environment(torch, torchvision, np, PIL):
    import torchaudio
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": props.total_memory,
        }
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    return {
        "captured_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "slurm": slurm,
    }


def evaluate(torch, nn, model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    samples = top1 = top5 = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            pred = logits.topk(5, dim=1).indices
            top1 += pred[:, 0].eq(labels).sum().item()
            top5 += pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            samples += labels.size(0)
    return {
        "samples": samples,
        "loss": loss_sum / samples,
        "top1": top1 / samples,
        "top5": top5 / samples,
        "top1_correct": top1,
        "top5_correct": top5,
    }


def checkpoint_valid(directory):
    directory = Path(directory)
    required = [
        "COMPLETE", "best.pt", "best_state_dict.pt", "best.pt.sha256",
        "config.json", "train_config.json", "STACK_CONTRACT.env",
        "builder_inputs.env",
        "metrics.csv", "test_metrics.json",
        "pretrained_provenance.json", "environment.json",
    ]
    if any(not (directory / name).is_file() for name in required):
        return False
    line = (directory / "best.pt.sha256").read_text(encoding="utf-8").strip().split()
    return len(line) >= 1 and line[0] == sha256_file(directory / "best.pt")


def reconcile_metrics(path, last_state):
    """Make epoch CSV consistent with the last atomic recovery checkpoint."""
    path = Path(path)
    last_epoch = int(last_state["epoch"])
    rows_by_epoch = {}
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != METRICS_HEADER:
                raise RuntimeError("metrics.csv header mismatch")
            for row in reader:
                epoch = int(row["epoch"])
                if epoch <= last_epoch:
                    rows_by_epoch[epoch] = [row[name] for name in METRICS_HEADER]
    saved_row = last_state.get("metrics_row")
    if saved_row is not None:
        rows_by_epoch[last_epoch] = [str(value) for value in saved_row]
    expected = list(range(1, last_epoch + 1))
    if sorted(rows_by_epoch) != expected:
        raise RuntimeError("metrics.csv cannot be reconciled through epoch {}".format(last_epoch))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_HEADER)
        for epoch in expected:
            writer.writerow(rows_by_epoch[epoch])


def cmd_map(args):
    if args.task_id is None:
        for task_id in range(40):
            model, seed = task_mapping(task_id)
            print("{}\t{}\t{}".format(task_id, model, seed))
    else:
        model, seed = task_mapping(args.task_id)
        print(json.dumps({"task_id": args.task_id, "model": model, "seed": seed}, sort_keys=True))


def cmd_prepare(args):
    import torchaudio
    np, PIL, torch, torchvision, nn, _, _, _, models, _ = imports()
    if sys.version_info[:3] != (3, 6, 8):
        raise RuntimeError("Python version mismatch: {}".format(sys.version))
    if torch.__version__ != "1.8.0+cu111":
        raise RuntimeError("torch version mismatch: {}".format(torch.__version__))
    if torchvision.__version__ != "0.9.0+cu111":
        raise RuntimeError("torchvision version mismatch: {}".format(torchvision.__version__))
    if torchaudio.__version__.split("+")[0] != "0.8.0":
        raise RuntimeError("torchaudio version mismatch: {}".format(torchaudio.__version__))
    if np.__version__ != "1.19.5" or PIL.__version__ != "8.4.0":
        raise RuntimeError("NumPy/Pillow version mismatch: {}/{}".format(np.__version__, PIL.__version__))
    _, pretrained_url, _ = build_model(models, nn, args.model, pretrained=True)
    checkpoint_name = Path(pretrained_url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise RuntimeError("pretrained checkpoint was not cached: {}".format(checkpoint_path))
    print(json.dumps({
        "model": args.model,
        "pretrained_api": "pretrained=True",
        "url": pretrained_url,
        "cache_path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
    }, sort_keys=True))


def cmd_validate(args):
    if not checkpoint_valid(args.run_dir):
        raise SystemExit(1)
    print("VALID=PASS {}".format(args.run_dir))


def cmd_verify_prepared(args):
    root = Path(args.root)
    for model in MODELS:
        provenance_path = root / "provenance" / ("pretrained_{}.json".format(model))
        if not provenance_path.is_file():
            raise RuntimeError("missing pretrained provenance: {}".format(provenance_path))
        item = json.loads(provenance_path.read_text(encoding="utf-8"))
        cached = Path(item["cache_path"])
        if not cached.is_file():
            raise RuntimeError("pretrained cache is missing: {}".format(cached))
        actual = sha256_file(cached)
        if actual != item["sha256"]:
            raise RuntimeError("pretrained SHA mismatch for {}: {} != {}".format(
                model, actual, item["sha256"]))
    print("PREPARED_INPUTS=PASS models=8")


def cmd_train(args):
    import torchaudio
    np, PIL, torch, torchvision, nn, DataLoader, Dataset, Subset, models, transforms = imports()
    root = Path(args.root)
    directory = run_dir(root, args.model, args.seed)
    directory.mkdir(parents=True, exist_ok=True)
    if checkpoint_valid(directory):
        print("RUN_ALREADY_COMPLETE model={} seed={}".format(args.model, args.seed))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name.upper():
        raise RuntimeError("expected NVIDIA A100, got {!r}".format(gpu_name))
    if sys.version_info[:3] != (3, 6, 8):
        raise RuntimeError("Python version mismatch: {}".format(sys.version))
    if torch.__version__ != "1.8.0+cu111" or torchvision.__version__ != "0.9.0+cu111":
        raise RuntimeError("training framework mismatch: torch={} torchvision={}".format(
            torch.__version__, torchvision.__version__))
    if torchaudio.__version__.split("+")[0] != "0.8.0":
        raise RuntimeError("torchaudio mismatch: {}".format(torchaudio.__version__))
    if np.__version__ != "1.19.5" or PIL.__version__ != "8.4.0":
        raise RuntimeError("NumPy/Pillow mismatch: {}/{}".format(np.__version__, PIL.__version__))

    set_determinism(np, torch, args.seed)
    device = torch.device("cuda:0")
    model, pretrained_url, head = build_model(models, nn, args.model, pretrained=True)
    model.to(device)

    cached_weight = Path(torch.hub.get_dir()) / "checkpoints" / Path(pretrained_url).name
    provenance = {
        "pretrained_api": "pretrained=True",
        "url": pretrained_url,
        "cached_file": str(cached_weight),
        "sha256": sha256_file(cached_weight),
    }
    atomic_json(directory / "pretrained_provenance.json", provenance)

    micro_batch_by_model = {
        "resnet18": 128, "resnet50": 128, "resnet101": 64,
        "mobilenet_v2": 256, "densenet121": 64, "vgg16": 32,
        "shufflenet_v2_x1_0": 256, "squeezenet1_1": 256,
    }
    micro_batch = micro_batch_by_model[args.model]
    accumulation_steps = 256 // micro_batch
    config = {
        "campaign": "a100_vai20_cifar100_multimodel_q1",
        "model": args.model,
        "constructor": "torchvision.models.{}".format(args.model),
        "classifier_location": head,
        "pretrained": provenance,
        "num_classes": NUM_CLASSES,
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "normalization_mean": list(MEAN),
        "normalization_std": list(STD),
        "dataset": {
            "name": "CIFAR-100",
            "archive_md5": "eb9058c3a382ffc7106e4002c42a8d85",
            "train_samples": 45000,
            "validation_samples": 5000,
            "test_samples": 10000,
            "split_seed": SPLIT_SEED,
            "split": "stratified_450_train_50_val_per_class",
        },
        "training": {
            "seed": args.seed,
            "epochs": 100,
            "effective_batch_size": 256,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation_steps,
            "train_workers": min(3, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)),
            "evaluation_workers": 1,
            "optimizer": "SGD",
            "base_lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 5,
            "schedule": "linear_warmup_then_cosine",
            "amp": True,
            "deterministic_algorithms": True,
            "best_metric": "validation_top1",
        },
        "stack_contract": {
            "python": "3.6.8", "torch": "1.8.0+cu111",
            "torchvision": "0.9.0+cu111", "numpy": "1.19.5", "pillow": "8.4.0",
            "torchaudio": "0.8.0",
            "u250_torch": "1.8.0+cpu", "u250_torchvision": "0.9.0+cpu",
            "vitis_ai": "2.0",
        },
        "requires_u250_adapter": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(directory / "config.json", config)
    deployment_config = {
        "experiment": "{}_cifar100_seed{}".format(args.model, args.seed),
        "dataset": {
            "name": "CIFAR-100",
            "official_md5": "eb9058c3a382ffc7106e4002c42a8d85",
            "train_samples": 45000,
            "validation_samples": 5000,
            "test_samples": 10000,
            "split": "stratified_450_train_50_val_per_class_from_official_train",
        },
        "model": {
            "architecture": "torchvision.models.{}".format(args.model),
            # False is the deployment reconstruction constructor.  Training
            # initialization is recorded separately and remains truthful.
            "pretrained": False,
            "training_initialization": "torchvision_0.9.0_imagenet_pretrained",
            "num_classes": NUM_CLASSES,
            "input_size": [3, INPUT_SIZE, INPUT_SIZE],
            "normalization_mean": list(MEAN),
            "normalization_std": list(STD),
        },
        "training": config["training"],
        "outputs": {
            "best_checkpoint": str(directory / "best.pt"),
            "best_state_dict": str(directory / "best_state_dict.pt"),
            "best_sha256": str(directory / "best.pt.sha256"),
        },
    }
    atomic_json(directory / "train_config.json", deployment_config)
    atomic_text(directory / "STACK_CONTRACT.env", "\n".join([
        "PYTHON_VERSION=3.6.8",
        "TORCH_BASE_VERSION=1.8.0",
        "TORCHVISION_BASE_VERSION=0.9.0",
        "TORCHAUDIO_BASE_VERSION=0.8.0",
        "NUMPY_VERSION=1.19.5",
        "PILLOW_VERSION=8.4.0",
        "A100_TORCH_VERSION=1.8.0+cu111",
        "A100_TORCHVISION_VERSION=0.9.0+cu111",
        "A100_CUDA_RUNTIME=11.1",
        "U250_TORCH_VERSION=1.8.0+cpu",
        "U250_TORCHVISION_VERSION=0.9.0+cpu",
        "VITIS_AI_RELEASE=2.0",
        "MODEL_IMPL=torchvision.models.{}(pretrained=False)".format(args.model),
        "NUM_CLASSES=100",
        "INPUT_SHAPE=1,3,224,224",
        "CHECKPOINT_FORMAT=CPU_STATE_DICT",
        "CHECKPOINT_LOAD_POLICY=STRICT_TRUE",
        "TRAIN_INITIALIZATION=IMAGENET_PRETRAINED",
        "TRAIN_SEED={}".format(args.seed),
        "TRAIN_POLICY=FIVE_SEED_CAMPAIGN",
        "",
    ]))
    atomic_json(directory / "environment.json", environment(torch, torchvision, np, PIL))

    CIFAR100Pickle = make_dataset_class(Dataset, np)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    raw_train = CIFAR100Pickle(root / "data", "train")
    train_idx, val_idx = stratified_indices(np, raw_train.targets)
    train_ds = Subset(CIFAR100Pickle(root / "data", "train", train_tf), train_idx)
    val_ds = Subset(CIFAR100Pickle(root / "data", "train", eval_tf), val_idx)
    test_ds = CIFAR100Pickle(root / "data", "test", eval_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_workers = int(config["training"]["train_workers"])
    evaluation_workers = int(config["training"]["evaluation_workers"])
    train_common = {
        "num_workers": train_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
        # Recreate workers each epoch so a saved DataLoader generator state gives
        # deterministic augmentation after an interrupted job resumes.
        "persistent_workers": False,
    }
    evaluation_common = {
        "num_workers": evaluation_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=True, generator=generator,
        drop_last=False, **train_common
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, drop_last=False, **evaluation_common
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, drop_last=False, **evaluation_common
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    last_path = directory / "last.pt"
    metrics_path = directory / "metrics.csv"
    start_epoch = 1
    best_val = -1.0
    best_epoch = 0

    if last_path.is_file():
        state = torch.load(str(last_path), map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_val = float(state["best_val_top1"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng(np, torch, state["rng"], generator)
        reconcile_metrics(metrics_path, state)
        print("RESUME epoch={} best_val_top1={:.6f}".format(start_epoch, best_val), flush=True)

    new_metrics = not metrics_path.exists() or start_epoch == 1
    metrics_file = open(metrics_path, "w" if new_metrics else "a", newline="", encoding="utf-8")
    writer = csv.writer(metrics_file)
    if new_metrics:
        writer.writerow(METRICS_HEADER)
        metrics_file.flush()

    for epoch in range(start_epoch, 101):
        lr = lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        start = time.time()
        count = correct = 0
        loss_sum = 0.0
        optimizer.zero_grad()
        current_group_steps = accumulation_steps
        for step, (images, labels) in enumerate(train_loader, 1):
            if (step - 1) % accumulation_steps == 0:
                current_group_steps = min(accumulation_steps, len(train_loader) - step + 1)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss / current_group_steps).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            batch = labels.size(0)
            count += batch
            loss_sum += loss.item() * batch
            correct += logits.detach().argmax(dim=1).eq(labels).sum().item()

        val = evaluate(torch, nn, model, val_loader, device)
        elapsed = time.time() - start
        improved = val["top1"] > best_val
        if improved:
            best_val = val["top1"]
            best_epoch = epoch
            best_payload = {
                "state_dict": cpu_state_dict(model),
                "architecture": "torchvision.models.{}(pretrained=False)".format(args.model),
                "num_classes": NUM_CLASSES,
                "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
                "normalization_mean": list(MEAN),
                "normalization_std": list(STD),
                "seed": args.seed,
                "split_seed": SPLIT_SEED,
                "epoch": epoch,
                "best_val_top1": val["top1"],
                "best_val_top5": val["top5"],
                "requires_u250_adapter": False,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "checkpoint_tensor_device": "cpu",
            }
            atomic_torch_save(torch, best_payload, directory / "best.pt")

        metrics_row = [
            epoch, "{:.12g}".format(lr), "{:.12g}".format(loss_sum / count),
            "{:.12g}".format(correct / count), "{:.12g}".format(val["loss"]),
            "{:.12g}".format(val["top1"]), "{:.12g}".format(val["top5"]),
            "{:.6f}".format(elapsed), int(improved),
        ]
        last_payload = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_top1": best_val,
            "rng": capture_rng(np, torch, generator),
            "metrics_row": metrics_row,
        }
        atomic_torch_save(torch, last_payload, last_path)
        writer.writerow(metrics_row)
        metrics_file.flush()
        print(json.dumps({
            "epoch": epoch, "lr": lr, "train_loss": loss_sum / count,
            "train_top1": correct / count, "validation": val,
            "best_val_top1": best_val, "best_epoch": best_epoch,
            "seconds": elapsed,
        }, sort_keys=True), flush=True)

    metrics_file.close()
    best = torch.load(str(directory / "best.pt"), map_location="cpu")
    verifier, _, _ = build_model(models, nn, args.model, pretrained=False)
    result = verifier.load_state_dict(best["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint reload failed")
    verifier.to(device)
    test = evaluate(torch, nn, verifier, test_loader, device)
    if test["samples"] != 10000:
        raise RuntimeError("test sample count {}, expected 10000".format(test["samples"]))

    atomic_torch_save(torch, best["state_dict"], directory / "best_state_dict.pt")
    checkpoint_sha = sha256_file(directory / "best.pt")
    atomic_text(directory / "best.pt.sha256", "{}  best.pt\n".format(checkpoint_sha))
    atomic_text(directory / "builder_inputs.env", "\n".join([
        "MODEL_NAME={}".format(args.model),
        "CHECKPOINT={}".format(directory / "best.pt"),
        "CHECKPOINT_SHA256={}".format(checkpoint_sha),
        "CHECKPOINT_SHA_FILE={}".format(directory / "best.pt.sha256"),
        "TRAIN_CONFIG={}".format(directory / "train_config.json"),
        "STACK_CONTRACT={}".format(directory / "STACK_CONTRACT.env"),
        "DATASET_MD5=eb9058c3a382ffc7106e4002c42a8d85",
        "",
    ]))
    report = {
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_val_top1": best["best_val_top1"],
        "best_val_top5": best["best_val_top5"],
        "official_test": test,
        "best_checkpoint_sha256": checkpoint_sha,
        "requires_u250_adapter": False,
        "completed_at_utc": utc_now(),
    }
    atomic_json(directory / "test_metrics.json", report)
    if not checkpoint_valid(directory):
        # COMPLETE is deliberately still absent; validate the other fields directly.
        for name in ["best.pt", "best_state_dict.pt", "best.pt.sha256", "config.json",
                     "train_config.json", "STACK_CONTRACT.env",
                     "builder_inputs.env",
                     "metrics.csv", "test_metrics.json", "pretrained_provenance.json",
                     "environment.json"]:
            if not (directory / name).is_file():
                raise RuntimeError("missing final artifact: {}".format(name))
    atomic_text(directory / "COMPLETE", json.dumps({
        "status": "PASS", "model": args.model, "seed": args.seed,
        "best_checkpoint_sha256": checkpoint_sha,
        "completed_at_utc": utc_now(),
    }, sort_keys=True) + "\n")
    if not checkpoint_valid(directory):
        raise RuntimeError("post-completion validation failed")
    if last_path.exists():
        last_path.unlink()
    print("TRAINING=PASS model={} seed={} sha256={}".format(args.model, args.seed, checkpoint_sha))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_tree_to_tar(tar, source, arc_prefix):
    source = Path(source)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            tar.add(path, arcname=str(Path(arc_prefix) / path.relative_to(source)), recursive=False)


def cmd_aggregate(args):
    root = Path(args.root)
    rows = []
    environments = []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            directory = run_dir(root, model, seed)
            if not checkpoint_valid(directory):
                missing.append(str(directory))
                continue
            report = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            environments.append(json.loads((directory / "environment.json").read_text(encoding="utf-8")))
            rows.append({
                "model": model,
                "seed": seed,
                "best_epoch": report["best_epoch"],
                "validation_top1": report["best_val_top1"],
                "validation_top5": report["best_val_top5"],
                "test_top1": report["official_test"]["top1"],
                "test_top5": report["official_test"]["top5"],
                "test_top1_correct": report["official_test"]["top1_correct"],
                "test_top5_correct": report["official_test"]["top5_correct"],
                "test_samples": report["official_test"]["samples"],
                "checkpoint_sha256": report["best_checkpoint_sha256"],
                "requires_u250_adapter": config["requires_u250_adapter"],
            })
    if missing:
        raise RuntimeError("campaign incomplete; missing/invalid runs:\n" + "\n".join(missing))
    if len(rows) != 40:
        raise AssertionError("expected 40 rows, got {}".format(len(rows)))

    all_fields = list(rows[0].keys())
    write_csv(root / "all_runs.csv", rows, all_fields)
    summary = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        out = {"model": model, "n": len(model_rows)}
        for metric in ["validation_top1", "validation_top5", "test_top1", "test_top5"]:
            values = [float(r[metric]) for r in model_rows]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            ci = T_CRIT_DF4_975 * std / math.sqrt(len(values))
            out[metric + "_mean"] = mean
            out[metric + "_std"] = std
            out[metric + "_ci95_half_width"] = ci
        out["requires_u250_adapter"] = False
        summary.append(out)
    summary_fields = list(summary[0].keys())
    write_csv(root / "model_summary.csv", summary, summary_fields)
    atomic_json(root / "model_summary.json", {
        "campaign": "a100_vai20_cifar100_multimodel_q1",
        "generated_at_utc": utc_now(),
        "confidence_interval": "two-sided Student-t, 95%, df=4, t=2.7764451052",
        "models": summary,
    })
    unique_env = {}
    for item in environments:
        key = json.dumps({
            "hostname": item.get("hostname"), "torch": item.get("torch"),
            "torchvision": item.get("torchvision"), "gpu": item.get("gpu"),
        }, sort_keys=True)
        unique_env[key] = item
    atomic_json(root / "environment_manifest.json", {
        "generated_at_utc": utc_now(),
        "run_count": len(environments),
        "unique_environments": list(unique_env.values()),
        "dataset": {"name": "CIFAR-100", "md5": "eb9058c3a382ffc7106e4002c42a8d85"},
        "training_stack": {
            "python": "3.6.8", "torch": "1.8.0+cu111",
            "torchvision": "0.9.0+cu111", "numpy": "1.19.5", "pillow": "8.4.0",
        },
    })

    manifest_path = root / "artifact_manifest.sha256"
    manifest_candidates = []
    for path in sorted((root / "results").rglob("*")):
        if path.is_file():
            manifest_candidates.append(path)
    for name in ["all_runs.csv", "model_summary.csv", "model_summary.json", "environment_manifest.json"]:
        manifest_candidates.append(root / name)
    provenance_root = root / "provenance"
    for path in sorted(provenance_root.rglob("*")):
        if path.is_file():
            manifest_candidates.append(path)
    lines = []
    for path in sorted(set(manifest_candidates)):
        lines.append("{}  {}".format(sha256_file(path), path.relative_to(root)))
    atomic_text(manifest_path, "\n".join(lines) + "\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package = root / "packages" / ("a100_vai20_cifar100_multimodel_weights_{}.tar.zst".format(timestamp))
    package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="multimodel_", suffix=".tar", dir=package.parent, delete=False) as f:
        tar_path = Path(f.name)
    try:
        with tarfile.open(tar_path, "w") as tar:
            add_tree_to_tar(tar, root / "results", "results")
            for name in ["all_runs.csv", "model_summary.csv", "model_summary.json",
                         "environment_manifest.json", "artifact_manifest.sha256"]:
                tar.add(root / name, arcname=name, recursive=False)
            add_tree_to_tar(tar, provenance_root, "provenance")
        import zstandard
        compressor = zstandard.ZstdCompressor(level=10, threads=-1)
        tmp_package = package.with_name(package.name + ".tmp")
        with open(tar_path, "rb") as src, open(tmp_package, "wb") as dst:
            compressor.copy_stream(src, dst)
        os.replace(tmp_package, package)
    finally:
        if tar_path.exists():
            tar_path.unlink()
    atomic_text(str(package) + ".sha256", "{}  {}\n".format(sha256_file(package), package.name))
    print("AGGREGATION=PASS")
    print("PACKAGE={}".format(package))
    print("PACKAGE_SHA256={}".format(sha256_file(package)))


def build_legacy_model(models, nn, name):
    """Construct only architectures present in torchvision 0.9."""
    constructors = {
        "resnet18": (models.resnet18, "fc"),
        "resnet50": (models.resnet50, "fc"),
        "resnet101": (models.resnet101, "fc"),
        "mobilenet_v2": (models.mobilenet_v2, "classifier.1"),
        "densenet121": (models.densenet121, "classifier"),
        "vgg16": (models.vgg16, "classifier.6"),
        "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, "fc"),
        "squeezenet1_1": (models.squeezenet1_1, "classifier.1.conv"),
    }
    constructor, head = constructors[name]
    model = constructor(pretrained=False)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif head == "classifier":
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif head == "classifier.1":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif head == "classifier.6":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    elif head == "classifier.1.conv":
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
        model.num_classes = NUM_CLASSES
    return model


def cmd_u250_preflight(args):
    import torch
    import torchvision
    from torch import nn
    from torchvision import models

    if torchvision.__version__.split("+")[0] != "0.9.0":
        raise RuntimeError("U250 preflight requires torchvision 0.9.0, got {}".format(
            torchvision.__version__))
    root = Path(args.root)
    results = []
    failures = []
    for model_name in MODELS:
        for seed in SEEDS:
            checkpoint = run_dir(root, model_name, seed) / "best_state_dict.pt"
            item = {"model": model_name, "seed": seed, "checkpoint": str(checkpoint)}
            try:
                if not checkpoint.is_file():
                    raise FileNotFoundError(str(checkpoint))
                state_dict = torch.load(str(checkpoint), map_location="cpu")
                model = build_legacy_model(models, nn, model_name)
                loaded = model.load_state_dict(state_dict, strict=True)
                if getattr(loaded, "missing_keys", []) or getattr(loaded, "unexpected_keys", []):
                    raise RuntimeError("strict state_dict load returned incompatible keys")
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
                if tuple(output.shape) != (1, NUM_CLASSES):
                    raise RuntimeError("unexpected output shape {}".format(tuple(output.shape)))
                item.update({"status": "PASS", "state_dict_sha256": sha256_file(checkpoint)})
            except Exception as exc:
                item.update({"status": "FAIL", "error": repr(exc)})
                failures.append(item)
            results.append(item)
    report = {
        "checked_at_utc": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "load_policy": "strict=True",
        "expected_pass_count": 40,
        "pass_count": sum(item["status"] == "PASS" for item in results),
        "results": results,
    }
    atomic_json(root / "u250_preflight.json", report)
    if failures:
        raise RuntimeError("U250_PREFLIGHT=FAIL count={} report={}".format(
            len(failures), root / "u250_preflight.json"))
    print("U250_PREFLIGHT=PASS checkpoints=40")
    print("REPORT={}".format(root / "u250_preflight.json"))


def cmd_selftest(args):
    seen = set()
    for task_id in range(40):
        pair = task_mapping(task_id)
        if pair in seen:
            raise AssertionError("duplicate mapping: {}".format(pair))
        seen.add(pair)
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if seen != expected:
        raise AssertionError("mapping is incomplete")
    if task_mapping(0) != ("resnet18", 20260920):
        raise AssertionError("pilot mapping changed")
    if task_mapping(39) != ("squeezenet1_1", 20260924):
        raise AssertionError("last mapping changed")
    print("SELF_TEST=PASS tasks=40 unique=40")


def main():
    parser = argparse.ArgumentParser()
    # Keep map/self-test usable on older login-node Python installations.  The
    # The actual runtime is gated to canonical Python 3.6.8 before training.
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("map")
    p.add_argument("--task-id", type=int)
    p.set_defaults(func=cmd_map)
    p = sub.add_parser("prepare")
    p.add_argument("--model", choices=MODELS, required=True)
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("validate")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("verify-prepared")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_verify_prepared)
    p = sub.add_parser("train")
    p.add_argument("--root", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--seed", choices=SEEDS, type=int, required=True)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("aggregate")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_aggregate)
    p = sub.add_parser("u250-preflight")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_u250_preflight)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = parser.parse_args()
    if args.command is None:
        parser.error("a command is required")
    args.func(args)


if __name__ == "__main__":
    main()
PY
  chmod 0755 "$tmp"
  if [[ -f "$DRIVER" ]] && cmp -s "$tmp" "$DRIVER"; then
    rm -f "$tmp"
  else
    mv -f "$tmp" "$DRIVER"
  fi
}

setup_runtime() {
  init_dirs
  command -v flock >/dev/null 2>&1 || die "flock is required"
  note "Waiting for runtime lock (timeout=${LOCK_TIMEOUT}s): $LOCK_ROOT/runtime.lock"
  (
    flock -w "$LOCK_TIMEOUT" -x 9 \
      || die "timeout waiting for runtime lock: $LOCK_ROOT/runtime.lock"
    note "Runtime lock acquired"
    [[ -x "$CONDA_ROOT/bin/conda" ]] || die "Conda was not found: $CONDA_ROOT/bin/conda"
    if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
      note "Creating exact Vitis-AI 2.0 training environment: $ENV_PREFIX"
      # Do not ask the current Conda defaults channel for the historical pip
      # package.  Some modern mirrors retain Python 3.6.8 but no longer expose
      # pip=21.3.1 as a Conda artifact.  Create only the exact interpreter,
      # then pin pip/setuptools/wheel from PyPI (or the offline wheelhouse).
      conda_args=(create -y -p "$ENV_PREFIX" --no-default-packages \
        "python=$PYTHON_VERSION")
      [[ "$OFFLINE" != "1" ]] || conda_args+=(--offline)
      CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY=false \
        "$CONDA_ROOT/bin/conda" "${conda_args[@]}"
    fi
    if [[ ! -f "$ENV_PREFIX/.multimodel_runtime_complete" ]]; then
      note "Bootstrapping pip=$PIP_VERSION, setuptools=$SETUPTOOLS_VERSION, wheel=$WHEEL_VERSION"
      if ! "$ENV_PREFIX/bin/python" -m pip --version >/dev/null 2>&1; then
        "$ENV_PREFIX/bin/python" -m ensurepip --upgrade
      fi
      if [[ "$OFFLINE" == "1" ]]; then
        [[ -d "$WHEELHOUSE" ]] || die "OFFLINE=1 but wheelhouse is absent: $WHEELHOUSE"
        "$ENV_PREFIX/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
          --upgrade "pip==$PIP_VERSION" "setuptools==$SETUPTOOLS_VERSION" \
          "wheel==$WHEEL_VERSION"
      else
        "$ENV_PREFIX/bin/python" -m pip install --upgrade \
          "pip==$PIP_VERSION" "setuptools==$SETUPTOOLS_VERSION" \
          "wheel==$WHEEL_VERSION"
      fi
      if [[ "$OFFLINE" == "1" ]]; then
        "$ENV_PREFIX/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
          "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==0.8.0" \
          "numpy==$NUMPY_VERSION" "Pillow==$PILLOW_VERSION" "zstandard==$ZSTANDARD_VERSION"
      else
        "$ENV_PREFIX/bin/python" -m pip install \
          "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==0.8.0" \
          -f https://download.pytorch.org/whl/torch_stable.html
        "$ENV_PREFIX/bin/python" -m pip install \
          "numpy==$NUMPY_VERSION" "Pillow==$PILLOW_VERSION" "zstandard==$ZSTANDARD_VERSION"
      fi
      "$ENV_PREFIX/bin/python" - <<'PY'
import sys, numpy, PIL, torch, torchvision, torchaudio
import pip, setuptools, wheel
assert sys.version_info[:3] == (3, 6, 8), sys.version
assert pip.__version__ == '21.3.1', pip.__version__
assert setuptools.__version__ == '59.6.0', setuptools.__version__
assert wheel.__version__ == '0.37.1', wheel.__version__
assert torch.__version__ == '1.8.0+cu111', torch.__version__
assert torchvision.__version__ == '0.9.0+cu111', torchvision.__version__
assert torchaudio.__version__.split('+')[0] == '0.8.0', torchaudio.__version__
assert numpy.__version__ == '1.19.5', numpy.__version__
assert PIL.__version__ == '8.4.0', PIL.__version__
assert torch.version.cuda == '11.1', torch.version.cuda
PY
      "$ENV_PREFIX/bin/python" -m pip freeze >"$PROJECT_ROOT/provenance/pip_freeze.txt"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$ENV_PREFIX/.multimodel_runtime_complete"
    else
      note "Exact Conda runtime already prepared: $ENV_PREFIX"
    fi
  ) 9>"$LOCK_ROOT/runtime.lock"
}

activate_runtime() {
  # Match the cluster convention while selecting the dedicated, version-gated env.
  source "$CONDA_ROOT/bin/activate"
  conda activate "$ENV_PREFIX"
}

select_a100_gpu() {
  export REQUIRED_VRAM
  local helper="$GPU_CHECK_HELPER"
  if [[ -z "$helper" ]]; then
    for candidate in \
      /datastore/icc_fce/gpu_check.sh \
      /datastore/icc_fce/scripts/gpu_check.sh \
      "$HOME/gpu_check.sh"; do
      if [[ -f "$candidate" ]]; then helper="$candidate"; break; fi
    done
  fi
  if [[ -n "$helper" ]]; then
    [[ -f "$helper" ]] \
      || die "GPU_CHECK_HELPER does not exist on compute node: $helper"
    # This optional site helper performs its own GPU selection, so discard any
    # inherited visibility mask before sourcing it.  It is never run by submit.
    unset CUDA_VISIBLE_DEVICES
    source "$helper"
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
      || die "gpu_check.sh did not export CUDA_VISIBLE_DEVICES"
    note "Compute-node GPU helper selected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  else
    [[ -n "${SLURM_JOB_ID:-}" ]] \
      || die "no GPU helper and worker is not running inside a Slurm job"
    note "Using GPU allocation from Slurm (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-scheduler-default})"
  fi

  # The authoritative hardware check runs only here, after Slurm has placed the
  # worker on a compute node and after the exact CUDA-enabled runtime is active.
  "$ENV_PREFIX/bin/python" - <<'PY'
import os
import torch

required_mib = int(os.environ['REQUIRED_VRAM'])
assert torch.cuda.is_available(), 'CUDA is unavailable on the allocated compute node'
assert torch.cuda.device_count() >= 1, 'Slurm/helper exposed no CUDA device'
name = torch.cuda.get_device_name(0)
total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
assert 'A100' in name.upper(), 'expected an A100, got: %s' % name
assert total_mib >= required_mib, (
    'allocated GPU has %d MiB total VRAM; REQUIRED_VRAM=%d MiB'
    % (total_mib, required_mib)
)
print('COMPUTE_GPU_CHECK=PASS name=%s total_vram_mib=%d' % (name, total_mib))
PY
}

prepare_dataset() {
  init_dirs
  note "Waiting for dataset lock (timeout=${LOCK_TIMEOUT}s): $LOCK_ROOT/dataset.lock"
  (
    flock -w "$LOCK_TIMEOUT" -x 9 \
      || die "timeout waiting for dataset lock: $LOCK_ROOT/dataset.lock"
    note "Dataset lock acquired"
    if [[ ! -f "$DATA_ARCHIVE" ]]; then
      [[ "$OFFLINE" != "1" ]] || die "OFFLINE=1 and CIFAR archive is absent: $DATA_ARCHIVE"
      note "Downloading CIFAR-100"
      local_tmp="${DATA_ARCHIVE}.download.$$"
      if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 5 --retry-delay 5 -o "$local_tmp" "$CIFAR_URL"
      elif command -v wget >/dev/null 2>&1; then
        wget --tries=5 --waitretry=5 -O "$local_tmp" "$CIFAR_URL"
      else
        die "curl or wget is required"
      fi
      mv -f "$local_tmp" "$DATA_ARCHIVE"
    fi
    actual_md5="$(md5sum "$DATA_ARCHIVE" | awk '{print $1}')"
    [[ "$actual_md5" == "$CIFAR_MD5" ]] || die "CIFAR-100 MD5 mismatch: $actual_md5"
    if [[ ! -f "$DATA_ROOT/cifar-100-python/train" || ! -f "$DATA_ROOT/cifar-100-python/test" ]]; then
      tar -xzf "$DATA_ARCHIVE" -C "$DATA_ROOT"
    fi
    [[ -f "$DATA_ROOT/cifar-100-python/train" ]] || die "CIFAR train split missing after extraction"
    [[ -f "$DATA_ROOT/cifar-100-python/test" ]] || die "CIFAR test split missing after extraction"
    printf '%s  %s\n' "$CIFAR_MD5" "$(basename "$DATA_ARCHIVE")" \
      >"$PROJECT_ROOT/provenance/cifar100.md5"
    note "CIFAR-100 verification=PASS md5=$CIFAR_MD5"
  ) 9>"$LOCK_ROOT/dataset.lock"
}

copy_provenance() {
  init_dirs
  local destination="$PROJECT_ROOT/provenance/train_multimodel_cifar100_a100.sh"
  (
    flock -x 9
    if [[ "$(readlink -f "$SELF")" != "$(readlink -f "$destination" 2>/dev/null || true)" ]]; then
      local tmp_destination="${destination}.tmp.$$"
      cp -f "$SELF" "$tmp_destination"
      mv -f "$tmp_destination" "$destination"
    fi
    sha256sum "$destination" >"${destination}.sha256"
  ) 9>"$LOCK_ROOT/provenance.lock"
}

prepare_pretrained() {
  local model="$1"
  note "Waiting for pretrained lock: $model"
  (
    flock -w "$LOCK_TIMEOUT" -x 9 \
      || die "timeout waiting for pretrained lock: $model"
    if [[ "$OFFLINE" == "1" ]]; then
      export TORCH_HOME
    fi
    "$ENV_PREFIX/bin/python" "$DRIVER" prepare --model "$model" \
      >"$PROJECT_ROOT/provenance/pretrained_${model}.json"
  ) 9>"$LOCK_ROOT/pretrained_${model}.lock"
}

prepare_data_mode() {
  note "PREP_STAGE=1/5 materialize embedded training driver"
  write_driver
  note "PREP_STAGE=2/5 create or verify exact Conda runtime"
  setup_runtime
  activate_runtime
  note "PREP_STAGE=3/5 download or verify CIFAR-100"
  prepare_dataset
  copy_provenance

  if [[ -f "$PROJECT_ROOT/provenance/PREPARED.env" && \
        -f "$PROJECT_ROOT/provenance/preparation_manifest.sha256" ]] && \
     (cd "$PROJECT_ROOT" && sha256sum -c provenance/preparation_manifest.sha256 >/dev/null) && \
     "$ENV_PREFIX/bin/python" "$DRIVER" verify-prepared --root "$PROJECT_ROOT" >/dev/null; then
    note "PREPARATION=ALREADY_COMPLETE root=$PROJECT_ROOT"
    return 0
  fi

  note "PREP_STAGE=4/5 download or verify eight pretrained models"
  local model
  for model in \
    resnet18 resnet50 resnet101 mobilenet_v2 \
    densenet121 vgg16 shufflenet_v2_x1_0 squeezenet1_1; do
    note "Preparing pretrained artifact: $model"
    prepare_pretrained "$model"
  done

  {
    echo "PREPARED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "CIFAR100_MD5=$CIFAR_MD5"
    echo "PYTHON_VERSION=$PYTHON_VERSION"
    echo "TORCH_VERSION=$TORCH_VERSION"
    echo "TORCHVISION_VERSION=$TORCHVISION_VERSION"
    echo "MODEL_COUNT=8"
    echo "STATUS=PASS"
  } >"$PROJECT_ROOT/provenance/PREPARED.env.tmp"
  mv -f "$PROJECT_ROOT/provenance/PREPARED.env.tmp" \
    "$PROJECT_ROOT/provenance/PREPARED.env"

  (
    cd "$PROJECT_ROOT"
    sha256sum \
      provenance/PREPARED.env \
      provenance/cifar100.md5 \
      provenance/pip_freeze.txt \
      provenance/pretrained_*.json \
      >provenance/preparation_manifest.sha256
  )
  note "PREP_STAGE=5/5 write and verify preparation manifest"
  note "PREPARATION=PASS root=$PROJECT_ROOT"
}

verify_prepared_inputs() {
  [[ -f "$PROJECT_ROOT/provenance/PREPARED.env" ]] \
    || die "PREPARED.env is missing; run prepare-data before allocating a GPU"
  (
    cd "$PROJECT_ROOT"
    sha256sum -c provenance/preparation_manifest.sha256
  )
  "$ENV_PREFIX/bin/python" "$DRIVER" verify-prepared --root "$PROJECT_ROOT"
}

start_telemetry() {
  local destination="$1"
  echo "timestamp,name,uuid,utilization_gpu_pct,memory_used_mib,power_draw_w,temperature_gpu_c" >"$destination"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >>"$destination" 2>/dev/null || true
      sleep 10
    done
  ) &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

worker_mode() {
  local task_id="${1:-${SLURM_ARRAY_TASK_ID:-}}"
  [[ -n "$task_id" ]] || die "worker mode requires a task id"
  require_uint "task id" "$task_id"
  (( task_id >= 0 && task_id < 40 )) || die "task id must be in [0,39]"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

  write_driver
  [[ -f "$PROJECT_ROOT/provenance/PREPARED.env" ]] \
    || die "input preparation is incomplete; run prepare-data or submit from the login node"
  setup_runtime
  activate_runtime
  verify_prepared_inputs
  select_a100_gpu
  [[ -n "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" ]] \
    || die "no NVIDIA GPU is visible after gpu_check.sh"
  prepare_dataset
  copy_provenance

  local mapping model seed directory
  mapping="$("$ENV_PREFIX/bin/python" "$DRIVER" map --task-id "$task_id")"
  model="$("$ENV_PREFIX/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["model"])' <<<"$mapping")"
  seed="$("$ENV_PREFIX/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["seed"])' <<<"$mapping")"
  directory="$RESULTS_ROOT/$model/seed_$seed"
  mkdir -p "$directory"

  if "$ENV_PREFIX/bin/python" "$DRIVER" validate --run-dir "$directory" >/dev/null 2>&1; then
    rm -f "$directory/last.pt"
    note "Already complete: model=$model seed=$seed"
    return 0
  fi
  nvidia-smi -q >"$directory/nvidia_smi_q.txt"
  start_telemetry "$directory/gpu_telemetry.csv"
  trap stop_telemetry EXIT INT TERM
  note "Training model=$model seed=$seed task_id=$task_id"
  set -o pipefail
  "$ENV_PREFIX/bin/python" "$DRIVER" train --root "$PROJECT_ROOT" --model "$model" --seed "$seed" \
    2>&1 | tee -a "$directory/training.log"
  stop_telemetry
  trap - EXIT INT TERM
  "$ENV_PREFIX/bin/python" "$DRIVER" validate --run-dir "$directory"
}

aggregate_mode() {
  write_driver
  setup_runtime
  prepare_dataset
  copy_provenance
  "$ENV_PREFIX/bin/python" "$DRIVER" aggregate --root "$PROJECT_ROOT" \
    2>&1 | tee "$LOG_ROOT/aggregate_$(date -u +%Y%m%dT%H%M%SZ).log"
}

u250_preflight_mode() {
  local root="${PROJECT_ROOT}"
  if [[ "${1:-}" == "--root" ]]; then
    [[ -n "${2:-}" ]] || die "--root requires a value"
    root="$2"
    shift 2
  fi
  (( $# == 0 )) || die "unexpected u250-preflight arguments: $*"
  write_driver
  local u250_python="${U250_PYTHON:-python3}"
  command -v "$u250_python" >/dev/null 2>&1 || die "$u250_python was not found"
  "$u250_python" "$DRIVER" u250-preflight --root "$root"
}

map_mode() {
  write_driver
  local python_cmd="${ENV_PREFIX}/bin/python"
  if [[ ! -x "$python_cmd" ]]; then
    python_cmd="$(command -v python3 || true)"
  fi
  [[ -n "$python_cmd" ]] || die "python3 is required for map mode"
  "$python_cmd" "$DRIVER" map "${@}"
}

self_test_mode() {
  write_driver
  local python_cmd="${ENV_PREFIX}/bin/python"
  if [[ ! -x "$python_cmd" ]]; then
    python_cmd="$(command -v python3 || true)"
  fi
  [[ -n "$python_cmd" ]] || die "python3 is required for self-test"
  "$python_cmd" -m py_compile "$DRIVER"
  "$python_cmd" "$DRIVER" selftest
  echo "SCRIPT_VERSION=$SCRIPT_VERSION"
}

stack_check_mode() {
  write_driver
  setup_runtime
  activate_runtime
  "$ENV_PREFIX/bin/python" - <<'PY'
import sys, numpy, PIL, torch, torchvision, torchaudio
expected = {
    "python": (3, 6, 8),
    "torch": "1.8.0+cu111",
    "torchvision": "0.9.0+cu111",
    "torchaudio": "0.8.0",
    "numpy": "1.19.5",
    "pillow": "8.4.0",
    "cuda": "11.1",
}
actual = {
    "python": sys.version_info[:3],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__.split("+")[0],
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "cuda": torch.version.cuda,
}
assert actual == expected, (actual, expected)
print("STACK_GATE=PASS")
for key in sorted(actual):
    print("{}={}".format(key.upper(), actual[key]))
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [prepare-data|worker [TASK_ID]|aggregate|u250-preflight [--root DIR]|map [--task-id N]|self-test|stack-check]

  prepare-data Prepare and verify Conda stack, CIFAR-100, and all pretrained weights.
  worker      Run one mapped model/seed task (normally called by Slurm).
  aggregate   Require all 40 valid runs, summarize, hash, and create tar.zst.
  u250-preflight  On torchvision 0.9+cpu, strict-load all 40 exported weights.
  map         Print the complete task mapping or one task as JSON.
  self-test   Check embedded Python syntax and the 40-task mapping.
  stack-check Create/verify the exact A100 PyTorch/Vitis-AI training contract.
EOF
}

mode="${1:-self-test}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  prepare-data) prepare_data_mode "$@" ;;
  worker) worker_mode "$@" ;;
  aggregate) aggregate_mode "$@" ;;
  u250-preflight) u250_preflight_mode "$@" ;;
  map) map_mode "$@" ;;
  self-test) self_test_mode "$@" ;;
  stack-check) stack_check_mode "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; die "unknown mode: $mode" ;;
esac
