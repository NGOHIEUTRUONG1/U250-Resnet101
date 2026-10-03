#!/bin/bash
set -Eeuo pipefail
IFS=$'\n\t'

# Fail-closed ResNet101 CIFAR-100 -> INT8 XModel -> DPUCADF8H/U250
# V5: tuned for large CPU nodes; controlled main-thread parallelism + single-thread DataLoader workers.
# Vitis/Vivado 2021.2, Vitis-AI 2.0, exact local source commit.
# Any version/SHA/target mismatch stops immediately.
#
# Optional:
#   CALIB_SAMPLES=200
#   TEST_SAMPLES=1000       # set 10000 for full CIFAR-100 test
#   CALIB_BATCH=32
#   TEST_BATCH=32
#   MAX_TOP1_DROP=0.05

ROOT="${ROOT:?Set ROOT to the U250 study workspace}"
TRAIN="$ROOT/train"
VA="$ROOT/source/Vitis-AI"
VITIS_SETTINGS="${VITIS_SETTINGS:?Set VITIS_SETTINGS to Vitis 2021.2 settings64.sh}"

EXPECTED_VA_COMMIT=d02dcb6041663dbc7ecbc0c6af9fafa087a789de
EXPECTED_CKPT_SHA=331e2ea6720abf004154b1e58a74b675af2a20f306746ccfda91528533e3ddb3
EXPECTED_DATASET_MD5=eb9058c3a382ffc7106e4002c42a8d85

IMAGE=xilinx/vitis-ai-cpu:2.0.0.1103
EXPECTED_IMAGE_DIGEST=sha256:7a87fa94fd65710d3eab2713d4fb997fd7828ed627bedc71c8c0a1cea9eb4098
PINNED_IMAGE="xilinx/vitis-ai-cpu@${EXPECTED_IMAGE_DIGEST}"
PINNED_DOCKER_URI="docker://xilinx/vitis-ai-cpu@${EXPECTED_IMAGE_DIGEST}"

UDOCKER_VERSION=1.3.17
UDOCKER_TARBALL_SHA256=f7f4d0cbd8b4efc93584629435808b95fc46c3c536839e3b773873d23a9f2f59
UDOCKER_URL=https://files.pythonhosted.org/packages/source/u/udocker/udocker-1.3.17.tar.gz
UDOCKER_BASE="$ROOT/toolchain/udocker-$UDOCKER_VERSION"
UDOCKER_STORE="$ROOT/toolchain/udocker_store"
UDOCKER_IMAGE="xilinx/vitis-ai-cpu:2.0.0.1103"
UDOCKER_CONTAINER="vai20_cpu_7a87fa94_p1fast"

CKPT="$TRAIN/resnet101_cifar100_best.pt"
CKPT_SHA_FILE="$TRAIN/resnet101_cifar100_best.pt.sha256"
CONFIG="$TRAIN/train_config.json"
CONTRACT="$TRAIN/STACK_CONTRACT.env"
DATASET_TGZ="$TRAIN/cifar-100-python.tar.gz"

CALIB_SAMPLES="${CALIB_SAMPLES:-200}"
TEST_SAMPLES="${TEST_SAMPLES:-1000}"
CALIB_BATCH="${CALIB_BATCH:-32}"
TEST_BATCH="${TEST_BATCH:-32}"
MAX_TOP1_DROP="${MAX_TOP1_DROP:-0.05}"

# CPU acceleration defaults. `nproc` respects the process CPU affinity on Linux.
HOST_CPUS="$(nproc 2>/dev/null || echo 1)"

# Keep a small CPU reserve for PRoot/kernel/I/O. On the user's 70-CPU node
# this selects 56 main compute threads by default.
if [ "$HOST_CPUS" -ge 64 ]; then
  DEFAULT_CPU_THREADS=56
elif [ "$HOST_CPUS" -ge 48 ]; then
  DEFAULT_CPU_THREADS=40
elif [ "$HOST_CPUS" -ge 32 ]; then
  DEFAULT_CPU_THREADS=28
elif [ "$HOST_CPUS" -ge 16 ]; then
  DEFAULT_CPU_THREADS=14
else
  DEFAULT_CPU_THREADS="$HOST_CPUS"
fi
CPU_THREADS="${CPU_THREADS:-$DEFAULT_CPU_THREADS}"
INTEROP_THREADS="${INTEROP_THREADS:-1}"

if [ "$HOST_CPUS" -ge 64 ]; then
  DEFAULT_NUM_WORKERS=8
elif [ "$HOST_CPUS" -ge 32 ]; then
  DEFAULT_NUM_WORKERS=6
elif [ "$HOST_CPUS" -ge 16 ]; then
  DEFAULT_NUM_WORKERS=4
elif [ "$HOST_CPUS" -ge 8 ]; then
  DEFAULT_NUM_WORKERS=2
else
  DEFAULT_NUM_WORKERS=0
fi
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"

case "$CPU_THREADS" in ''|*[!0-9]*) echo "invalid CPU_THREADS=$CPU_THREADS"; exit 2;; esac
case "$INTEROP_THREADS" in ''|*[!0-9]*) echo "invalid INTEROP_THREADS=$INTEROP_THREADS"; exit 2;; esac
case "$NUM_WORKERS" in ''|*[!0-9]*) echo "invalid NUM_WORKERS=$NUM_WORKERS"; exit 2;; esac
[ "$CPU_THREADS" -ge 1 ] || { echo "CPU_THREADS must be >=1"; exit 2; }
[ "$INTEROP_THREADS" -ge 1 ] || { echo "INTEROP_THREADS must be >=1"; exit 2; }

if [ $((CPU_THREADS + NUM_WORKERS)) -gt "$HOST_CPUS" ]; then
  echo "ERROR: CPU_THREADS + NUM_WORKERS exceeds visible CPUs: $CPU_THREADS + $NUM_WORKERS > $HOST_CPUS"
  exit 2
fi

echo "HOST_CPUS=$HOST_CPUS"
echo "CPU_THREADS=$CPU_THREADS"
echo "INTEROP_THREADS=$INTEROP_THREADS"
echo "NUM_WORKERS=$NUM_WORKERS"

TS="$(date +%Y%m%d_%H%M%S)"
WORK="$ROOT/xmodel_build/resnet101_cifar100_vai20_${TS}"
LOG="$WORK/MASTER.log"
HOST_REPORT="$WORK/HOST_VERIFY.txt"
CONTAINER_SCRIPT="$WORK/container_build.sh"
PY_DRIVER="$WORK/quantize_resnet101.py"

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

die() {
  echo
  echo "================================================================"
  echo "RESULT=STOP"
  echo "REASON=$*"
  echo "WORK=$WORK"
  echo "================================================================"
  exit 1
}
stage() {
  echo
  echo "================================================================"
  echo "STAGE: $*"
  echo "================================================================"
}

trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "FAILED_RC=$rc"; echo "WORK=$WORK"; fi' EXIT

stage "0 - PARAMETERS"
cat <<EOF
ROOT=$ROOT
VA=$VA
VITIS_SETTINGS=$VITIS_SETTINGS
IMAGE=$IMAGE
CALIB_SAMPLES=$CALIB_SAMPLES
TEST_SAMPLES=$TEST_SAMPLES
MAX_TOP1_DROP=$MAX_TOP1_DROP
WORK=$WORK
EOF

stage "1 - HOST FILES / CONTRACT"
for f in "$VITIS_SETTINGS" "$CONTRACT" "$CONFIG" "$CKPT" "$CKPT_SHA_FILE" "$DATASET_TGZ"; do
  [ -f "$f" ] || die "required file missing: $f"
done

# These STACK_CONTRACT checks describe the TRAINING provenance of the checkpoint.
# They intentionally remain independent from the pinned Vitis-AI 2.0 quantization
# container versions checked later inside the container.
grep -qx 'PYTHON_VERSION=3.6.8' "$CONTRACT" || die "contract Python mismatch"
grep -qx 'TORCH_BASE_VERSION=1.8.0' "$CONTRACT" || die "contract torch mismatch"
grep -qx 'TORCHVISION_BASE_VERSION=0.9.0' "$CONTRACT" || die "contract torchvision mismatch"
grep -qx 'NUMPY_VERSION=1.19.5' "$CONTRACT" || die "contract numpy mismatch"
grep -qx 'PILLOW_VERSION=8.4.0' "$CONTRACT" || die "contract Pillow mismatch"
grep -qx 'U250_TORCH_VERSION=1.8.0+cpu' "$CONTRACT" || die "contract U250 torch mismatch"
grep -qx 'U250_TORCHVISION_VERSION=0.9.0+cpu' "$CONTRACT" || die "contract U250 torchvision mismatch"
grep -qx 'VITIS_AI_RELEASE=2.0' "$CONTRACT" || die "contract Vitis-AI mismatch"
grep -qx 'MODEL_IMPL=torchvision.models.resnet101(pretrained=False)' "$CONTRACT" || die "model contract mismatch"
grep -qx 'NUM_CLASSES=100' "$CONTRACT" || die "num_classes mismatch"
grep -qx 'INPUT_SHAPE=1,3,224,224' "$CONTRACT" || die "input shape mismatch"
grep -qx 'CHECKPOINT_FORMAT=CPU_STATE_DICT' "$CONTRACT" || die "checkpoint format mismatch"
grep -qx 'CHECKPOINT_LOAD_POLICY=STRICT_TRUE' "$CONTRACT" || die "checkpoint load policy mismatch"

ACTUAL_CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
SAVED_CKPT_SHA="$(awk 'NR==1 {print $1}' "$CKPT_SHA_FILE")"
echo "ACTUAL_CKPT_SHA=$ACTUAL_CKPT_SHA"
echo "SAVED_CKPT_SHA=$SAVED_CKPT_SHA"
[ "$ACTUAL_CKPT_SHA" = "$EXPECTED_CKPT_SHA" ] || die "checkpoint SHA mismatch"
[ "$SAVED_CKPT_SHA" = "$EXPECTED_CKPT_SHA" ] || die "checkpoint SHA file mismatch"

ACTUAL_DATASET_MD5="$(md5sum "$DATASET_TGZ" | awk '{print $1}')"
echo "DATASET_MD5=$ACTUAL_DATASET_MD5"
[ "$ACTUAL_DATASET_MD5" = "$EXPECTED_DATASET_MD5" ] || die "CIFAR-100 MD5 mismatch"

stage "2 - EXACT VITIS-AI SOURCE"
[ -d "$VA/.git" ] || die "Vitis-AI git checkout missing"
ACTUAL_VA_COMMIT="$(cd "$VA" && git rev-parse HEAD)"
echo "ACTUAL_VA_COMMIT=$ACTUAL_VA_COMMIT"
[ "$ACTUAL_VA_COMMIT" = "$EXPECTED_VA_COMMIT" ] || die "Vitis-AI commit mismatch"
Q_SRC="$VA/tools/Vitis-AI-Quantizer/vai_q_pytorch"
[ -d "$Q_SRC/pytorch_binding/pytorch_nndct" ] || die "vai_q_pytorch source missing"

stage "3 - ISOLATED VITIS/VIVADO 2021.2"
env -i HOME="$HOME" USER="${USER:-benchmark}" SHELL=/bin/bash LANG=C PATH=/usr/bin:/bin \
  VITIS_SETTINGS="$VITIS_SETTINGS" \
  /bin/bash --noprofile --norc -c '
    set -e
    source "$VITIS_SETTINGS"
    echo "VPP=$(command -v v++)"
    echo "VIVADO=$(command -v vivado)"
    v++ --version
    vivado -version
  ' > "$HOST_REPORT" 2>&1 || die "isolated Vitis/Vivado test failed"
cat "$HOST_REPORT"
grep -q 'v++ v2021\.2' "$HOST_REPORT" || die "v++ is not 2021.2"
grep -q 'SW Build 3363252' "$HOST_REPORT" || die "v++ build mismatch"
grep -q 'Vivado v2021\.2' "$HOST_REPORT" || die "Vivado is not 2021.2"
grep -q 'SW Build 3367213' "$HOST_REPORT" || die "Vivado build mismatch"
grep -q 'IP Build 3369179' "$HOST_REPORT" || die "Vivado IP build mismatch"

stage "4 - CONFIG JSON"
if ! env -i HOME="$HOME" LANG=C PATH=/usr/bin:/bin /usr/bin/python3 - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    c=json.load(f)
assert c["dataset"]["name"]=="CIFAR-100"
assert c["dataset"]["official_md5"]=="eb9058c3a382ffc7106e4002c42a8d85"
assert c["model"]["architecture"]=="torchvision.models.resnet101"
assert c["model"]["pretrained"] is False
assert c["model"]["num_classes"]==100
assert c["model"]["input_size"]==[3,224,224]
assert c["model"]["normalization_mean"]==[0.5071,0.4867,0.4408]
assert c["model"]["normalization_std"]==[0.2675,0.2565,0.2761]
print("CONFIG_JSON_CHECK=PASS")
PY
then
  die "train_config.json mismatch"
fi

stage "5 - CONTAINER RUNTIME / PINNED VITIS-AI 2.0 IMAGE"

CONTAINER_RUNTIME=""
SIF=""

# Prefer a daemon/container engine when the current user can actually use it.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  CONTAINER_RUNTIME="docker"
  ENGINE=(docker)
elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
  CONTAINER_RUNTIME="podman"
  ENGINE=(podman)
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
  ENGINE=(apptainer)
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
  ENGINE=(singularity)
else
  echo "RUNTIME_CHECK:"
  echo "  docker=$(command -v docker 2>/dev/null || true)"
  echo "  podman=$(command -v podman 2>/dev/null || true)"
  echo "  apptainer=$(command -v apptainer 2>/dev/null || true)"
  echo "  singularity=$(command -v singularity 2>/dev/null || true)"
  echo
  echo "No system container runtime found."
  echo "Trying user-space udocker + PRoot P2 (no root, no user namespace required)."

  for c in curl tar sha256sum awk grep sed find chmod; do
    command -v "$c" >/dev/null 2>&1 || die "required host command missing for udocker bootstrap: $c"
  done

  mkdir -p "$ROOT/toolchain" "$UDOCKER_STORE"

  UDOCKER_TGZ="$ROOT/toolchain/udocker-${UDOCKER_VERSION}.tar.gz"

  if [ ! -s "$UDOCKER_TGZ" ]; then
    echo "Downloading udocker $UDOCKER_VERSION from PyPI..."
    curl -fL --retry 4 --retry-delay 3 \
      "$UDOCKER_URL" \
      -o "$UDOCKER_TGZ" ||
      die "failed to download udocker"
  fi

  ACTUAL_UDOCKER_SHA="$(sha256sum "$UDOCKER_TGZ" | awk '{print $1}')"
  echo "UDOCKER_TARBALL_SHA256=$ACTUAL_UDOCKER_SHA"
  [ "$ACTUAL_UDOCKER_SHA" = "$UDOCKER_TARBALL_SHA256" ] ||
    die "udocker tarball SHA256 mismatch"

  rm -rf "$UDOCKER_BASE"
  tar -xzf "$UDOCKER_TGZ" -C "$ROOT/toolchain" ||
    die "failed to extract udocker"

  # PyPI sdist and GitHub release tarballs both normally extract as udocker-1.3.17.
  [ -d "$UDOCKER_BASE" ] || die "expected udocker directory missing after extract"

  UDOCKER_BIN=""
  for cand in \
    "$UDOCKER_BASE/udocker/udocker" \
    "$UDOCKER_BASE/udocker" \
    "$UDOCKER_BASE/udocker/maincmd.py"
  do
    if [ -f "$cand" ] && [ -x "$cand" ]; then
      UDOCKER_BIN="$cand"
      break
    elif [ -f "$cand" ]; then
      chmod +x "$cand" || true
      if [ -x "$cand" ]; then
        UDOCKER_BIN="$cand"
        break
      fi
    fi
  done

  [ -n "$UDOCKER_BIN" ] || die "udocker executable not found in extracted release"

  export UDOCKER_DIR="$UDOCKER_STORE"
  export UDOCKER_DEFAULT_EXECUTION_MODE=P2
  export PROOT_TMP_DIR="$ROOT/toolchain/proot_tmp"
  mkdir -p "$PROOT_TMP_DIR"

  # Keep the polluted login shell away from the helper Python as much as possible.
  export PYTHONNOUSERSITE=1

  echo "UDOCKER_BIN=$UDOCKER_BIN"
  echo "UDOCKER_DIR=$UDOCKER_DIR"

  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" version 2>&1 | head -n 20 ||
    die "udocker itself does not run"

  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" install ||
    die "udocker install failed"

  # Verify the Docker Hub tag before udocker pulls it.  We intentionally
  # compare the registry's immutable manifest digest with the known VAI 2.0
  # digest.  If the tag ever changes, STOP.
  echo
  echo "Verifying Docker Hub manifest digest before pull..."

  TOKEN_JSON="$(
    curl -fsSL --retry 4 \
      'https://auth.docker.io/token?service=registry.docker.io&scope=repository:xilinx/vitis-ai-cpu:pull'
  )" || die "Docker Hub token request failed"

  TOKEN="$(
    printf '%s' "$TOKEN_JSON" |
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
      /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
  )" || die "cannot parse Docker Hub token"

  HDR="$WORK/docker_manifest_headers.txt"
  curl -fsSI --retry 4 \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://registry-1.docker.io/v2/xilinx/vitis-ai-cpu/manifests/2.0.0.1103" \
    > "$HDR" ||
    die "cannot query Vitis-AI image manifest"

  REGISTRY_DIGEST="$(
    tr -d '\r' < "$HDR" |
    awk 'BEGIN{IGNORECASE=1} /^docker-content-digest:/ {print $2; exit}'
  )"

  echo "REGISTRY_MANIFEST_DIGEST=$REGISTRY_DIGEST"
  echo "EXPECTED_IMAGE_DIGEST=$EXPECTED_IMAGE_DIGEST"

  [ "$REGISTRY_DIGEST" = "$EXPECTED_IMAGE_DIGEST" ] ||
    die "Docker Hub tag does not resolve to expected Vitis-AI 2.0 digest"

  # Avoid pulling a 6+ GB image without reasonable free space.
  AVAIL_KB="$(df -Pk "$ROOT" | awk 'NR==2{print $4}')"
  echo "AVAILABLE_KB=$AVAIL_KB"
  [ "${AVAIL_KB:-0}" -ge 26214400 ] ||
    die "less than 25 GiB free under $ROOT; Vitis-AI image is too large"

  echo
  echo "Pulling exact verified Vitis-AI tag with udocker..."
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" pull "$UDOCKER_IMAGE" ||
    die "udocker pull failed"

  # Re-create only our named container; do not touch unrelated udocker data.
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" rm "$UDOCKER_CONTAINER" >/dev/null 2>&1 || true

  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" create --name="$UDOCKER_CONTAINER" "$UDOCKER_IMAGE" ||
    die "udocker create failed"

  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" setup --execmode=P1 "$UDOCKER_CONTAINER" ||
    die "udocker P1 setup failed"

  echo
  echo "Testing accelerated PRoot P1 execution..."
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" run \
      --env="HOME=/tmp" \
      "$UDOCKER_CONTAINER" \
      /bin/bash -lc 'echo UDOCKER_PROOT_P1=PASS; uname -m; cat /etc/os-release | head -n 3' ||
    die "udocker/PRoot P1 cannot execute on this cluster; use previous P2 script if needed"

  CONTAINER_RUNTIME="udocker"
  ENGINE=("$UDOCKER_BIN")
fi

echo "CONTAINER_RUNTIME=$CONTAINER_RUNTIME"
if [ "$CONTAINER_RUNTIME" = "udocker" ]; then
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    "$UDOCKER_BIN" version 2>&1 | head -n 10 || true
else
  "${ENGINE[@]}" --version 2>&1 | head -n 5 || true
fi

case "$CONTAINER_RUNTIME" in
  udocker)
    echo "UDOCKER_IMAGE=$UDOCKER_IMAGE"
    echo "UDOCKER_CONTAINER=$UDOCKER_CONTAINER"
    echo "IMAGE_DIGEST_VERIFIED_VIA_REGISTRY=$EXPECTED_IMAGE_DIGEST"
    ;;

  docker|podman)
    # Pull by immutable digest, NOT by tag. This makes a tag change irrelevant.
    if ! "${ENGINE[@]}" image inspect "$PINNED_IMAGE" >/dev/null 2>&1; then
      echo "Pulling immutable image:"
      echo "  $PINNED_IMAGE"
      "${ENGINE[@]}" pull "$PINNED_IMAGE" ||
        die "$CONTAINER_RUNTIME pull failed"
    fi

    REPO_DIGESTS="$("${ENGINE[@]}" image inspect "$PINNED_IMAGE" \
      --format '{{json .RepoDigests}}' 2>/dev/null || true)"
    echo "IMAGE_REPO_DIGESTS=$REPO_DIGESTS"

    printf '%s\n' "$REPO_DIGESTS" | grep -q "$EXPECTED_IMAGE_DIGEST" ||
      die "$CONTAINER_RUNTIME image digest verification failed"
    ;;

  apptainer|singularity)
    # HPC-safe path: no daemon and no root permission required.
    # The source URI itself is pinned to the exact OCI digest.
    SIF_DIR="$ROOT/toolchain/containers"
    mkdir -p "$SIF_DIR"
    SIF="$SIF_DIR/vitis-ai-cpu_2.0.0.1103_${EXPECTED_IMAGE_DIGEST#sha256:}.sif"

    if [ ! -s "$SIF" ]; then
      echo "Pulling OCI digest through $CONTAINER_RUNTIME:"
      echo "  $PINNED_DOCKER_URI"
      echo "  -> $SIF"
      "${ENGINE[@]}" pull "$SIF" "$PINNED_DOCKER_URI" ||
        die "$CONTAINER_RUNTIME pull failed"
    fi

    [ -s "$SIF" ] || die "SIF image missing after pull"

    # SIF is a deterministic local artifact for this run; record and later
    # re-check its SHA256. OCI source identity is protected by the digest URI.
    SIF_SHA_BEFORE="$(sha256sum "$SIF" | awk '{print $1}')"
    echo "SIF=$SIF"
    echo "SIF_SHA256_BEFORE=$SIF_SHA_BEFORE"

    "${ENGINE[@]}" inspect "$SIF" > "$WORK/${CONTAINER_RUNTIME}_inspect.txt" 2>&1 ||
      die "$CONTAINER_RUNTIME cannot inspect pulled SIF"
    ;;
esac

stage "6 - GENERATE DRIVER"
cat > "$PY_DRIVER" <<'PY'
from __future__ import print_function
import pytorch_nndct
import argparse, json, os, pickle, sys, tarfile, hashlib
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pytorch_nndct.apis import torch_quantizer

EXPECTED_SHA="331e2ea6720abf004154b1e58a74b675af2a20f306746ccfda91528533e3ddb3"
MEAN=[0.5071,0.4867,0.4408]
STD=[0.2675,0.2565,0.2761]

def die(x): raise RuntimeError(x)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

class CIFAR100Pickle(Dataset):
    def __init__(self,p,transform=None):
        with open(p,"rb") as f: d=pickle.load(f,encoding="bytes")
        if b"data" not in d or b"fine_labels" not in d: die("bad CIFAR-100 pickle")
        self.data=d[b"data"]; self.labels=d[b"fine_labels"]; self.transform=transform
        if self.data.shape[1] != 3072: die("unexpected CIFAR shape")
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        x=self.data[i].reshape(3,32,32).transpose(1,2,0)
        x=Image.fromarray(x)
        if self.transform: x=self.transform(x)
        return x,int(self.labels[i])

def make_model():
    m=models.resnet101(pretrained=False)
    m.fc=nn.Linear(m.fc.in_features,100)
    return m

def load_checkpoint(model,p):
    if sha256_file(p)!=EXPECTED_SHA: die("checkpoint SHA changed")
    ckpt=torch.load(p,map_location="cpu")
    if not isinstance(ckpt,dict) or "state_dict" not in ckpt: die("bad checkpoint format")
    for k,v in {
      "architecture":"torchvision.models.resnet101(pretrained=False)",
      "num_classes":100,
      "input_shape":[1,3,224,224],
    }.items():
        if k in ckpt and ckpt[k]!=v: die("checkpoint metadata mismatch "+k)
    r=model.load_state_dict(ckpt["state_dict"],strict=True)
    if getattr(r,"missing_keys",[]): die("missing state_dict keys")
    if getattr(r,"unexpected_keys",[]): die("unexpected state_dict keys")
    if tuple(model.fc.weight.shape)!=(100,2048): die("fc shape mismatch")
    return ckpt

def extract_dataset(archive,out):
    root=os.path.join(out,"cifar-100-python")
    tr=os.path.join(root,"train"); te=os.path.join(root,"test")
    if os.path.isfile(tr) and os.path.isfile(te): return tr,te
    if not os.path.isdir(out): os.makedirs(out)
    with tarfile.open(archive,"r:gz") as tf:
        base=os.path.abspath(out)
        for m in tf.getmembers():
            d=os.path.abspath(os.path.join(out,m.name))
            if not (d==base or d.startswith(base+os.sep)): die("unsafe tar path")
        tf.extractall(out)
    if not os.path.isfile(tr) or not os.path.isfile(te): die("dataset extraction failed")
    return tr,te

def spread(total,n):
    if n<=0 or n>=total: return list(range(total))
    return [int((i*total)//n) for i in range(n)]

def _loader_worker_init(_worker_id):
    # DataLoader workers must stay single-threaded. Otherwise every worker
    # inherits the main OMP/MKL/PyTorch thread count and causes massive
    # oversubscription on large CPU nodes.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def loaders(a):
    tr,te=extract_dataset(a.dataset,a.dataset_dir)
    tf=transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(MEAN,STD),
    ])
    trd=CIFAR100Pickle(tr,tf); ted=CIFAR100Pickle(te,tf)
    if len(trd)!=50000 or len(ted)!=10000: die("dataset size mismatch")
    cl=DataLoader(Subset(trd,spread(len(trd),a.calib_samples)),
                  batch_size=a.calib_batch,shuffle=False,num_workers=a.num_workers,
                  worker_init_fn=_loader_worker_init,
                  persistent_workers=(a.num_workers>0))
    tl=DataLoader(Subset(ted,spread(len(ted),a.test_samples)),
                  batch_size=a.test_batch,shuffle=False,num_workers=a.num_workers,
                  worker_init_fn=_loader_worker_init,
                  persistent_workers=(a.num_workers>0))
    el=DataLoader(Subset(ted,[0]),batch_size=1,shuffle=False,num_workers=0)
    return cl,tl,el

def evaluate(model,loader):
    model.eval(); total=top1=top5=0; loss=0.0
    ce=nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x,y in loader:
            o=model(x)
            if o.ndim!=2 or o.shape[1]!=100: die("output shape mismatch")
            loss += float(ce(o,y).item()); total += int(y.numel())
            top1 += int(o.argmax(1).eq(y).sum().item())
            _,p5=o.topk(5,1,True,True)
            top5 += int(p5.eq(y.view(-1,1)).any(1).sum().item())
    if total==0: die("empty loader")
    return {"samples":total,"loss":loss/total,"top1":top1/float(total),"top5":top5/float(total)}

def version_gate():
    import numpy as np, PIL, xir
    py="%d.%d.%d"%sys.version_info[:3]
    v={"python":py,"torch":torch.__version__,"torchvision":torchvision.__version__,
       "numpy":np.__version__,"pillow":PIL.__version__,
       "pytorch_nndct_file":pytorch_nndct.__file__,"xir_file":xir.__file__}
    if py!="3.6.15": die("Python mismatch: "+py)
    if torch.__version__!="1.7.1": die("torch mismatch: "+torch.__version__)
    if torchvision.__version__!="0.8.2": die("torchvision mismatch: "+torchvision.__version__)
    if np.__version__!="1.17.5": die("numpy mismatch: "+np.__version__)
    if PIL.__version__!="8.0.0": die("Pillow mismatch: "+PIL.__version__)
    prefix=os.path.realpath(os.environ.get("CONDA_PREFIX",""))
    nndct_path=os.path.realpath(pytorch_nndct.__file__)
    if not prefix or not nndct_path.startswith(prefix+os.sep):
        die("pytorch_nndct not loaded from active pinned vitis-ai-pytorch env: "+nndct_path)
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--action",choices=["calib","test","export"],required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--dataset-dir",required=True)
    ap.add_argument("--quant-dir",required=True)
    ap.add_argument("--metrics-dir",required=True)
    ap.add_argument("--calib-samples",type=int,required=True)
    ap.add_argument("--test-samples",type=int,required=True)
    ap.add_argument("--calib-batch",type=int,required=True)
    ap.add_argument("--test-batch",type=int,required=True)
    ap.add_argument("--max-top1-drop",type=float,required=True)
    ap.add_argument("--cpu-threads",type=int,required=True)
    ap.add_argument("--interop-threads",type=int,required=True)
    ap.add_argument("--num-workers",type=int,required=True)
    a=ap.parse_args()

    if a.cpu_threads < 1: die("cpu_threads must be >=1")
    if a.interop_threads < 1: die("interop_threads must be >=1")
    if a.num_workers < 0: die("num_workers must be >=0")

    torch.set_num_threads(a.cpu_threads)
    try:
        torch.set_num_interop_threads(a.interop_threads)
    except RuntimeError as e:
        print("INTEROP_THREADS_WARNING="+str(e))

    print("TORCH_NUM_THREADS=%d"%torch.get_num_threads())
    try:
        print("TORCH_NUM_INTEROP_THREADS=%d"%torch.get_num_interop_threads())
    except Exception:
        pass
    print("DATALOADER_NUM_WORKERS=%d"%a.num_workers)

    versions=version_gate()
    for d in [a.quant_dir,a.metrics_dir]:
        if not os.path.isdir(d): os.makedirs(d)
    cl,tl,el=loaders(a)
    model=make_model().cpu().eval()
    load_checkpoint(model,a.checkpoint)

    z=torch.randn(1,3,224,224)
    traced=torch.jit.trace(model,z,strict=True)
    if tuple(traced(z).shape)!=(1,100): die("JIT trace output mismatch")
    print("STRICT_LOAD=PASS")
    print("JIT_TRACE_CHECK=PASS")

    if a.action=="calib":
        dummy=torch.randn(a.calib_batch,3,224,224)
        q=torch_quantizer("calib",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
        qm=q.quant_model; qm.eval()
        cm=evaluate(qm,cl)
        q.export_quant_config()
        with open(os.path.join(a.metrics_dir,"calib_metrics.json"),"w") as f:
            json.dump({"versions":versions,"calibration":cm},f,indent=2,sort_keys=True)
        print("CALIBRATION=PASS")
        return

    if a.action=="test":
        fm=evaluate(model,tl)
        dummy=torch.randn(a.test_batch,3,224,224)
        q=torch_quantizer("test",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
        qm=q.quant_model; qm.eval()
        im=evaluate(qm,tl)
        drop=fm["top1"]-im["top1"]
        out={"versions":versions,"float":fm,"int8":im,
             "top1_absolute_drop":drop,"max_allowed_top1_drop":a.max_top1_drop}
        with open(os.path.join(a.metrics_dir,"quant_test_metrics.json"),"w") as f:
            json.dump(out,f,indent=2,sort_keys=True)
        print("FLOAT_METRICS="+json.dumps(fm,sort_keys=True))
        print("INT8_METRICS="+json.dumps(im,sort_keys=True))
        print("TOP1_ABSOLUTE_DROP=%.8f"%drop)
        if drop>a.max_top1_drop: die("INT8 top1 drop exceeds limit")
        print("QUANT_ACCURACY_GATE=PASS")
        return

    dummy=torch.randn(1,3,224,224)
    q=torch_quantizer("test",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
    qm=q.quant_model; qm.eval()
    with torch.no_grad():
        for x,_ in el:
            if tuple(x.shape)!=(1,3,224,224): die("export is not batch 1")
            if tuple(qm(x).shape)!=(1,100): die("export forward shape mismatch")
            break
    q.export_xmodel(a.quant_dir,False)
    print("EXPORT_XMODEL_CALL=PASS")

if __name__=="__main__":
    main()
PY
chmod 0644 "$PY_DRIVER"

cat > "$CONTAINER_SCRIPT" <<'INNER'
#!/bin/bash
set -Eeuo pipefail
IFS=$'\n\t'

ROOT="${VAI_PROJECT_ROOT:-/workspace/u250_clean_2021_2}"
TRAIN="$ROOT/train"
VA="$ROOT/source/Vitis-AI"
WORK="$1"
CALIB_SAMPLES="$2"
TEST_SAMPLES="$3"
CALIB_BATCH="$4"
TEST_BATCH="$5"
MAX_TOP1_DROP="$6"
EXPECTED_VA_COMMIT="$7"
CPU_THREADS="$8"
INTEROP_THREADS="$9"
NUM_WORKERS="${10}"

Q_SRC="$VA/tools/Vitis-AI-Quantizer/vai_q_pytorch"
QDIR="$WORK/quantize_result"
MDIR="$WORK/metrics"
CDIR="$WORK/compiled"
DDIR="$WORK/dataset"
LOGDIR="$WORK/logs"
mkdir -p "$QDIR" "$MDIR" "$CDIR" "$DDIR" "$LOGDIR"

die(){ echo "CONTAINER_RESULT=STOP"; echo "CONTAINER_REASON=$*"; exit 1; }
stage(){ echo; echo "---- CONTAINER_STAGE: $* ----"; }

stage "A - VITIS-AI PYTORCH ENV"
[ -f /opt/vitis_ai/conda/etc/profile.d/conda.sh ] || die "conda activation script missing"

# Vitis-AI 2.0 conda activation scripts (notably libblas_mkl_activate.sh)
# read variables such as MKL_INTERFACE_LAYER before defining them.
# With `set -u` this aborts activation with "unbound variable".
# Disable nounset ONLY while sourcing/activating Conda, then restore it.
set +u
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda env list | tee "$LOGDIR/conda_envs.txt"
conda env list | grep -q 'vitis-ai-pytorch' || {
  set -u
  die "vitis-ai-pytorch env missing"
}
conda activate vitis-ai-pytorch || {
  rc=$?
  set -u
  die "conda activate vitis-ai-pytorch failed rc=$rc"
}
set -u

echo "CONDA_ACTIVATE_NOUNSET_GUARD=PASS" | tee "$LOGDIR/conda_activate_guard.txt"
echo "CONDA_PREFIX=${CONDA_PREFIX:-UNSET}" | tee -a "$LOGDIR/conda_activate_guard.txt"
command -v python | tee -a "$LOGDIR/conda_activate_guard.txt"
command -v vai_c_xir || die "vai_c_xir missing"

# CPU threading contract. These affect CPU math kernels without changing
# model/checkpoint/quantizer content.
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

{
  echo "VISIBLE_CPUS=$(nproc 2>/dev/null || echo UNKNOWN)"
  echo "AFFINITY=$(taskset -pc $$ 2>/dev/null || true)"
  echo "CPU_THREADS=$CPU_THREADS"
  echo "INTEROP_THREADS=$INTEROP_THREADS"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
} | tee "$LOGDIR/cpu_threading_contract.txt"

stage "B - STRICT BASE VERSION GATE"
python - <<'PY' | tee "$LOGDIR/base_versions.txt"
from __future__ import print_function
import sys, torch, torchvision, numpy, PIL, xir
py="%d.%d.%d"%sys.version_info[:3]
print("PYTHON="+py)
print("TORCH="+torch.__version__)
print("TORCHVISION="+torchvision.__version__)
print("NUMPY="+numpy.__version__)
print("PILLOW="+PIL.__version__)
print("XIR="+str(getattr(xir,"__file__","N/A")))
# Exact package fingerprint observed from the immutable pinned
# xilinx/vitis-ai-cpu:2.0.0.1103 image digest used by this script.
expected = {
    "python": "3.6.15",
    "torch": "1.7.1",
    "torchvision": "0.8.2",
    "numpy": "1.17.5",
    "pillow": "8.0.0",
}
actual = {
    "python": py,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
}
for k in ("python","torch","torchvision","numpy","pillow"):
    if actual[k] != expected[k]:
        raise AssertionError("%s mismatch: got=%s expected=%s" %
                             (k, actual[k], expected[k]))
print("BASE_LIBRARY_VERSION_GATE=PASS")
print("BASE_LIBRARY_FINGERPRINT=python=%s,torch=%s,torchvision=%s,numpy=%s,pillow=%s" %
      (actual["python"],actual["torch"],actual["torchvision"],
       actual["numpy"],actual["pillow"]))
PY

{
  echo "===== CONDA PACKAGE FINGERPRINT ====="
  conda list | grep -E '^(python|pytorch|torchvision|numpy|pillow|xir|vart|pytorch-nndct|pytorch_nndct)[[:space:]]' || true
  echo "===== PIP PACKAGE FINGERPRINT ====="
  python -m pip freeze | grep -Ei '^(torch|torchvision|numpy|Pillow|pytorch-nndct|pytorch_nndct|xir|vart)==' || true
} | tee "$LOGDIR/base_package_fingerprint.txt"

stage "C - RECHECK MOUNTED SOURCE COMMIT"
ACTUAL="$(cd "$VA" && git rev-parse HEAD)"
echo "LOCAL_VA_COMMIT=$ACTUAL"
[ "$ACTUAL" = "$EXPECTED_VA_COMMIT" ] || die "Vitis-AI source commit changed"

stage "D - VERIFY PREINSTALLED VITIS-AI 2.0 QUANTIZER"
# The immutable Vitis-AI 2.0 CPU image already contains vai_q_pytorch.
# Do not reinstall it: the legacy setup.py declares old dependency pins and
# a pip source install is not the vendor-supported path for this release.
unset CUDA_HOME || true
unset PYTHONPATH || true
export PYTHONNOUSERSITE=1

python - <<'PY' | tee "$LOGDIR/pinned_quantizer_verify.txt"
from __future__ import print_function
import os
import torch, torchvision, xir
import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer

pkg=os.path.realpath(pytorch_nndct.__file__)
prefix=os.path.realpath(os.environ.get("CONDA_PREFIX",""))
print("CONDA_PREFIX="+prefix)
print("PYTORCH_NNDCT_FILE="+pkg)
print("TORCH="+torch.__version__)
print("TORCHVISION="+torchvision.__version__)
print("XIR_FILE="+str(getattr(xir,"__file__","N/A")))

try:
    from pytorch_nndct.version import __version__ as nndct_version
except Exception:
    nndct_version="UNAVAILABLE"
try:
    from pytorch_nndct.version import git_version as nndct_git_version
except Exception:
    nndct_git_version="UNAVAILABLE"

print("PYTORCH_NNDCT_VERSION="+str(nndct_version))
print("PYTORCH_NNDCT_GIT_VERSION="+str(nndct_git_version))

if not prefix:
    raise AssertionError("CONDA_PREFIX is empty")
if not pkg.startswith(prefix + os.sep):
    raise AssertionError("pytorch_nndct is not loaded from active vitis-ai-pytorch environment: "+pkg)
if torch.__version__!="1.7.1":
    raise AssertionError("torch mismatch: "+torch.__version__)
if torchvision.__version__!="0.8.2":
    raise AssertionError("torchvision mismatch: "+torchvision.__version__)

print("PINNED_IMAGE_QUANTIZER_GATE=PASS")
PY

# Record an exact byte-level fingerprint of the quantizer code actually used.
PNDCT_DIR="$(python - <<'PY'
import os, pytorch_nndct
print(os.path.dirname(os.path.realpath(pytorch_nndct.__file__)))
PY
)"
NNDCT_SHARED_DIR="$(python - <<'PY'
import os, nndct_shared
print(os.path.dirname(os.path.realpath(nndct_shared.__file__)))
PY
)"
{
  echo "PYTORCH_NNDCT_DIR=$PNDCT_DIR"
  echo "NNDCT_SHARED_DIR=$NNDCT_SHARED_DIR"
  find "$PNDCT_DIR" "$NNDCT_SHARED_DIR" -type f \
    \( -name '*.py' -o -name '*.so' \) -print0 2>/dev/null \
    | sort -z | xargs -0 -r sha256sum
} | tee "$LOGDIR/pinned_quantizer_file_manifest.sha256"

echo "PINNED_QUANTIZER_FINGERPRINT=PASS"

stage "E - EXACT U250 ARCH"
ARCH=/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json
if [ ! -f "$ARCH" ]; then
  echo "Required: $ARCH"
  echo "Available DPUCADF8H files:"
  find /opt/vitis_ai/compiler/arch/DPUCADF8H -maxdepth 3 -type f -print 2>/dev/null || true
  die "exact U250 arch.json absent; no fallback allowed"
fi
cat "$ARCH" | tee "$LOGDIR/u250_arch.json"
grep -q 'DPUCADF8H_ISA0' "$ARCH" || die "U250 arch target is not DPUCADF8H_ISA0"
echo "ARCH_GATE=PASS"

DRIVER="$WORK/quantize_resnet101.py"
CKPT="$TRAIN/resnet101_cifar100_best.pt"
DATASET="$TRAIN/cifar-100-python.tar.gz"
COMMON=(
  --checkpoint "$CKPT"
  --dataset "$DATASET"
  --dataset-dir "$DDIR"
  --quant-dir "$QDIR"
  --metrics-dir "$MDIR"
  --calib-samples "$CALIB_SAMPLES"
  --test-samples "$TEST_SAMPLES"
  --calib-batch "$CALIB_BATCH"
  --test-batch "$TEST_BATCH"
  --max-top1-drop "$MAX_TOP1_DROP"
  --cpu-threads "$CPU_THREADS"
  --interop-threads "$INTEROP_THREADS"
  --num-workers "$NUM_WORKERS"
)

stage "F - CALIBRATION"
python "$DRIVER" --action calib "${COMMON[@]}" 2>&1 | tee "$LOGDIR/calib.log"
[ -f "$QDIR/quant_info.json" ] || die "quant_info.json missing"

stage "G - FLOAT VS INT8 QUALITY"
python "$DRIVER" --action test "${COMMON[@]}" 2>&1 | tee "$LOGDIR/quant_test.log"
grep -q 'QUANT_ACCURACY_GATE=PASS' "$LOGDIR/quant_test.log" || die "accuracy gate failed"

stage "H - BATCH-1 XMODEL EXPORT"
python "$DRIVER" --action export "${COMMON[@]}" 2>&1 | tee "$LOGDIR/export.log"

mapfile -t QXMODELS < <(find "$QDIR" -maxdepth 1 -type f -name '*_int.xmodel' -print | sort)
if [ "${#QXMODELS[@]}" -ne 1 ]; then
  find "$QDIR" -maxdepth 1 -type f -printf '%f\n' | sort || true
  die "expected exactly one *_int.xmodel"
fi
QX="${QXMODELS[0]}"
echo "QUANTIZED_XMODEL=$QX"
sha256sum "$QX" | tee "$MDIR/quantized_xmodel.sha256"

python - "$QX" <<'PY' | tee "$LOGDIR/quantized_xir_verify.txt"
from __future__ import print_function
import sys, xir
g=xir.Graph.deserialize(sys.argv[1])
print("QUANTIZED_GRAPH_NAME="+g.get_name())
print("QUANTIZED_GRAPH_OPS=%d"%len(g.get_ops()))
assert len(g.get_ops())>0
print("QUANTIZED_XIR_DESERIALIZE=PASS")
PY

stage "I - COMPILE FOR DPUCADF8H/U250"
rm -rf "$CDIR"; mkdir -p "$CDIR"
set +e
vai_c_xir -x "$QX" -a "$ARCH" -o "$CDIR" -n resnet101_cifar100 \
  2>&1 | tee "$LOGDIR/compile.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || die "vai_c_xir rc=$RC"
grep -q 'DPUCADF8H_ISA0' "$LOGDIR/compile.log" ||
  die "compiler log does not prove DPUCADF8H_ISA0"

FINAL="$CDIR/resnet101_cifar100.xmodel"
[ -s "$FINAL" ] || die "compiled xmodel missing"

stage "J - POST-COMPILE DPU VERIFY"
python - "$FINAL" <<'PY' | tee "$LOGDIR/final_xmodel_verify.txt"
from __future__ import print_function
import sys, xir
g=xir.Graph.deserialize(sys.argv[1])
root=g.get_root_subgraph()
subs=[]
def walk(s):
    subs.append(s)
    try: cs=s.toposort_child_subgraph()
    except Exception: cs=[]
    for c in cs: walk(c)
walk(root)
dpu=[]
print("GRAPH_NAME="+g.get_name())
print("GRAPH_OPS=%d"%len(g.get_ops()))
for i,s in enumerate(subs):
    attrs={}
    for k in ["device","dpu_name","fingerprint","runner"]:
        try:
            if s.has_attr(k): attrs[k]=s.get_attr(k)
        except Exception: pass
    print("SUBGRAPH[%d]=%s ATTRS=%r"%(i,s.get_name(),attrs))
    if str(attrs.get("device","")).upper()=="DPU": dpu.append(s)
assert len(g.get_ops())>0
assert len(dpu)>=1, "no DPU subgraph"
print("DPU_SUBGRAPH_COUNT=%d"%len(dpu))
print("FINAL_XMODEL_DPU_GATE=PASS")
PY
grep -q 'FINAL_XMODEL_DPU_GATE=PASS' "$LOGDIR/final_xmodel_verify.txt" ||
  die "final DPU verification failed"

sha256sum "$FINAL" | tee "$MDIR/final_xmodel.sha256"
md5sum "$FINAL" | tee "$MDIR/final_xmodel.md5"
{
  echo "IMAGE_VERSION=${VERSION-UNKNOWN}"
  echo "IMAGE_GIT_HASH=${GIT_HASH-UNKNOWN}"
  echo "PYTHON=$(python --version 2>&1)"
  echo "VAI_C_XIR=$(command -v vai_c_xir)"
  echo "ARCH=$ARCH"
  echo "LOCAL_VITIS_AI_COMMIT=$ACTUAL"
} | tee "$MDIR/final_environment.txt"

HOST_UID="$(stat -c %u "$ROOT" 2>/dev/null || true)"
HOST_GID="$(stat -c %g "$ROOT" 2>/dev/null || true)"
if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ] && [ "$(id -u)" = "0" ]; then
  chown -R "$HOST_UID:$HOST_GID" "$WORK" 2>/dev/null || chmod -R a+rwX "$WORK" || true
else
  chmod -R u+rwX "$WORK" || true
fi
echo "CONTAINER_RESULT=PASS"
echo "FINAL_XMODEL=$FINAL"
INNER
chmod 0755 "$CONTAINER_SCRIPT"

stage "7 - RUN CONTAINER"
CWORK="/workspace/u250_clean_2021_2/xmodel_build/resnet101_cifar100_vai20_${TS}"

set +e
case "$CONTAINER_RUNTIME" in
  udocker)
    # Bind the project directly to /workspace, a standard path in Vitis-AI.
    CWORK_UD="/workspace/xmodel_build/resnet101_cifar100_vai20_${TS}"

    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
      "$UDOCKER_BIN" run \
      --volume="$ROOT:/workspace" \
      --env="VAI_PROJECT_ROOT=/workspace" \
      --env="HOME=/tmp" \
      --env="OMP_NUM_THREADS=$CPU_THREADS" \
      --env="MKL_NUM_THREADS=$CPU_THREADS" \
      --env="OPENBLAS_NUM_THREADS=$CPU_THREADS" \
      --env="NUMEXPR_NUM_THREADS=$CPU_THREADS" \
      --workdir=/workspace \
      "$UDOCKER_CONTAINER" \
      /bin/bash --noprofile --norc \
      "$CWORK_UD/container_build.sh" \
      "$CWORK_UD" "$CALIB_SAMPLES" "$TEST_SAMPLES" "$CALIB_BATCH" "$TEST_BATCH" \
      "$MAX_TOP1_DROP" "$EXPECTED_VA_COMMIT" "$CPU_THREADS" "$INTEROP_THREADS" "$NUM_WORKERS" \
      2>&1 | tee "$WORK/CONTAINER.log"
    RUN_RC=${PIPESTATUS[0]}
    ;;

  docker)
    "${ENGINE[@]}" run --rm \
      -v "$ROOT:/workspace/u250_clean_2021_2:rw" \
      "$PINNED_IMAGE" \
      /bin/bash --noprofile --norc \
      "$CWORK/container_build.sh" \
      "$CWORK" "$CALIB_SAMPLES" "$TEST_SAMPLES" "$CALIB_BATCH" "$TEST_BATCH" \
      "$MAX_TOP1_DROP" "$EXPECTED_VA_COMMIT" "$CPU_THREADS" "$INTEROP_THREADS" "$NUM_WORKERS" \
      2>&1 | tee "$WORK/CONTAINER.log"
    RUN_RC=${PIPESTATUS[0]}
    ;;

  podman)
    # label=disable avoids SELinux relabel attempts on shared/HPC filesystems.
    "${ENGINE[@]}" run --rm \
      --security-opt label=disable \
      -v "$ROOT:/workspace/u250_clean_2021_2:rw" \
      "$PINNED_IMAGE" \
      /bin/bash --noprofile --norc \
      "$CWORK/container_build.sh" \
      "$CWORK" "$CALIB_SAMPLES" "$TEST_SAMPLES" "$CALIB_BATCH" "$TEST_BATCH" \
      "$MAX_TOP1_DROP" "$EXPECTED_VA_COMMIT" "$CPU_THREADS" "$INTEROP_THREADS" "$NUM_WORKERS" \
      2>&1 | tee "$WORK/CONTAINER.log"
    RUN_RC=${PIPESTATUS[0]}
    ;;

  apptainer|singularity)
    "${ENGINE[@]}" exec \
      --cleanenv \
      --bind "$ROOT:/workspace/u250_clean_2021_2" \
      "$SIF" \
      /bin/bash --noprofile --norc \
      "$CWORK/container_build.sh" \
      "$CWORK" "$CALIB_SAMPLES" "$TEST_SAMPLES" "$CALIB_BATCH" "$TEST_BATCH" \
      "$MAX_TOP1_DROP" "$EXPECTED_VA_COMMIT" "$CPU_THREADS" "$INTEROP_THREADS" "$NUM_WORKERS" \
      2>&1 | tee "$WORK/CONTAINER.log"
    RUN_RC=${PIPESTATUS[0]}
    ;;
esac
set -e

[ "$RUN_RC" -eq 0 ] || die "$CONTAINER_RUNTIME execution failed rc=$RUN_RC"
grep -q 'CONTAINER_RESULT=PASS' "$WORK/CONTAINER.log" ||
  die "container exited without PASS marker"

if [ "$CONTAINER_RUNTIME" = "apptainer" ] || [ "$CONTAINER_RUNTIME" = "singularity" ]; then
  SIF_SHA_AFTER="$(sha256sum "$SIF" | awk '{print $1}')"
  echo "SIF_SHA256_AFTER=$SIF_SHA_AFTER"
  [ "$SIF_SHA_BEFORE" = "$SIF_SHA_AFTER" ] ||
    die "SIF changed during execution"
fi

stage "8 - HOST POST-VERIFY / PUBLISH"
FINAL_IN_WORK="$WORK/compiled/resnet101_cifar100.xmodel"
[ -s "$FINAL_IN_WORK" ] || die "final xmodel missing after container"
[ -f "$WORK/metrics/final_xmodel.sha256" ] || die "final SHA report missing"
[ -f "$WORK/metrics/quant_test_metrics.json" ] || die "quant test metrics missing"

EXPECTED_FINAL_SHA="$(awk 'NR==1{print $1}' "$WORK/metrics/final_xmodel.sha256")"
ACTUAL_FINAL_SHA="$(sha256sum "$FINAL_IN_WORK" | awk '{print $1}')"
[ "$EXPECTED_FINAL_SHA" = "$ACTUAL_FINAL_SHA" ] ||
  die "final xmodel changed after container verification"

PUBLISHED="$TRAIN/resnet101_cifar100_DPUCADF8H_U250_${TS}.xmodel"
cp --preserve=mode,timestamps "$FINAL_IN_WORK" "$PUBLISHED"
PUBLISHED_SHA="$(sha256sum "$PUBLISHED" | awk '{print $1}')"
[ "$PUBLISHED_SHA" = "$ACTUAL_FINAL_SHA" ] || die "published SHA mismatch"
sha256sum "$PUBLISHED" > "${PUBLISHED}.sha256"

cat > "$WORK/BUILD_PASS.txt" <<EOF
RESULT=PASS
DATE=$(date)
VITIS=2021.2
VIVADO=2021.2
VITIS_AI=2.0
VITIS_AI_COMMIT=$EXPECTED_VA_COMMIT
CONTAINER_RUNTIME=$CONTAINER_RUNTIME
CPU_THREADS=$CPU_THREADS
INTEROP_THREADS=$INTEROP_THREADS
NUM_WORKERS=$NUM_WORKERS
UDOCKER_EXECMODE=$([ "$CONTAINER_RUNTIME" = "udocker" ] && echo P1 || echo N/A)
OCI_IMAGE=$IMAGE
OCI_DIGEST=$EXPECTED_IMAGE_DIGEST
DPU=DPUCADF8H
BOARD=U250
CHECKPOINT_SHA256=$EXPECTED_CKPT_SHA
FINAL_XMODEL_SHA256=$ACTUAL_FINAL_SHA
FINAL_XMODEL=$PUBLISHED
WORK=$WORK
EOF

cat "$WORK/BUILD_PASS.txt"
echo
echo "================================================================"
echo "RESULT=PASS"
echo "FINAL_XMODEL=$PUBLISHED"
echo "FINAL_SHA256=$ACTUAL_FINAL_SHA"
echo "METRICS=$WORK/metrics/quant_test_metrics.json"
echo "COMPILE_LOG=$WORK/logs/compile.log"
echo "VERIFY_LOG=$WORK/logs/final_xmodel_verify.txt"
echo "FULL_LOG=$LOG"
echo "================================================================"
