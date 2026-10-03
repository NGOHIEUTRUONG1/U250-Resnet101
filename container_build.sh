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
