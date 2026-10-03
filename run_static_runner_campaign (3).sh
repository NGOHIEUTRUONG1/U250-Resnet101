#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <cu-count> <xclbin> <fingerprint-env> <output-root> <run-static:0|1>" >&2
  exit 2
fi

CU_COUNT=$1
XCLBIN=$2
FINGERPRINT_ENV=$3
OUTPUT_ROOT=$4
RUN_STATIC=$5
ROOT=/home/truongnh/u250_clean_2021_2
UDOCKER=$ROOT/toolchain/udocker-1.3.17/udocker/maincmd.py
UDOCKER_STORE=$ROOT/toolchain/udocker_store
CONTAINER=vai20_cpu_7a87fa94_p1fast
XMODEL=$ROOT/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel
BENCH=$ROOT/runtime/benchmark_methods/vart_static_shard_benchmark.py

for path in "$XCLBIN" "$FINGERPRINT_ENV" "$XMODEL" "$BENCH"; do
  [[ -f "$path" ]] || { echo "missing file: $path" >&2; exit 3; }
done

OUTDIR=$OUTPUT_ROOT/${CU_COUNT}cu
mkdir -p "$OUTDIR"
sha256sum "$XCLBIN" "$FINGERPRINT_ENV" "$XMODEL" "$BENCH" > "$OUTDIR/input_sha256.txt"

# Keep XRT's library path out of the Python process that starts udocker.
( source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
  xbutil program --device 0000:1b:00.0 --user "$XCLBIN"
) | tee "$OUTDIR/xbutil_program.txt"

XCLBIN_C=/workspace${XCLBIN#$ROOT}
FP_C=/workspace${FINGERPRINT_ENV#$ROOT}
OUT_C=/workspace${OUTDIR#$ROOT}

INNER="source /opt/xilinx/xrt/setup.sh; source /opt/xilinx/xrm/setup.sh; \
mkdir -p '$OUT_C'; pkill -x xrmd 2>/dev/null || true; rm -f /dev/shm/xrm.data; \
nohup xrmd >'$OUT_C/xrmd.log' 2>&1 & sleep 2; \
printf '%s' '{\"request\":{\"name\":\"load\",\"requestId\":2,\"parameters\":[{\"device\":0,\"xclbin\":\"$XCLBIN_C\"}]}}' > /tmp/load.json; \
xrmadm /tmp/load.json | tee '$OUT_C/xrm_load.json'; \
/usr/bin/python3 /workspace/runtime/benchmark_methods/vart_static_shard_benchmark.py \
  --xmodel /workspace/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel \
  --fingerprint-env '$FP_C' --xclbin '$XCLBIN_C' --device-index 0 \
  --workers 1 --repeats 10 --duration 60 --warmup 20 --output-dir '$OUT_C/workers_1' \
  | tee '$OUT_C/workers_1.log'"

if [[ "$RUN_STATIC" == 1 && "$CU_COUNT" -gt 1 ]]; then
  INNER="$INNER; /usr/bin/python3 /workspace/runtime/benchmark_methods/vart_static_shard_benchmark.py \
    --xmodel /workspace/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel \
    --fingerprint-env '$FP_C' --xclbin '$XCLBIN_C' --device-index 0 \
    --workers '$CU_COUNT' --repeats 10 --duration 60 --warmup 20 --output-dir '$OUT_C/workers_$CU_COUNT' \
    | tee '$OUT_C/workers_$CU_COUNT.log'"
fi

env -u LD_LIBRARY_PATH -u PYTHONPATH UDOCKER_DIR="$UDOCKER_STORE" \
  /usr/bin/python3 "$UDOCKER" run \
  --volume="$ROOT:/workspace" --volume=/dev/dri:/dev/dri --volume=/dev/shm:/dev/shm \
  --volume=/sys/bus/pci:/sys/bus/pci --volume=/sys/class/drm:/sys/class/drm \
  "$CONTAINER" /bin/bash --noprofile --norc -c "$INNER"

find "$OUTDIR" -type f ! -name MANIFEST_SHA256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTDIR/MANIFEST_SHA256.txt"
echo "CAMPAIGN_CONFIG_PASS=${CU_COUNT}CU"
