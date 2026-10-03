#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
IFS=$'\n\t'

ROOT="${ROOT:-/home/truongnh/u250_clean_2021_2}"
BDF="${BDF:-0000:1b:00.0}"
CU_MODE="${CU_MODE:-1}"

case "$CU_MODE" in
  1|2|3) ;;
  *) echo "ERROR: CU_MODE must be 1, 2, or 3"; exit 2 ;;
esac

XMODEL="${XMODEL:-$ROOT/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel}"
VAI_ROOT="${VAI_ROOT:-$ROOT/source/Vitis-AI}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/train}"

if [[ "$CU_MODE" == "1" ]]; then
  XCLBIN="${XCLBIN:-$ROOT/09_results/crossslr_golden_f500_v6_1_20260922_052031/1cu/1cu_DPUCADF8H_300_to_500_CROSSSLR_TIMING_CLOSED.xclbin}"
  EXPECTED_XCLBIN_SHA="${EXPECTED_XCLBIN_SHA:-156afe3e437241fa332ff817b5450c3069c460cf478e189c7eb01f12b0901928}"
  BUILD_ROOT="${BUILD_ROOT:-$ROOT/05_dpu/crossslr_golden_f500_v6_1_20260922_052031}"
else
  : "${XCLBIN:?CU_MODE=2 or 3 requires XCLBIN=/exact/path/to/2cu_or_3cu.xclbin}"
  : "${EXPECTED_XCLBIN_SHA:?CU_MODE=2 or 3 requires EXPECTED_XCLBIN_SHA=<exact sha256>}"
  BUILD_ROOT="${BUILD_ROOT:-$ROOT/05_dpu}"
fi

EXPECTED_XMODEL_SHA="${EXPECTED_XMODEL_SHA:-da6b56491525b3ccdf46f4aff5ba63ac4d8545ebf809d30fd024f7735c581007}"
EXPECTED_VAI_COMMIT="${EXPECTED_VAI_COMMIT:-d02dcb6041663dbc7ecbc0c6af9fafa087a789de}"
EXPECTED_CIFAR100_MD5="${EXPECTED_CIFAR100_MD5:-eb9058c3a382ffc7106e4002c42a8d85}"

MODEL_FP_HEX="0x0700000000000000"
MODEL_FP_DEC="504403158265495552"

UDOCKER_BASE="${UDOCKER_BASE:-$ROOT/toolchain/udocker-1.3.17}"
UDOCKER_STORE="${UDOCKER_STORE:-$ROOT/toolchain/udocker_store}"
UDOCKER_CONTAINER="${UDOCKER_CONTAINER:-vai20_cpu_7a87fa94_p1fast}"

PERF_REPEATS="${PERF_REPEATS:-5}"
BENCH_THREADS="${BENCH_THREADS:-$CU_MODE}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"
LAT_WARMUP="${LAT_WARMUP:-100}"
LAT_ITERS="${LAT_ITERS:-1000}"
SENSOR_PERIOD="${SENSOR_PERIOD:-2}"

[[ "$BENCH_THREADS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: BENCH_THREADS must be positive integer"
  exit 2
}

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/09_results/ieee_${CU_MODE}cu_full_${TS}"
LOG="$ROOT/08_logs/ieee_${CU_MODE}cu_full_${TS}.log"
RAW="$OUT/raw"
TABLES="$OUT/tables"
REPORTS="$OUT/implementation_reports"
STATUS="$OUT/FINAL_STATUS.txt"

mkdir -p "$OUT" "$RAW" "$TABLES" "$REPORTS" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

SENSOR_PID=""
SENSOR_STOP="$OUT/.sensor_stop"
BENCH_ACTIVE="$OUT/.benchmark_active"

cleanup() {
  set +e
  touch "$SENSOR_STOP" 2>/dev/null || true
  if [[ -n "${SENSOR_PID:-}" ]]; then
    wait "$SENSOR_PID" 2>/dev/null || true
  fi
  rm -f "$BENCH_ACTIVE" 2>/dev/null || true
}
trap cleanup EXIT

fail() {
  local reason="$*"
  {
    echo "RESULT=FAIL"
    echo "REASON=$reason"
    echo "OUT=$OUT"
    echo "LOG=$LOG"
  } | tee "$STATUS"
  exit 1
}

echo "============================================================================="
echo " IEEE-GRADE FULL TEST — U250 / DPUCADF8H / ${CU_MODE} CU"
echo "============================================================================="

# -----------------------------------------------------------------------------
# 0. Reproducibility identity
# -----------------------------------------------------------------------------
echo "===== 0. REPRODUCIBILITY IDENTITY ====="

[[ -s "$XCLBIN" ]] || fail "XCLBIN_MISSING"
[[ -s "$XMODEL" ]] || fail "XMODEL_MISSING"
[[ -d "$VAI_ROOT/.git" ]] || fail "VAI_GIT_CHECKOUT_MISSING"

XCLBIN_SHA="$(sha256sum "$XCLBIN" | awk '{print $1}')"
XMODEL_SHA="$(sha256sum "$XMODEL" | awk '{print $1}')"
VAI_COMMIT="$(cd "$VAI_ROOT" && git rev-parse HEAD)"

echo "XCLBIN_SHA256=$XCLBIN_SHA"
echo "XMODEL_SHA256=$XMODEL_SHA"
echo "VITIS_AI_COMMIT=$VAI_COMMIT"

[[ "$XCLBIN_SHA" == "$EXPECTED_XCLBIN_SHA" ]] || fail "XCLBIN_SHA_MISMATCH"
[[ "$XMODEL_SHA" == "$EXPECTED_XMODEL_SHA" ]] || fail "XMODEL_SHA_MISMATCH"
[[ "$VAI_COMMIT" == "$EXPECTED_VAI_COMMIT" ]] || fail "VAI_COMMIT_MISMATCH"

DATASET_ARCHIVE=""
for C in \
  "$TRAIN_DIR/cifar-100-python.tar.gz" \
  "$ROOT/train/cifar-100-python.tar.gz" \
  "/datastore/truongnh/resnet_paper/cifar-100-python.tar.gz"
do
  if [[ -s "$C" ]]; then
    DATASET_ARCHIVE="$C"
    break
  fi
done
[[ -n "$DATASET_ARCHIVE" ]] || fail "CIFAR100_ARCHIVE_NOT_FOUND"

DATASET_MD5="$(md5sum "$DATASET_ARCHIVE" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "$DATASET_ARCHIVE" | awk '{print $1}')"

echo "CIFAR100_ARCHIVE=$DATASET_ARCHIVE"
echo "CIFAR100_MD5=$DATASET_MD5"
echo "CIFAR100_SHA256=$DATASET_SHA256"

[[ "$DATASET_MD5" == "$EXPECTED_CIFAR100_MD5" ]] ||
  fail "CIFAR100_OFFICIAL_MD5_MISMATCH"

{
  echo "DATE_ISO=$(date -Is)"
  echo "HOSTNAME=$(hostname)"
  echo "KERNEL=$(uname -a)"
  echo "===== CPU ====="
  lscpu 2>&1 || true
  echo "===== MEMORY ====="
  free -h 2>&1 || true
  echo "===== PCI ====="
  lspci -nn 2>&1 | grep -i -E 'xilinx|display|vga|3d' || true
  echo "===== XRT ====="
  /opt/xilinx/xrt/bin/xbutil --version 2>&1 || true
  echo "===== VITIS ====="
  /home/truongnh/local_vitis/Vitis/2021.2/bin/v++ --version 2>&1 || true
  echo "===== VIVADO ====="
  /home/truongnh/local_vitis/Vivado/2021.2/bin/vivado -version 2>&1 || true
  echo "===== GIT ====="
  git --version 2>&1 || true
} | tee "$RAW/environment.txt"

for F in \
  "$TRAIN_DIR/float_test_metrics.json" \
  "$TRAIN_DIR/quant_test_metrics.json"
do
  [[ -f "$F" ]] && cp -p "$F" "$RAW/$(basename "$F")"
done

# -----------------------------------------------------------------------------
# 1. Static implementation evidence
# -----------------------------------------------------------------------------
echo "===== 1. VIVADO IMPLEMENTATION EVIDENCE ====="

{
  echo "BUILD_ROOT=$BUILD_ROOT"
  if [[ -d "$BUILD_ROOT" ]]; then
    find "$BUILD_ROOT" -type f \
      \( -iname '*.rpt' -o -iname '*.log' -o -iname '*.jou' \) \
      -print | sort
  else
    echo "BUILD_ROOT_MISSING"
  fi
} | tee "$RAW/implementation_file_index.txt"

if [[ -d "$BUILD_ROOT" ]]; then
  while IFS= read -r F; do
    B="$(basename "$F")"
    DEST="$REPORTS/$B"
    N=1
    while [[ -e "$DEST" ]]; do
      DEST="$REPORTS/${B%.*}_${N}.${B##*.}"
      N=$((N+1))
    done
    cp -p "$F" "$DEST" 2>/dev/null || true
  done < <(
    find "$BUILD_ROOT" -type f \
      \( -iname '*timing*summary*.rpt' \
         -o -iname '*utilization*.rpt' \
         -o -iname '*power*.rpt' \
         -o -iname '*clock*.rpt' \
         -o -iname '*route*status*.rpt' \) \
      -print | sort
  )

  {
    echo "===== TIMING EVIDENCE ====="
    grep -RniE \
      'Design Timing Summary|WNS\(ns\)|WHS\(ns\)|Slack \(MET\)|Timing constraints are met' \
      "$REPORTS" 2>/dev/null | head -n 300 || true

    echo "===== UTILIZATION EVIDENCE ====="
    grep -RniE \
      'CLB LUTs|Slice LUTs|CLB Registers|Slice Registers|Block RAM|BRAM|URAM|DSPs|DSP48' \
      "$REPORTS" 2>/dev/null | head -n 500 || true

    echo "===== POWER EVIDENCE ====="
    grep -RniE \
      'Total On-Chip Power|Device Static|Dynamic|Junction Temperature' \
      "$REPORTS" 2>/dev/null | head -n 300 || true
  } | tee "$RAW/implementation_evidence_extract.txt"
fi

if [[ -x /opt/xilinx/xrt/bin/xclbinutil ]]; then
  /opt/xilinx/xrt/bin/xclbinutil --info --input "$XCLBIN" \
    > "$RAW/xclbinutil_info.txt" 2>&1 || true
fi

# -----------------------------------------------------------------------------
# 2. Card health and exact xclbin program
# -----------------------------------------------------------------------------
echo "===== 2. CARD HEALTH + PROGRAM ====="

XBUTIL="/opt/xilinx/xrt/bin/xbutil"
[[ -x "$XBUTIL" ]] || fail "XBUTIL_MISSING"

"$XBUTIL" examine --device "$BDF" --report all \
  > "$RAW/xbutil_examine_pre_all.txt" 2>&1 || true

set +e
"$XBUTIL" validate --device "$BDF" --run quick \
  > "$RAW/xbutil_validate_quick.txt" 2>&1
XB_VALIDATE_RC=$?
set -e
echo "XBUTIL_VALIDATE_QUICK_RC=$XB_VALIDATE_RC"

"$XBUTIL" program --device "$BDF" --user "$XCLBIN" \
  > "$RAW/xbutil_program.txt" 2>&1 ||
  { cat "$RAW/xbutil_program.txt"; fail "XBUTIL_PROGRAM_FAILED"; }

CARD_OK=0
for _ in $(seq 1 10); do
  set +e
  "$XBUTIL" examine --device "$BDF" \
    > "$RAW/xbutil_after_program.txt" 2>&1
  RC=$?
  set -e
  if [[ "$RC" -eq 0 ]] &&
     grep -q 'DPUCADF8H:DPUCADF8H_' "$RAW/xbutil_after_program.txt" &&
     grep -Eq 'Data[[:space:]]*:[[:space:]]*300 MHz' "$RAW/xbutil_after_program.txt" &&
     grep -Eq 'Kernel[[:space:]]*:[[:space:]]*300 MHz' "$RAW/xbutil_after_program.txt"; then
    CARD_OK=1
    break
  fi
  sleep 1
done

cat "$RAW/xbutil_after_program.txt"
[[ "$CARD_OK" -eq 1 ]] || fail "CARD_GATE_FAILED"

POST_UUID="$(
  awk '/^Xclbin UUID/{seen=1;next} seen && NF{print $1;exit}' \
    "$RAW/xbutil_after_program.txt"
)"
[[ -n "$POST_UUID" ]] || fail "XCLBIN_UUID_PARSE_FAILED"

mapfile -t CU_NAMES < <(
  grep -oE 'DPUCADF8H:DPUCADF8H_[0-9]+' "$RAW/xbutil_after_program.txt" | sort -u
)
CU_COUNT="${#CU_NAMES[@]}"
[[ "$CU_COUNT" == "$CU_MODE" ]] ||
  fail "EXPECTED_${CU_MODE}_DPU_CU_GOT_${CU_COUNT}"

CU_NAMES_FILE="$OUT/cu_names.txt"
FP_ENV_FILE="$OUT/fingerprint_env.tsv"
: > "$CU_NAMES_FILE"
: > "$FP_ENV_FILE"
for CU_NAME in "${CU_NAMES[@]}"; do
  echo "$CU_NAME" >> "$CU_NAMES_FILE"
  printf '%s.fingerprint\t%s\n' "$CU_NAME" "$MODEL_FP_DEC" >> "$FP_ENV_FILE"
done

echo "POST_PROGRAM_UUID=$POST_UUID"
echo "DPU_CU_COUNT=$CU_COUNT"
cat "$CU_NAMES_FILE"
echo "CARD_PROGRAM_GATE=PASS"

for R in electrical thermal mechanical pcie-info memory; do
  "$XBUTIL" examine --device "$BDF" --report "$R" \
    > "$RAW/xbutil_${R}_baseline.txt" 2>&1 || true
done

# -----------------------------------------------------------------------------
# 3. Exact BDF -> XRT index -> UUID
# -----------------------------------------------------------------------------
echo "===== 3. PYXRT BDF / INDEX / UUID ====="

PYXRT_FILE="$(
  find /opt/xilinx/xrt -type f -name 'pyxrt*.so' -print 2>/dev/null | head -1
)"
[[ -n "$PYXRT_FILE" ]] || fail "PYXRT_NOT_FOUND"
PYXRT_DIR="$(dirname "$PYXRT_FILE")"

MAP_PY="$OUT/map.py"
cat > "$MAP_PY" <<'PY'
import sys
import pyxrt

want_bdf = sys.argv[1].lower()
want_uuid = sys.argv[2].lower()
hits = []

for i in range(64):
    try:
        d = pyxrt.device(i)
    except Exception:
        continue
    bdf = str(d.get_info(pyxrt.xrt_info_device.bdf))
    uuid = d.get_xclbin_uuid().to_string()
    print("INDEX=%d BDF=%s UUID=%s" % (i, bdf, uuid))
    if bdf.lower() == want_bdf:
        hits.append((i, bdf, uuid))

print("TARGET_MATCH_COUNT=%d" % len(hits))
if len(hits) != 1:
    raise SystemExit(20)

i, bdf, uuid = hits[0]
if uuid.lower() != want_uuid:
    raise SystemExit(21)

print("TARGET_INDEX=%d" % i)
print("TARGET_BDF=%s" % bdf)
print("TARGET_UUID=%s" % uuid)
print("PYXRT_GATE=PASS")
PY

env -i \
  HOME="${HOME:-/home/truongnh}" \
  USER="${USER:-truongnh}" \
  LOGNAME="${LOGNAME:-${USER:-truongnh}}" \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  PYTHONPATH="$PYXRT_DIR" \
  XILINX_XRT=/opt/xilinx/xrt \
  /usr/bin/python3 "$MAP_PY" "$BDF" "$POST_UUID" \
  > "$RAW/pyxrt_map.txt" 2>&1 ||
  { cat "$RAW/pyxrt_map.txt"; fail "PYXRT_GATE_FAILED"; }

cat "$RAW/pyxrt_map.txt"
grep -qx 'PYXRT_GATE=PASS' "$RAW/pyxrt_map.txt" ||
  fail "PYXRT_PASS_MARKER_MISSING"

XRT_INDEX="$(awk -F= '/^TARGET_INDEX=/{print $2;exit}' "$RAW/pyxrt_map.txt")"
[[ "$XRT_INDEX" =~ ^[0-9]+$ ]] || fail "XRT_INDEX_PARSE_FAILED"

# -----------------------------------------------------------------------------
# 4. Resolve pinned VAI2 udocker
# -----------------------------------------------------------------------------
echo "===== 4. PINNED VAI2 UDOCKER ====="

export UDOCKER_DIR="$UDOCKER_STORE"
export UDOCKER_DEFAULT_EXECUTION_MODE=P2
export PROOT_TMP_DIR="$ROOT/toolchain/proot_tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$PROOT_TMP_DIR"

UDOCKER_BIN=""
UDOCKER_MODE=""

for C in \
  "$UDOCKER_BASE/udocker/udocker" \
  "$UDOCKER_BASE/udocker" \
  "$UDOCKER_BASE/udocker/maincmd.py"
do
  [[ -f "$C" ]] || continue
  chmod +x "$C" 2>/dev/null || true

  set +e
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
    PYTHONNOUSERSITE=1 \
    UDOCKER_DIR="$UDOCKER_DIR" \
    UDOCKER_DEFAULT_EXECUTION_MODE="$UDOCKER_DEFAULT_EXECUTION_MODE" \
    PROOT_TMP_DIR="$PROOT_TMP_DIR" \
    "$C" version > "$RAW/udocker_probe.txt" 2>&1
  RC=$?
  set -e

  if [[ "$RC" -eq 0 ]]; then
    UDOCKER_BIN="$C"
    UDOCKER_MODE=direct
    break
  fi

  if [[ "$(basename "$C")" == "maincmd.py" ]]; then
    set +e
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
      PYTHONNOUSERSITE=1 \
      UDOCKER_DIR="$UDOCKER_DIR" \
      UDOCKER_DEFAULT_EXECUTION_MODE="$UDOCKER_DEFAULT_EXECUTION_MODE" \
      PROOT_TMP_DIR="$PROOT_TMP_DIR" \
      /usr/bin/python3 "$C" version > "$RAW/udocker_probe.txt" 2>&1
    RC=$?
    set -e
    if [[ "$RC" -eq 0 ]]; then
      UDOCKER_BIN="$C"
      UDOCKER_MODE=python3
      break
    fi
  fi
done

[[ -n "$UDOCKER_BIN" ]] ||
  { cat "$RAW/udocker_probe.txt" 2>/dev/null || true; fail "UDOCKER_NOT_RUNNABLE"; }

cat "$RAW/udocker_probe.txt"

run_udocker() {
  if [[ "$UDOCKER_MODE" == "direct" ]]; then
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
      PYTHONNOUSERSITE=1 \
      UDOCKER_DIR="$UDOCKER_DIR" \
      UDOCKER_DEFAULT_EXECUTION_MODE="$UDOCKER_DEFAULT_EXECUTION_MODE" \
      PROOT_TMP_DIR="$PROOT_TMP_DIR" \
      "$UDOCKER_BIN" "$@"
  else
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
      PYTHONNOUSERSITE=1 \
      UDOCKER_DIR="$UDOCKER_DIR" \
      UDOCKER_DEFAULT_EXECUTION_MODE="$UDOCKER_DEFAULT_EXECUTION_MODE" \
      PROOT_TMP_DIR="$PROOT_TMP_DIR" \
      /usr/bin/python3 "$UDOCKER_BIN" "$@"
  fi
}

# -----------------------------------------------------------------------------
# 5. Host electrical / thermal sampler
# -----------------------------------------------------------------------------
echo "===== 5. SENSOR SAMPLER ====="

rm -f "$SENSOR_STOP" "$BENCH_ACTIVE"

sensor_loop() {
  local N=0
  while [[ ! -e "$SENSOR_STOP" ]]; do
    N=$((N+1))
    local ACTIVE=0
    [[ -e "$BENCH_ACTIVE" ]] && ACTIVE=1
    {
      echo "SAMPLE_BEGIN index=$N epoch=$(date +%s) iso=$(date -Is) benchmark_active=$ACTIVE"
      echo "--- ELECTRICAL ---"
      "$XBUTIL" examine --device "$BDF" --report electrical 2>&1 || true
      echo "--- THERMAL ---"
      "$XBUTIL" examine --device "$BDF" --report thermal 2>&1 || true
      echo "SAMPLE_END index=$N"
    } >> "$RAW/sensors_runtime.txt"
    sleep "$SENSOR_PERIOD"
  done
}

sensor_loop &
SENSOR_PID=$!

# -----------------------------------------------------------------------------
# 6. One container session: XRM + query + 10k accuracy + latency + repeated FPS
# -----------------------------------------------------------------------------
echo "===== 6. FULL VAI2 1-CU RUNTIME SUITE ====="

[[ -d /dev/dri ]] || fail "HOST_DEV_DRI_MISSING"
[[ -d /dev/shm ]] || fail "HOST_DEV_SHM_MISSING"
[[ -d /sys/bus/pci ]] || fail "HOST_PCI_SYSFS_MISSING"

VOL_ARGS=(
  "--volume=$ROOT:/workspace"
  "--volume=/dev/dri:/dev/dri"
  "--volume=/dev/shm:/dev/shm"
  "--volume=/sys/bus/pci:/sys/bus/pci"
)
[[ -d /sys/class/drm ]] &&
  VOL_ARGS+=("--volume=/sys/class/drm:/sys/class/drm")
[[ -d /opt/xilinx/dsa ]] &&
  VOL_ARGS+=("--volume=/opt/xilinx/dsa:/opt/xilinx/dsa")
[[ -d /opt/xilinx/overlaybins ]] &&
  VOL_ARGS+=("--volume=/opt/xilinx/overlaybins:/opt/xilinx/overlaybins")

mapfile -t XCLMGMT_NODES < <(
  find /dev -maxdepth 1 -type c -name 'xclmgmt*' -print 2>/dev/null | sort
)
for D in "${XCLMGMT_NODES[@]}"; do
  VOL_ARGS+=("--volume=$D:$D")
done

C_OUT="/workspace/${OUT#$ROOT/}"
C_XCLBIN="/workspace/${XCLBIN#$ROOT/}"
C_XMODEL="/workspace/${XMODEL#$ROOT/}"
if [[ "$DATASET_ARCHIVE" == "$ROOT/"* ]]; then
  C_DATASET="/workspace/${DATASET_ARCHIVE#$ROOT/}"
else
  VOL_ARGS+=("--volume=$DATASET_ARCHIVE:/ieee_data/cifar-100-python.tar.gz")
  C_DATASET="/ieee_data/cifar-100-python.tar.gz"
fi

INNER="$OUT/container_ieee_suite.sh"
cat > "$INNER" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

BDF="$1"
DEVICE_INDEX="$2"
XCLBIN="$3"
XMODEL="$4"
DATASET_ARCHIVE="$5"
EXPECTED_DATASET_MD5="$6"
CU_COUNT_EXPECTED="$7"
FP_ENV_FILE="$8"
PERF_REPEATS="$9"
BENCH_THREADS="${10}"
BENCH_TIMEOUT="${11}"
LAT_WARMUP="${12}"
LAT_ITERS="${13}"
OUT="${14}"

inner_fail() {
  echo "INNER_RESULT=FAIL"
  echo "INNER_REASON=$1"
  exit "${2:-90}"
}

set +u
source /opt/xilinx/xrt/setup.sh
set -u

/opt/xilinx/xrt/bin/xbutil --version
/usr/bin/python3 --version

/opt/xilinx/xrt/bin/xbutil examine --device "$BDF" \
  > "$OUT/raw/container_card_start.txt" 2>&1
CONTAINER_CU_COUNT="$(
  grep -oE 'DPUCADF8H:DPUCADF8H_[0-9]+' "$OUT/raw/container_card_start.txt" |
    sort -u | wc -l | tr -d ' '
)"
[[ "$CONTAINER_CU_COUNT" == "$CU_COUNT_EXPECTED" ]] ||
  inner_fail "CONTAINER_CU_COUNT_EXPECTED_${CU_COUNT_EXPECTED}_GOT_${CONTAINER_CU_COUNT}" 91

# ---- XRM ----
[[ -d /opt/xilinx/xrm ]] || inner_fail "CONTAINER_XRM_NOT_INSTALLED" 92
[[ -f /opt/xilinx/xrm/setup.sh ]] || inner_fail "CONTAINER_XRM_SETUP_MISSING" 93

set +u
source /opt/xilinx/xrm/setup.sh
set -u

XRMD_BIN="$(command -v xrmd || true)"
XRMADM_BIN="$(command -v xrmadm || true)"
[[ -n "$XRMD_BIN" && -x "$XRMD_BIN" ]] || inner_fail "XRMD_BINARY_MISSING" 94
[[ -n "$XRMADM_BIN" && -x "$XRMADM_BIN" ]] || inner_fail "XRMADM_BINARY_MISSING" 95

SERVICE_FILE=""
for F in /etc/systemd/system/xrmd.service /lib/systemd/system/xrmd.service /usr/lib/systemd/system/xrmd.service; do
  [[ -f "$F" ]] && { SERVICE_FILE="$F"; break; }
done

XRMD_EXEC=""
if [[ -n "$SERVICE_FILE" ]]; then
  cp -p "$SERVICE_FILE" "$OUT/raw/xrmd.service"
  XRMD_EXEC="$(
    sed -n 's/^[[:space:]]*ExecStart=[[:space:]]*//p' "$SERVICE_FILE" | head -1
  )"
fi

if ! pgrep -af '(^|/)xrmd([[:space:]]|$)' >/dev/null 2>&1; then
  rm -f /dev/shm/xrm.data 2>/dev/null || true
  if [[ -n "$XRMD_EXEC" ]]; then
    nohup /bin/bash -c "$XRMD_EXEC" > "$OUT/raw/xrmd.log" 2>&1 &
  else
    nohup "$XRMD_BIN" > "$OUT/raw/xrmd.log" 2>&1 &
  fi

  READY=0
  for _ in $(seq 1 20); do
    if pgrep -af '(^|/)xrmd([[:space:]]|$)' >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 0.5
  done
  [[ "$READY" -eq 1 ]] || inner_fail "XRMD_START_FAILED" 96
fi

pgrep -af '(^|/)xrmd([[:space:]]|$)' | tee "$OUT/raw/xrmd_process.txt"
echo "XRMD_PROCESS_GATE=PASS"

LIST_JSON="$OUT/raw/xrm_list.json"
cat > "$LIST_JSON" <<'JSON'
{
  "request": {
    "name": "list",
    "requestId": 1
  }
}
JSON

"$XRMADM_BIN" "$LIST_JSON" > "$OUT/raw/xrm_list_before.txt" 2>&1 ||
  inner_fail "XRMADM_LIST_BEFORE_FAILED" 97
grep -qi '"status"[[:space:]]*:[[:space:]]*"ok"' "$OUT/raw/xrm_list_before.txt" ||
  inner_fail "XRM_DAEMON_CLIENT_STATUS_NOT_OK" 98
echo "XRM_DAEMON_CLIENT_GATE=PASS"

LOAD_JSON="$OUT/raw/xrm_load_exact.json"
cat > "$LOAD_JSON" <<JSON
{
  "request": {
    "name": "load",
    "requestId": 2,
    "parameters": [
      {
        "device": $DEVICE_INDEX,
        "xclbin": "$XCLBIN"
      }
    ]
  }
}
JSON

"$XRMADM_BIN" "$LOAD_JSON" > "$OUT/raw/xrm_load.txt" 2>&1 ||
  inner_fail "XRMADM_LOAD_FAILED" 99
grep -qi '"status"[[:space:]]*:[[:space:]]*"ok"' "$OUT/raw/xrm_load.txt" ||
  inner_fail "XRMADM_LOAD_STATUS_NOT_OK" 100
echo "XRM_EXACT_XCLBIN_LOAD=PASS"

"$XRMADM_BIN" "$LIST_JSON" > "$OUT/raw/xrm_list_loaded.txt" 2>&1 ||
  inner_fail "XRMADM_LIST_AFTER_FAILED" 101
# XRM may retain extra/stale resource metadata across daemon lifetime.
# Do NOT equate total XRM DB entry count with physical CU count.
#
# Physical CU count is already fail-closed from xbutil after exact programming.
# Here we only require that every CU on the currently programmed card is
# represented in XRM's resource database.
XRM_EXPECTED_MATCHES=0
while IFS=$'\t' read -r FP_KEY FP_VALUE; do
  [[ -n "$FP_KEY" ]] || continue

  CU_FULL="${FP_KEY%.fingerprint}"
  CU_INSTANCE="${CU_FULL#*:}"

  if grep -Fq "$CU_INSTANCE" "$OUT/raw/xrm_list_loaded.txt"; then
    echo "XRM_EXPECTED_CU_PRESENT=$CU_INSTANCE"
    XRM_EXPECTED_MATCHES=$((XRM_EXPECTED_MATCHES + 1))
  else
    echo "XRM_EXPECTED_CU_MISSING=$CU_INSTANCE"
    inner_fail "XRM_DB_MISSING_EXPECTED_CU_${CU_INSTANCE}" 102
  fi
done < "$FP_ENV_FILE"

[[ "$XRM_EXPECTED_MATCHES" == "$CU_COUNT_EXPECTED" ]] ||
  inner_fail "XRM_EXPECTED_MATCH_COUNT_${XRM_EXPECTED_MATCHES}_EXPECTED_${CU_COUNT_EXPECTED}" 103

# Record all DPU-like entries for audit, but extras are NOT a failure because
# xbutil is the authoritative source for the current physical xclbin/CU count.
grep -oE 'DPUCADF8H_[0-9]+' "$OUT/raw/xrm_list_loaded.txt" |
  sort -u > "$OUT/raw/xrm_dpu_entries_observed.txt" || true

echo "XRM_DB_EXPECTED_CUS=$XRM_EXPECTED_MATCHES"
echo "XRM_RESOURCE_DATABASE_GATE=PASS"

# ---- Query / xmodel contract ----
RUNTIME_GATE_PY="$OUT/raw/runtime_gate.py"
cat > "$RUNTIME_GATE_PY" <<'PY'
import os
import subprocess
import sys

fp_env_file, xclbin, xmodel, idx, out, expected_cus = sys.argv[1:7]
env = os.environ.copy()
with open(fp_env_file) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        name, value = line.split("\t", 1)
        env[name] = value
        print("FP_OVERRIDE=%s=%s" % (name, value))
env["XLNX_VART_FIRMWARE"] = xclbin
env["XLNX_ENABLE_DEVICES"] = idx

def run(label, argv):
    p = subprocess.run(
        argv, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True
    )
    text = p.stdout or ""
    open(os.path.join(out, label + ".txt"), "w").write(text)
    print(text)
    if p.returncode != 0:
        raise SystemExit(10)
    return text

q = run("xdputil_query", ["/usr/bin/python3","-m","xdputil","query"])
if "TARGET_FACTORY_INVALID_TYPE" in q:
    raise SystemExit(11)
if "DPUCADF8H" not in q:
    raise SystemExit(12)
import re
actual_cus = len(set(re.findall(r'DPUCADF8H:DPUCADF8H_[0-9]+', q)))
if actual_cus != int(expected_cus):
    print("QUERY_CU_COUNT_EXPECTED=%s ACTUAL=%d" % (expected_cus, actual_cus))
    raise SystemExit(14)
print("XDPUTIL_QUERY_CU_COUNT=%d" % actual_cus)
print("XDPUTIL_QUERY_GATE=PASS")

x = run("xmodel_list", ["/usr/bin/python3","-m","xdputil","xmodel",xmodel,"-l"])
n = "".join(x.split())
for token in [
    '"device":"DPU"',
    '"shape":"[1,224,224,3]"',
    '"shape":"[1,1,1,100]"',
    '"fixpos":5',
    '"fixpos":2',
]:
    if token not in n:
        print("MISSING_XMODEL_TOKEN=%s" % token)
        raise SystemExit(13)
print("XMODEL_CONTRACT_GATE=PASS")
PY

/usr/bin/python3 "$RUNTIME_GATE_PY" \
  "$FP_ENV_FILE" "$XCLBIN" "$XMODEL" "$DEVICE_INDEX" "$OUT/raw" "$CU_COUNT_EXPECTED" ||
  inner_fail "QUERY_OR_XMODEL_CONTRACT_FAILED" 103

# ---- Official CIFAR-100 gate ----
ACT_MD5="$(md5sum "$DATASET_ARCHIVE" | awk '{print $1}')"
[[ "$ACT_MD5" == "$EXPECTED_DATASET_MD5" ]] ||
  inner_fail "CIFAR100_MD5_MISMATCH" 104

# ---- Actual preprocessing environment gate ----
# Preprocessing is isolated in the installed vitis-ai-pytorch interpreter.
# VART/XRT remain in /usr/bin/python3; data crosses only as raw INT8 bytes.
echo "===== 6D.5. PREPROCESSING ENVIRONMENT GATE ====="
CONDA_ENV="/opt/vitis_ai/conda/envs/vitis-ai-pytorch"
PREPROC_PY="$CONDA_ENV/bin/python"
[[ -x "$PREPROC_PY" ]] ||
  inner_fail "VITIS_AI_PYTORCH_PYTHON_MISSING" 105

"$PREPROC_PY" - <<'PY' 2>&1 | tee "$OUT/raw/preprocessing_environment.txt"
from __future__ import print_function
import os, sys, numpy, PIL, torch, torchvision
from torchvision import transforms
print("PREPROCESS_PYTHON=%s" % ("%d.%d.%d" % sys.version_info[:3]))
print("PREPROCESS_NUMPY=%s" % numpy.__version__)
print("PREPROCESS_PILLOW=%s" % PIL.__version__)
print("PREPROCESS_TORCH=%s" % torch.__version__)
print("PREPROCESS_TORCHVISION=%s" % torchvision.__version__)
print("PREPROCESS_PIL_FILE=%s" % os.path.realpath(PIL.__file__))
print("PREPROCESS_TORCHVISION_FILE=%s" % os.path.realpath(torchvision.__file__))
tf=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071,0.4867,0.4408),
        (0.2675,0.2565,0.2761),
    ),
])
print("PREPROCESS_TRANSFORM=%r" % tf)
print("PREPROCESS_ENV_GATE=PASS")
PY
grep -qx 'PREPROCESS_ENV_GATE=PASS' "$OUT/raw/preprocessing_environment.txt" ||
  inner_fail "PREPROCESS_ENV_GATE_FAILED" 106

# ---- Full 10,000-image hardware accuracy ----
ACC_PRE_PY="$OUT/raw/preprocess_stream.py"
cat > "$ACC_PRE_PY" <<'PY'
from __future__ import print_function
import json, os, pickle, struct, sys, tarfile
archive, work, input_fix, meta_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

import numpy as np
import PIL, torch, torchvision
from PIL import Image
from torchvision import transforms

os.makedirs(work, exist_ok=True)
root=os.path.join(work,"cifar-100-python")
test_file=os.path.join(root,"test")
if not os.path.isfile(test_file):
    with tarfile.open(archive,"r:gz") as tf:
        base=os.path.abspath(work)
        for m in tf.getmembers():
            d=os.path.abspath(os.path.join(work,m.name))
            if not (d==base or d.startswith(base+os.sep)):
                raise RuntimeError("unsafe tar path")
        tf.extractall(work)

with open(test_file,"rb") as f:
    d=pickle.load(f,encoding="bytes")
images=d[b"data"]
labels=np.asarray(d[b"fine_labels"],dtype=np.int64)
if images.shape!=(10000,3072) or labels.shape!=(10000,):
    raise RuntimeError("CIFAR-100 shape mismatch")
images=images.reshape(-1,3,32,32).transpose(0,2,3,1)

tf=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071,0.4867,0.4408),
        (0.2675,0.2565,0.2761),
    ),
])
meta={
  "python":"%d.%d.%d"%sys.version_info[:3],
  "numpy":np.__version__,
  "pillow":PIL.__version__,
  "torch":torch.__version__,
  "torchvision":torchvision.__version__,
  "transform":repr(tf),
  "input_fix_point":input_fix,
  "samples":10000,
}
with open(meta_path,"w") as f:
    json.dump(meta,f,indent=2,sort_keys=True)

scale=float(2**input_fix)
buf=sys.stdout.buffer
for i in range(10000):
    im=Image.fromarray(images[i],mode="RGB")
    t=tf(im)                         # CHW float32
    a=t.numpy().transpose(1,2,0)    # HWC float32
    q=(a*np.float32(scale)).astype(np.int8,copy=False)
    if q.shape!=(224,224,3):
        raise RuntimeError("preprocessed shape mismatch")
    buf.write(struct.pack("<B",int(labels[i])))
    buf.write(q.tobytes(order="C"))
buf.flush()
PY

ACC_PY="$OUT/raw/hw_accuracy_cifar100.py"
cat > "$ACC_PY" <<'PY'
from __future__ import print_function
import json, os, struct, subprocess, sys, time
xmodel,archive,work,fp_env_file,xclbin,device_index,preproc_python,preproc_script,preproc_meta,preproc_stderr,out_json,out_csv=sys.argv[1:13]

with open(fp_env_file) as f:
    for line in f:
        line=line.rstrip("\n")
        if not line: continue
        name,value=line.split("\t",1)
        os.environ[name]=value
os.environ["XLNX_VART_FIRMWARE"]=xclbin
os.environ["XLNX_ENABLE_DEVICES"]=device_index

import numpy as np
import xir, vart

g=xir.Graph.deserialize(xmodel)
root=g.get_root_subgraph()
dpu=[sg for sg in root.toposort_child_subgraph()
     if sg.has_attr("device") and str(sg.get_attr("device")).upper()=="DPU"]
if len(dpu)!=1:
    raise RuntimeError("xmodel DPU subgraph count=%d"%len(dpu))
runner=vart.Runner.create_runner(dpu[0],"run")
inputs=runner.get_input_tensors()
outputs=runner.get_output_tensors()

if len(inputs)!=1:
    raise RuntimeError("expected exactly 1 runner input tensor, got %d"%len(inputs))
if len(outputs)!=1:
    raise RuntimeError("expected exactly 1 runner output tensor, got %d"%len(outputs))

it=inputs[0]
ot=outputs[0]
in_dims=tuple(int(x) for x in it.dims)
out_dims=tuple(int(x) for x in ot.dims)
in_fix=int(it.get_attr("fix_point"))
out_fix=int(ot.get_attr("fix_point"))

def tensor_name(t):
    try:
        return str(t.name)
    except Exception:
        try:
            return str(t.get_name())
        except Exception:
            return "UNKNOWN"

def numel(dims):
    n=1
    for d in dims:
        n*=int(d)
    return n

print("RUNNER_INPUT_NAME=%s"%tensor_name(it))
print("RUNNER_INPUT_DIMS=%r"%(in_dims,))
print("RUNNER_INPUT_NUMEL=%d"%numel(in_dims))
print("RUNNER_INPUT_FIX_POINT=%d"%in_fix)
print("RUNNER_OUTPUT_NAME=%s"%tensor_name(ot))
print("RUNNER_OUTPUT_DIMS=%r"%(out_dims,))
print("RUNNER_OUTPUT_NUMEL=%d"%numel(out_dims))
print("RUNNER_OUTPUT_FIX_POINT=%d"%out_fix)

# VART exposes the actual runtime batch in dimension 0.
# The current measured contract is B=4. Keep this generic for later 3-CU use.
if len(in_dims)!=4:
    raise RuntimeError("runner input must be rank-4, got %r"%(in_dims,))

runner_batch=int(in_dims[0])
if runner_batch <= 0:
    raise RuntimeError("invalid runner batch: %d"%runner_batch)

if in_dims[1:]==(224,224,3):
    input_layout="NHWC"
elif in_dims[1:]==(3,224,224):
    input_layout="NCHW"
else:
    raise RuntimeError(
        "unsupported runner input dims %r; expected (B,224,224,3) "
        "or (B,3,224,224)"%(in_dims,)
    )

# Output must carry the same runtime batch and exactly 100 logits per image.
if len(out_dims)<2 or int(out_dims[0])!=runner_batch:
    raise RuntimeError(
        "runner output batch mismatch: input batch=%d output dims=%r"%
        (runner_batch,out_dims)
    )

if numel(out_dims)!=(runner_batch*100):
    raise RuntimeError(
        "runner output must contain batch*100 logits: batch=%d dims=%r numel=%d"%
        (runner_batch,out_dims,numel(out_dims))
    )

if in_fix!=5:
    raise RuntimeError("runner input fix_point mismatch: expected 5 got %d"%in_fix)
if out_fix!=2:
    raise RuntimeError("runner output fix_point mismatch: expected 2 got %d"%out_fix)

print("RUNNER_BATCH=%d"%runner_batch)
print("RUNNER_INPUT_LAYOUT=%s"%input_layout)
print("RUNNER_TENSOR_CONTRACT_GATE=PASS")

inp=np.empty(in_dims,dtype=np.int8,order="C")
out=np.empty(out_dims,dtype=np.int8,order="C")
sample_bytes=224*224*3

err=open(preproc_stderr,"wb")
p=subprocess.Popen(
    [preproc_python,preproc_script,archive,work,str(in_fix),preproc_meta],
    stdout=subprocess.PIPE,stderr=err,bufsize=sample_bytes*2
)

def read_exact(n):
    chunks=[]; left=n
    while left:
        b=p.stdout.read(left)
        if not b:
            raise RuntimeError("preprocessor stream ended early; left=%d"%left)
        chunks.append(b); left-=len(b)
    return b"".join(chunks)

top1=top5=0
errors=[]
t0=time.time()
rc=None
processed=0

try:
    while processed < 10000:
        valid=min(runner_batch,10000-processed)
        labels=[]

        # Zero padding is only for a possible final partial runtime batch.
        # Padded slots are never scored. For CIFAR-100 and B=4, 10000 is exact.
        inp.fill(0)

        for b in range(valid):
            label=struct.unpack("<B",read_exact(1))[0]
            raw=read_exact(sample_bytes)
            q_hwc=np.frombuffer(raw,dtype=np.int8).reshape(224,224,3)

            if input_layout=="NHWC":
                inp[b,...]=q_hwc
            elif input_layout=="NCHW":
                inp[b,...]=q_hwc.transpose(2,0,1)
            else:
                raise RuntimeError("internal unsupported input layout: %s"%input_layout)

            labels.append(label)

        jid=runner.execute_async([inp],[out])
        runner.wait(jid)

        logits_batch=out.reshape(runner_batch,-1)
        if logits_batch.shape!=(runner_batch,100):
            raise RuntimeError(
                "flattened output shape mismatch: %r"%(logits_batch.shape,)
            )

        for b,label in enumerate(labels):
            logits=logits_batch[b].astype(np.int16,copy=False)
            p1=int(np.argmax(logits))
            p5=np.argsort(-logits)[:5]

            ok1=int(p1==label)
            ok5=int(label in p5)
            top1+=ok1
            top5+=ok5

            image_index=processed+b
            if not ok1:
                errors.append((image_index,label,p1,ok5))

        processed += valid
        if processed % 500 == 0 or processed == 10000:
            print(
                "ACC_PROGRESS=%d/10000 TOP1=%.6f TOP5=%.6f RUNNER_BATCH=%d"%
                (processed,top1/float(processed),top5/float(processed),runner_batch),
                flush=True,
            )

    extra=p.stdout.read(1)
    if extra:
        raise RuntimeError("preprocessor emitted trailing bytes")

    rc=p.wait()
finally:
    try:
        p.stdout.close()
    except Exception:
        pass
    err.close()

if rc!=0:
    raise RuntimeError("preprocessor rc=%s; see %s"%(rc,preproc_stderr))

elapsed=time.time()-t0
with open(preproc_meta) as f:
    pp=json.load(f)
res={
  "samples":10000,
  "top1_correct":int(top1),
  "top5_correct":int(top5),
  "top1":top1/10000.0,
  "top5":top5/10000.0,
  "elapsed_seconds_including_streamed_preprocess":elapsed,
  "images_per_second_including_streamed_preprocess":10000.0/elapsed,
  "input_dims":list(in_dims),
  "output_dims":list(out_dims),
  "runner_batch":runner_batch,
  "input_layout":input_layout,
  "input_numel":numel(in_dims),
  "output_numel":numel(out_dims),
  "input_fix_point":in_fix,
  "output_fix_point":out_fix,
  "preprocessing_environment":pp,
}
with open(out_json,"w") as f: json.dump(res,f,indent=2,sort_keys=True)
with open(out_csv,"w") as f:
    f.write("index,label,pred_top1,top5_correct\n")
    for r in errors: f.write("%d,%d,%d,%d\n"%r)

print("HW_CIFAR100_SAMPLES=10000")
print("HW_TOP1=%.8f"%res["top1"])
print("HW_TOP5=%.8f"%res["top5"])
print("HW_RUNNER_INPUT_DIMS=%r"%(in_dims,))
print("HW_RUNNER_OUTPUT_DIMS=%r"%(out_dims,))
print("HW_RUNNER_BATCH=%d"%runner_batch)
print("HW_RUNNER_INPUT_LAYOUT=%s"%input_layout)
print("PREPROCESS_PYTHON=%s"%pp["python"])
print("PREPROCESS_PILLOW=%s"%pp["pillow"])
print("PREPROCESS_TORCHVISION=%s"%pp["torchvision"])
print("HW_ACCURACY_GATE=PASS")
PY

/usr/bin/python3 "$ACC_PY" \
  "$XMODEL" "$DATASET_ARCHIVE" "$OUT/raw/cifar100_extract" \
  "$FP_ENV_FILE" "$XCLBIN" "$DEVICE_INDEX" \
  "$PREPROC_PY" "$ACC_PRE_PY" \
  "$OUT/tables/preprocessing_environment.json" \
  "$OUT/raw/preprocess_stream.stderr" \
  "$OUT/tables/hardware_accuracy.json" \
  "$OUT/tables/hardware_accuracy_errors.csv" \
  2>&1 | tee "$OUT/raw/hardware_accuracy.log"

grep -q '^HW_ACCURACY_GATE=PASS$' "$OUT/raw/hardware_accuracy.log" ||
  inner_fail "FULL_HARDWARE_ACCURACY_FAILED" 107

# ---- DPU latency distribution ----
LAT_PY="$OUT/raw/latency.py"
cat > "$LAT_PY" <<'PY'
from __future__ import print_function
import csv
import json
import os
import statistics
import sys
import time

xmodel, fp_env_file, xclbin, device_index = sys.argv[1:5]
warmup = int(sys.argv[5])
iters = int(sys.argv[6])
out_json = sys.argv[7]
out_csv = sys.argv[8]

with open(fp_env_file) as f:
    for line in f:
        line=line.rstrip("\n")
        if not line: continue
        name,value=line.split("\t",1)
        os.environ[name]=value
os.environ["XLNX_VART_FIRMWARE"] = xclbin
os.environ["XLNX_ENABLE_DEVICES"] = device_index

import numpy as np
import xir
import vart

g = xir.Graph.deserialize(xmodel)
root = g.get_root_subgraph()
dpu = [
    s for s in root.toposort_child_subgraph()
    if s.has_attr("device") and str(s.get_attr("device")).upper() == "DPU"
]
if len(dpu) != 1:
    raise RuntimeError("expected one DPU subgraph")

r = vart.Runner.create_runner(dpu[0], "run")
inputs = r.get_input_tensors()
outputs = r.get_output_tensors()

if len(inputs) != 1 or len(outputs) != 1:
    raise RuntimeError(
        "latency runner tensor count mismatch inputs=%d outputs=%d"%
        (len(inputs),len(outputs))
    )

it = inputs[0]
ot = outputs[0]
lat_in_dims = tuple(int(x) for x in it.dims)
lat_out_dims = tuple(int(x) for x in ot.dims)

if len(lat_in_dims) != 4 or int(lat_in_dims[0]) <= 0:
    raise RuntimeError("invalid latency input dims %r"%(lat_in_dims,))

runner_batch = int(lat_in_dims[0])

lat_out_numel = 1
for d in lat_out_dims:
    lat_out_numel *= int(d)

if not lat_out_dims or int(lat_out_dims[0]) != runner_batch:
    raise RuntimeError(
        "latency output batch mismatch input batch=%d output=%r"%
        (runner_batch,lat_out_dims)
    )

if lat_out_numel != runner_batch*100:
    raise RuntimeError(
        "latency output elements mismatch batch=%d output=%r numel=%d"%
        (runner_batch,lat_out_dims,lat_out_numel)
    )

print("LAT_RUNNER_INPUT_DIMS=%r"%(lat_in_dims,))
print("LAT_RUNNER_OUTPUT_DIMS=%r"%(lat_out_dims,))
print("LAT_RUNNER_BATCH=%d"%runner_batch)

inp = np.zeros(lat_in_dims, dtype=np.int8, order="C")
out = np.empty(lat_out_dims, dtype=np.int8, order="C")

for _ in range(warmup):
    jid = r.execute_async([inp],[out])
    r.wait(jid)

vals = []
for _ in range(iters):
    t0 = time.perf_counter()
    jid = r.execute_async([inp],[out])
    r.wait(jid)
    vals.append((time.perf_counter()-t0)*1000.0)

sv = sorted(vals)
def pct(p):
    pos = (len(sv)-1)*p
    lo = int(pos)
    hi = min(lo+1,len(sv)-1)
    frac = pos-lo
    return sv[lo]*(1-frac)+sv[hi]*frac

batch_mean = statistics.mean(vals)
batch_median = statistics.median(vals)

res = {
    "runner_batch": runner_batch,
    "input_dims": list(lat_in_dims),
    "output_dims": list(lat_out_dims),
    "warmup_iterations": warmup,
    "measured_iterations": iters,
    "mean_ms": batch_mean,
    "std_ms": statistics.stdev(vals) if len(vals)>1 else 0.0,
    "median_ms": batch_median,
    "p95_ms": pct(0.95),
    "p99_ms": pct(0.99),
    "min_ms": min(vals),
    "max_ms": max(vals),
    "effective_mean_ms_per_image_at_full_batch": batch_mean/float(runner_batch),
    "effective_median_ms_per_image_at_full_batch": batch_median/float(runner_batch),
}
with open(out_json,"w") as f:
    json.dump(res,f,indent=2,sort_keys=True)
with open(out_csv,"w") as f:
    w=csv.writer(f)
    w.writerow(["iteration","latency_ms"])
    for i,v in enumerate(vals):
        w.writerow([i,"%.9f"%v])

print("LAT_RUNNER_BATCH=%d" % runner_batch)
print("LAT_BATCH_MEAN_MS=%.9f" % res["mean_ms"])
print("LAT_BATCH_P95_MS=%.9f" % res["p95_ms"])
print("LAT_EFFECTIVE_MEAN_MS_PER_IMAGE_AT_FULL_BATCH=%.9f" %
      res["effective_mean_ms_per_image_at_full_batch"])
print("LATENCY_GATE=PASS")
PY

/usr/bin/python3 "$LAT_PY" \
  "$XMODEL" "$FP_ENV_FILE" "$XCLBIN" "$DEVICE_INDEX" \
  "$LAT_WARMUP" "$LAT_ITERS" \
  "$OUT/tables/latency_summary.json" \
  "$OUT/tables/latency_samples.csv" \
  2>&1 | tee "$OUT/raw/latency.log"

grep -q '^LATENCY_GATE=PASS$' "$OUT/raw/latency.log" ||
  inner_fail "LATENCY_TEST_FAILED" 106

# ---- Repeated throughput ----
BENCH_PY="$OUT/raw/benchmark_repeated.py"
cat > "$BENCH_PY" <<'PY'
from __future__ import print_function
import csv
import os
import re
import subprocess
import sys

xmodel, fp_env_file, xclbin, device_index, threads = sys.argv[1:6]
repeats = int(sys.argv[6])
timeout = int(sys.argv[7])
out = sys.argv[8]
marker = sys.argv[9]

env = os.environ.copy()
with open(fp_env_file) as f:
    for line in f:
        line=line.rstrip("\n")
        if not line: continue
        name,value=line.split("\t",1)
        env[name]=value
env["XLNX_VART_FIRMWARE"] = xclbin
env["XLNX_ENABLE_DEVICES"] = device_index
rows=[]

for run in range(1,repeats+1):
    print("BENCHMARK_RUN_BEGIN=%d" % run, flush=True)
    open(marker,"w").write("run=%d\n"%run)
    try:
        p=subprocess.run(
            ["/usr/bin/python3","-m","xdputil","benchmark",xmodel,threads],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            universal_newlines=True,timeout=timeout
        )
        text=p.stdout or ""
    finally:
        try: os.remove(marker)
        except OSError: pass

    open(os.path.join(out,"benchmark_run_%02d.txt"%run),"w").write(text)
    print(text,end="" if text.endswith("\n") else "\n")

    low=text.lower()
    if p.returncode != 0: raise SystemExit(70)
    if "exception in thread" in low: raise SystemExit(71)
    if "failed to create xrm context" in low: raise SystemExit(72)
    if "target_factory_invalid_type" in low: raise SystemExit(73)

    m=re.search(
        r'FPS\s*=\s*([0-9]+(?:\.[0-9]+)?)'
        r'.*?number_of_frames\s*=\s*([0-9]+)'
        r'.*?time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*seconds',
        text,re.I|re.S
    )
    if not m: raise SystemExit(74)

    fps=float(m.group(1)); frames=int(m.group(2)); seconds=float(m.group(3))
    rows.append((run,int(threads),fps,frames,seconds))
    print("BENCHMARK_RESULT run=%d FPS=%.6f frames=%d seconds=%.6f" %
          (run,fps,frames,seconds),flush=True)

csv_path=os.path.join(out,"..","tables","benchmark_runs.csv")
with open(csv_path,"w") as f:
    w=csv.writer(f)
    w.writerow(["run","software_threads","fps","frames","seconds"])
    for row in rows: w.writerow(row)

print("BENCHMARK_REPEATED_GATE=PASS")
PY

/usr/bin/python3 "$BENCH_PY" \
  "$XMODEL" "$FP_ENV_FILE" "$XCLBIN" "$DEVICE_INDEX" \
  "$BENCH_THREADS" "$PERF_REPEATS" "$BENCH_TIMEOUT" \
  "$OUT/raw" "$OUT/.benchmark_active" \
  2>&1 | tee "$OUT/raw/benchmark_repeated.log"

grep -q '^BENCHMARK_REPEATED_GATE=PASS$' "$OUT/raw/benchmark_repeated.log" ||
  inner_fail "REPEATED_BENCHMARK_FAILED" 107

"$XRMADM_BIN" "$LIST_JSON" > "$OUT/raw/xrm_list_post_runtime.txt" 2>&1 ||
  inner_fail "XRMADM_POST_RUNTIME_LIST_FAILED" 108
POST_XRM_EXPECTED_MATCHES=0
while IFS=$'\t' read -r FP_KEY FP_VALUE; do
  [[ -n "$FP_KEY" ]] || continue

  CU_FULL="${FP_KEY%.fingerprint}"
  CU_INSTANCE="${CU_FULL#*:}"

  if grep -Fq "$CU_INSTANCE" "$OUT/raw/xrm_list_post_runtime.txt"; then
    echo "POST_XRM_EXPECTED_CU_PRESENT=$CU_INSTANCE"
    POST_XRM_EXPECTED_MATCHES=$((POST_XRM_EXPECTED_MATCHES + 1))
  else
    inner_fail "POST_XRM_MISSING_EXPECTED_CU_${CU_INSTANCE}" 109
  fi
done < "$FP_ENV_FILE"

[[ "$POST_XRM_EXPECTED_MATCHES" == "$CU_COUNT_EXPECTED" ]] ||
  inner_fail "POST_XRM_EXPECTED_MATCH_COUNT_${POST_XRM_EXPECTED_MATCHES}_EXPECTED_${CU_COUNT_EXPECTED}" 110

echo "POST_XRM_EXPECTED_CUS=$POST_XRM_EXPECTED_MATCHES"

/opt/xilinx/xrt/bin/xbutil examine --device "$BDF" \
  > "$OUT/raw/container_card_post_runtime.txt" 2>&1 ||
  inner_fail "CONTAINER_CARD_POST_RUNTIME_FAILED" 110
POST_CONTAINER_CU_COUNT="$(
  grep -oE 'DPUCADF8H:DPUCADF8H_[0-9]+' "$OUT/raw/container_card_post_runtime.txt" |
    sort -u | wc -l | tr -d ' '
)"
[[ "$POST_CONTAINER_CU_COUNT" == "$CU_COUNT_EXPECTED" ]] ||
  inner_fail "CONTAINER_POST_RUNTIME_CU_COUNT_EXPECTED_${CU_COUNT_EXPECTED}_GOT_${POST_CONTAINER_CU_COUNT}" 111

echo "CONTAINER_IEEE_SUITE=PASS"
SH
chmod +x "$INNER"

C_INNER="/workspace/${INNER#$ROOT/}"

set +e
run_udocker run \
  "${VOL_ARGS[@]}" \
  --env="HOME=/tmp" \
  --workdir=/workspace \
  "$UDOCKER_CONTAINER" \
  /bin/bash --noprofile --norc \
  "$C_INNER" \
  "$BDF" "$XRT_INDEX" "$C_XCLBIN" "$C_XMODEL" "$C_DATASET" \
  "$EXPECTED_CIFAR100_MD5" "$CU_MODE" "/workspace/${FP_ENV_FILE#$ROOT/}" \
  "$PERF_REPEATS" "$BENCH_THREADS" "$BENCH_TIMEOUT" \
  "$LAT_WARMUP" "$LAT_ITERS" "$C_OUT" \
  > "$RAW/container_ieee_suite.log" 2>&1
CONTAINER_RC=$?
set -e

cat "$RAW/container_ieee_suite.log"
[[ "$CONTAINER_RC" -eq 0 ]] || fail "CONTAINER_IEEE_SUITE_FAILED_RC_${CONTAINER_RC}"
grep -qx 'CONTAINER_IEEE_SUITE=PASS' "$RAW/container_ieee_suite.log" ||
  fail "CONTAINER_IEEE_SUITE_PASS_MARKER_MISSING"

touch "$SENSOR_STOP"
wait "$SENSOR_PID" 2>/dev/null || true
SENSOR_PID=""

# -----------------------------------------------------------------------------
# 7. Post-runtime card state
# -----------------------------------------------------------------------------
echo "===== 7. FINAL CARD STATE ====="

"$XBUTIL" examine --device "$BDF" --report all \
  > "$RAW/xbutil_examine_post_all.txt" 2>&1 || true

"$XBUTIL" examine --device "$BDF" \
  > "$RAW/xbutil_post_runtime.txt" 2>&1 ||
  fail "HOST_CARD_POST_RUNTIME_EXAMINE_FAILED"

POST_HOST_CU_COUNT="$(
  grep -oE 'DPUCADF8H:DPUCADF8H_[0-9]+' "$RAW/xbutil_post_runtime.txt" |
    sort -u | wc -l | tr -d ' '
)"
[[ "$POST_HOST_CU_COUNT" == "$CU_MODE" ]] ||
  fail "HOST_POST_RUNTIME_CU_COUNT_EXPECTED_${CU_MODE}_GOT_${POST_HOST_CU_COUNT}"
READY_COUNT="$(
  grep 'DPUCADF8H:DPUCADF8H_' "$RAW/xbutil_post_runtime.txt" |
    grep -c '(READY)' || true
)"
[[ "$READY_COUNT" == "$CU_MODE" ]] ||
  fail "HOST_POST_RUNTIME_READY_COUNT_EXPECTED_${CU_MODE}_GOT_${READY_COUNT}"

echo "FINAL_CARD_GATE=PASS"

# -----------------------------------------------------------------------------
# 8. IEEE statistics + tables
# -----------------------------------------------------------------------------
echo "===== 8. IEEE RESULT AGGREGATION ====="

# Host shell may carry Vitis/Conda LD_LIBRARY_PATH/PYTHONHOME/PYTHONPATH.
# The runtime suite is already complete at this point.  Run post-processing
# with an isolated system-Python environment so /usr/bin/python3 cannot load
# an incompatible libpython (e.g. _Py_LegacyLocaleDetected failure).
clean_host_python() {
  env -i     HOME="${HOME:-/home/truongnh}"     USER="${USER:-truongnh}"     LOGNAME="${LOGNAME:-${USER:-truongnh}}"     PATH=/usr/bin:/bin     PYTHONNOUSERSITE=1     /usr/bin/python3 "$@"
}

clean_host_python - <<'PY' | tee "$RAW/clean_host_python_gate.txt"
from __future__ import print_function
import os
import sys
print("CLEAN_HOST_PYTHON=%s" % ("%d.%d.%d" % sys.version_info[:3]))
print("CLEAN_HOST_EXECUTABLE=%s" % sys.executable)
print("CLEAN_HOST_LD_LIBRARY_PATH=%s" % os.environ.get("LD_LIBRARY_PATH", "UNSET"))
print("CLEAN_HOST_PYTHONHOME=%s" % os.environ.get("PYTHONHOME", "UNSET"))
print("CLEAN_HOST_PYTHONPATH=%s" % os.environ.get("PYTHONPATH", "UNSET"))
print("CLEAN_HOST_PYTHON_GATE=PASS")
PY

grep -qx 'CLEAN_HOST_PYTHON_GATE=PASS' "$RAW/clean_host_python_gate.txt" ||
  fail "CLEAN_HOST_PYTHON_GATE_FAILED"

SUMMARY_PY="$OUT/make_ieee_summary.py"
cat > "$SUMMARY_PY" <<'PY'
from __future__ import print_function
import csv
import json
import math
import os
import re
import statistics
import sys

out, xclbin_sha, xmodel_sha, uuid, bdf, xrt_index, dataset_md5, vai_commit = sys.argv[1:9]
validate_rc=int(sys.argv[9])
hardware_cus=int(sys.argv[10])
software_threads=int(sys.argv[11])
tables=os.path.join(out,"tables")
raw=os.path.join(out,"raw")

def load_json(p):
    with open(p) as f: return json.load(f)

acc=load_json(os.path.join(tables,"hardware_accuracy.json"))
lat=load_json(os.path.join(tables,"latency_summary.json"))

bench=[]
with open(os.path.join(tables,"benchmark_runs.csv")) as f:
    for r in csv.DictReader(f):
        bench.append({
            "run":int(r["run"]),
            "threads":int(r["software_threads"]),
            "fps":float(r["fps"]),
            "frames":int(r["frames"]),
            "seconds":float(r["seconds"])
        })

fps=[r["fps"] for r in bench]
n=len(fps)
mean_fps=statistics.mean(fps)
sd_fps=statistics.stdev(fps) if n>1 else 0.0
cv_pct=(sd_fps/mean_fps*100.0) if mean_fps else 0.0
t95={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}
tc=t95.get(n,1.96 if n>10 else None)
ci95=(tc*sd_fps/math.sqrt(n)) if tc is not None and n>1 else None

powers=[]
temps=[]
sensor=os.path.join(raw,"sensors_runtime.txt")
active=False
if os.path.isfile(sensor):
    for line in open(sensor,errors="replace"):
        if line.startswith("SAMPLE_BEGIN"):
            active="benchmark_active=1" in line
            continue
        if not active: continue

        m=re.search(
            r'(?i)(?:total\s+)?power[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(?:W|Watts)\b',
            line
        )
        if m: powers.append(float(m.group(1)))

        m=re.search(
            r'(?i)(?:temp(?:erature)?|thermal)[^0-9-]*(-?[0-9]+(?:\.[0-9]+)?)\s*(?:C|Celsius)\b',
            line
        )
        if m: temps.append(float(m.group(1)))

result={
  "hardware":{
    "board":"Alveo U250",
    "hardware_cu_count":hardware_cus,
    "cu_family":"DPUCADF8H",
    "bdf":bdf,
    "xrt_device_index":int(xrt_index),
    "xclbin_uuid":uuid,
    "external_data_clock_mhz":300,
    "external_kernel_clock_mhz":300,
    "dpu_internal_clock_mhz":500
  },
  "artifacts":{
    "xclbin_sha256":xclbin_sha,
    "xmodel_sha256":xmodel_sha,
    "xmodel_fingerprint":"0x0700000000000000",
    "vitis_ai_commit":vai_commit,
    "cifar100_archive_md5":dataset_md5
  },
  "accuracy_hardware":acc,
  "latency_dpu_runner":lat,
  "throughput":{
    "software_threads":software_threads,
    "n":n,
    "runs":bench,
    "fps_mean":mean_fps,
    "fps_sd":sd_fps,
    "fps_min":min(fps),
    "fps_max":max(fps),
    "fps_cv_percent":cv_pct,
    "fps_95ci_half_width":ci95
  },
  "runtime_sensors":{
    "active_power_samples":len(powers),
    "board_power_w_mean":statistics.mean(powers) if powers else None,
    "board_power_w_sd":statistics.stdev(powers) if len(powers)>1 else (0.0 if powers else None),
    "board_power_w_min":min(powers) if powers else None,
    "board_power_w_max":max(powers) if powers else None,
    "active_temperature_samples":len(temps),
    "temperature_c_max":max(temps) if temps else None
  },
  "card_validation":{
    "xbutil_validate_quick_rc":validate_rc,
    "note":"Recorded for audit; not a hard gate because validation binaries vary by installation."
  }
}

with open(os.path.join(tables,"ieee_results.json"),"w") as f:
    json.dump(result,f,indent=2,sort_keys=True)

rows=[
  ("Hardware CU count",hardware_cus,"CU","exact card enumeration"),
  ("DPU internal clock",500,"MHz","certified design"),
  ("External kernel clock",300,"MHz","xbutil"),
  ("CIFAR-100 HW samples",acc["samples"],"images","official test set"),
  ("HW Top-1",acc["top1"]*100.0,"%","10,000-image hardware inference"),
  ("HW Top-5",acc["top5"]*100.0,"%","10,000-image hardware inference"),
  ("DPU runner batch",lat["runner_batch"],"images/call","VART runtime tensor contract"),
  ("DPU batch latency mean",lat["mean_ms"],"ms","execute_async+wait at full runner batch"),
  ("DPU batch latency median",lat["median_ms"],"ms","full runner batch"),
  ("DPU batch latency p95",lat["p95_ms"],"ms","full runner batch"),
  ("DPU batch latency p99",lat["p99_ms"],"ms","full runner batch"),
  ("Effective mean time/image",lat["effective_mean_ms_per_image_at_full_batch"],"ms/image","batch latency / runner batch"),
  ("Throughput mean",mean_fps,"FPS","repeated xdputil benchmark"),
  ("Throughput SD",sd_fps,"FPS","sample standard deviation"),
  ("Throughput min",min(fps),"FPS","repeated runs"),
  ("Throughput max",max(fps),"FPS","repeated runs"),
  ("Throughput CV",cv_pct,"%","repeated runs")
]
if ci95 is not None:
    rows.append(("Throughput 95% CI half-width",ci95,"FPS","Student-t"))
if powers:
    rows.append(("Board power mean",statistics.mean(powers),"W","benchmark-active XRT samples"))
    rows.append(("Board power max",max(powers),"W","benchmark-active XRT samples"))
if temps:
    rows.append(("Max temperature",max(temps),"C","benchmark-active XRT samples"))

with open(os.path.join(tables,"ieee_results_summary.csv"),"w") as f:
    w=csv.writer(f)
    w.writerow(["metric","value","unit","evidence"])
    for row in rows: w.writerow(row)

md=[]
md.append("# IEEE %d-CU Validation Summary"%hardware_cus)
md.append("")
md.append("## Reproducibility identity")
md.append("")
md.append("| Item | Value |")
md.append("|---|---|")
md.append("| Board | Alveo U250 |")
md.append("| Hardware CU count | %d |"%hardware_cus)
md.append("| CU family | `DPUCADF8H` |")
md.append("| BDF / XRT index | `%s` / `%s` |"%(bdf,xrt_index))
md.append("| XCLBIN UUID | `%s` |"%uuid)
md.append("| XCLBIN SHA256 | `%s` |"%xclbin_sha)
md.append("| XMODEL SHA256 | `%s` |"%xmodel_sha)
md.append("| XMODEL fingerprint | `0x0700000000000000` |")
md.append("| Vitis-AI commit | `%s` |"%vai_commit)
md.append("| CIFAR-100 MD5 | `%s` |"%dataset_md5)
md.append("| DPU internal clock | 500 MHz |")
md.append("| External Data / Kernel clock | 300 / 300 MHz |")
md.append("")
md.append("## Hardware accuracy")
md.append("")
md.append("| Metric | Result |")
md.append("|---|---:|")
md.append("| Test images | %d |"%acc["samples"])
md.append("| Top-1 | %.4f %% |"%(acc["top1"]*100.0))
md.append("| Top-5 | %.4f %% |"%(acc["top5"]*100.0))
md.append("")
pp=acc.get("preprocessing_environment",{})
md.append("Preprocessing environment actually used: Python `%s`, Torch `%s`, Torchvision `%s`, Pillow `%s`, NumPy `%s`."%(
    pp.get("python","N/A"),pp.get("torch","N/A"),pp.get("torchvision","N/A"),
    pp.get("pillow","N/A"),pp.get("numpy","N/A")
))
md.append("")
md.append("## DPU runner latency (runtime batch=%d)"%lat["runner_batch"])
md.append("")
md.append("| Metric | ms |")
md.append("|---|---:|")
for k,l in [("mean_ms","Batch mean"),("std_ms","Batch SD"),("median_ms","Batch median"),
            ("p95_ms","Batch P95"),("p99_ms","Batch P99"),
            ("min_ms","Batch min"),("max_ms","Batch max")]:
    md.append("| %s | %.6f |"%(l,lat[k]))
md.append("| Effective mean/image at full batch | %.6f |"%
          lat["effective_mean_ms_per_image_at_full_batch"])
md.append("")
md.append(
    "Latency is measured per VART runner call with runtime batch %d. "
    "The effective per-image value is batch latency divided by batch size; "
    "it is not reported as single-request latency."%lat["runner_batch"]
)
md.append("")
md.append("## Throughput — %d hardware CU(s), %d software thread(s)"%(hardware_cus,software_threads))
md.append("")
md.append("| Run | FPS | Frames | Seconds |")
md.append("|---:|---:|---:|---:|")
for r in bench:
    md.append("| %d | %.6f | %d | %.6f |"%(r["run"],r["fps"],r["frames"],r["seconds"]))
md.append("")
md.append("Mean FPS = **%.6f**, SD = **%.6f**, CV = **%.3f%%**."%(mean_fps,sd_fps,cv_pct))
if ci95 is not None:
    md.append("95%% CI = **%.6f ± %.6f FPS**."%(mean_fps,ci95))
md.append("")
md.append("## Runtime sensors")
md.append("")
if powers:
    md.append("Board power: mean **%.3f W**, max **%.3f W**."%(statistics.mean(powers),max(powers)))
else:
    md.append("Board power was not parsable from the card/XRT sensor text; raw sensor evidence is preserved and no power number is fabricated.")
if temps:
    md.append("Maximum parsed temperature during benchmark: **%.3f C**."%max(temps))
else:
    md.append("Temperature was not parsable from the card/XRT sensor text; raw sensor evidence is preserved.")
md.append("")
md.append("## Scope")
md.append("")
md.append(
    "Top-1/Top-5 are measured on the deployed xmodel using all 10,000 official "
    "CIFAR-100 test images. The preprocessing environment and exact "
    "torchvision transform are recorded from the installed Vitis-AI PyTorch "
    "environment. Throughput is measured with xdputil benchmark using the "
    "reported hardware CU count and software-thread count. Latency is measured "
    "for the actual VART runtime batch exposed by the runner. FPGA "
    "LUT/FF/BRAM/URAM/DSP and timing values are taken only from copied Vivado "
    "implementation reports."
)

md_name="IEEE_%dCU_RESULTS.md"%hardware_cus
with open(os.path.join(out,md_name),"w",encoding="utf-8") as f:
    f.write("\n".join(md)+"\n")
print("IEEE_MD_FILE=%s"%md_name)

print("THROUGHPUT_MEAN_FPS=%.9f"%mean_fps)
print("THROUGHPUT_SD_FPS=%.9f"%sd_fps)
print("HW_TOP1_PERCENT=%.6f"%(acc["top1"]*100.0))
print("HW_TOP5_PERCENT=%.6f"%(acc["top5"]*100.0))
print("RUNNER_BATCH=%d"%lat["runner_batch"])
print("BATCH_LATENCY_MEAN_MS=%.9f"%lat["mean_ms"])
print("BATCH_LATENCY_P95_MS=%.9f"%lat["p95_ms"])
print("EFFECTIVE_MEAN_MS_PER_IMAGE_AT_FULL_BATCH=%.9f"%
      lat["effective_mean_ms_per_image_at_full_batch"])
print("IEEE_SUMMARY_GATE=PASS")
PY

clean_host_python "$SUMMARY_PY" \
  "$OUT" "$XCLBIN_SHA" "$XMODEL_SHA" "$POST_UUID" "$BDF" "$XRT_INDEX" \
  "$DATASET_MD5" "$VAI_COMMIT" "$XB_VALIDATE_RC" "$CU_MODE" "$BENCH_THREADS" \
  2>&1 | tee "$RAW/ieee_summary.log"

grep -q '^IEEE_SUMMARY_GATE=PASS$' "$RAW/ieee_summary.log" ||
  fail "IEEE_SUMMARY_AGGREGATION_FAILED"

# -----------------------------------------------------------------------------
# 9. Manifest + final gates
# -----------------------------------------------------------------------------
echo "===== 9. EVIDENCE MANIFEST ====="

(
  cd "$OUT"
  find . -type f \
    ! -name 'MANIFEST_SHA256.txt' \
    ! -name 'FINAL_STATUS.txt' \
    -print0 |
    sort -z |
    xargs -0 sha256sum
) > "$OUT/MANIFEST_SHA256.txt"

[[ -s "$TABLES/hardware_accuracy.json" ]] || fail "HARDWARE_ACCURACY_JSON_MISSING"
[[ -s "$TABLES/latency_summary.json" ]] || fail "LATENCY_SUMMARY_MISSING"
[[ -s "$TABLES/benchmark_runs.csv" ]] || fail "BENCHMARK_CSV_MISSING"
[[ -s "$TABLES/ieee_results_summary.csv" ]] || fail "IEEE_CSV_MISSING"
IEEE_RESULT_MD="$OUT/IEEE_${CU_MODE}CU_RESULTS.md"
[[ -s "$IEEE_RESULT_MD" ]] || fail "IEEE_MD_MISSING"

ACC_SAMPLES="$(
  clean_host_python - "$TABLES/hardware_accuracy.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["samples"])
PY
)"
[[ "$ACC_SAMPLES" == "10000" ]] || fail "HARDWARE_ACCURACY_NOT_FULL_10000"

RUNNER_BATCH="$(
  clean_host_python - "$TABLES/latency_summary.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["runner_batch"])
PY
)"

BENCH_ROWS="$(
  awk 'NR>1 && NF{n++} END{print n+0}' "$TABLES/benchmark_runs.csv"
)"
[[ "$BENCH_ROWS" == "$PERF_REPEATS" ]] ||
  fail "BENCHMARK_REPEAT_COUNT_MISMATCH_EXPECTED_${PERF_REPEATS}_GOT_${BENCH_ROWS}"

{
  echo "RESULT=PASS"
  echo "IEEE_FULL_VALIDATION=PASS"
  echo "HARDWARE_CU_COUNT=$CU_MODE"
  echo "PHYSICAL_CU_COUNT_SOURCE=xbutil_after_exact_program"
  echo "XRM_GATE=all_expected_physical_CUs_present"
  echo "BDF=$BDF"
  echo "XRT_DEVICE_INDEX=$XRT_INDEX"
  echo "XCLBIN_UUID=$POST_UUID"
  echo "XCLBIN_SHA256=$XCLBIN_SHA"
  echo "XMODEL_SHA256=$XMODEL_SHA"
  echo "MODEL_FINGERPRINT=$MODEL_FP_HEX"
  echo "CIFAR100_MD5=$DATASET_MD5"
  echo "CIFAR100_HW_TEST_IMAGES=10000"
  echo "SOFTWARE_THREADS=$BENCH_THREADS"
  echo "PERF_REPEATS=$PERF_REPEATS"
  echo "RUNNER_BATCH=$RUNNER_BATCH"
  echo "LATENCY_WARMUP=$LAT_WARMUP"
  echo "LATENCY_MEASURED=$LAT_ITERS"
  echo "IEEE_SUMMARY=$IEEE_RESULT_MD"
  echo "IEEE_CSV=$TABLES/ieee_results_summary.csv"
  echo "IEEE_JSON=$TABLES/ieee_results.json"
  echo "MANIFEST=$OUT/MANIFEST_SHA256.txt"
  echo "OUT=$OUT"
  echo "LOG=$LOG"
} | tee "$STATUS"

echo "============================================================================="
cat "$STATUS"
cat "$RAW/ieee_summary.log"
echo "============================================================================="
