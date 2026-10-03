#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

ROOT="${ROOT:?Set ROOT to the U250 study workspace}"
EXCLUSIVE_DEVICE="${EXCLUSIVE_DEVICE:-0}"
BDF="${BDF:-0000:1b:00.0}"
THREAD_LIST="${THREAD_LIST:-1 2 3 4}"
REPEATS="${REPEATS:-3}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"

XCLBIN="${XCLBIN:-$ROOT/09_results/crossslr_golden_f500_v6_1_20260922_052031/1cu/1cu_DPUCADF8H_300_to_500_CROSSSLR_TIMING_CLOSED.xclbin}"
XMODEL="${XMODEL:-$ROOT/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel}"
EXPECTED_XCLBIN_SHA="${EXPECTED_XCLBIN_SHA:-156afe3e437241fa332ff817b5450c3069c460cf478e189c7eb01f12b0901928}"
EXPECTED_XMODEL_SHA="${EXPECTED_XMODEL_SHA:-da6b56491525b3ccdf46f4aff5ba63ac4d8545ebf809d30fd024f7735c581007}"
MODEL_FP_DEC="${MODEL_FP_DEC:-504403158265495552}"

UDOCKER_BASE="${UDOCKER_BASE:-$ROOT/toolchain/udocker-1.3.17}"
UDOCKER_STORE="${UDOCKER_STORE:-$ROOT/toolchain/udocker_store}"
UDOCKER_CONTAINER="${UDOCKER_CONTAINER:-vai20_cpu_7a87fa94_p1fast}"

[[ "$EXCLUSIVE_DEVICE" == "1" ]] || {
  echo "ERROR: set EXCLUSIVE_DEVICE=1 after checking that the U250 is idle" >&2
  exit 2
}
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid REPEATS" >&2; exit 2; }
[[ -f "$XCLBIN" && -f "$XMODEL" ]] || { echo "ERROR: missing artifact" >&2; exit 2; }

actual_xclbin_sha="$(sha256sum "$XCLBIN" | awk '{print $1}')"
actual_xmodel_sha="$(sha256sum "$XMODEL" | awk '{print $1}')"
[[ "$actual_xclbin_sha" == "$EXPECTED_XCLBIN_SHA" ]] || {
  echo "ERROR: XCLBIN SHA-256 mismatch" >&2; exit 3;
}
[[ "$actual_xmodel_sha" == "$EXPECTED_XMODEL_SHA" ]] || {
  echo "ERROR: XModel SHA-256 mismatch" >&2; exit 3;
}

ts="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$ROOT/09_results/xdputil_1cu_thread_sweep_$ts}"
RAW="$OUT/raw"
mkdir -p "$RAW"

exec > >(tee -a "$OUT/campaign.log") 2>&1
echo "CAMPAIGN=xdputil_1cu_thread_sweep"
echo "UTC_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "THREAD_LIST=$THREAD_LIST"
echo "REPEATS=$REPEATS"
echo "XCLBIN_SHA256=$actual_xclbin_sha"
echo "XMODEL_SHA256=$actual_xmodel_sha"

# Fail if another process currently has the target render node open.
render_node="$(basename "$(readlink -f /sys/bus/pci/devices/$BDF/drm/renderD* 2>/dev/null | head -1)" 2>/dev/null || true)"
if [[ -n "$render_node" ]] && fuser "/dev/dri/$render_node" >"$RAW/fuser_pre.txt" 2>&1; then
  cat "$RAW/fuser_pre.txt"
  echo "ERROR: target U250 is in use" >&2
  exit 4
fi

XBUTIL=/opt/xilinx/xrt/bin/xbutil
"$XBUTIL" program --device "$BDF" --user "$XCLBIN" >"$RAW/xbutil_program.txt" 2>&1
"$XBUTIL" examine --device "$BDF" >"$RAW/xbutil_after_program.txt" 2>&1

mapfile -t cu_names < <(
  grep -oE 'DPUCADF8H:DPUCADF8H_[0-9]+' "$RAW/xbutil_after_program.txt" | sort -u
)
[[ "${#cu_names[@]}" -eq 1 ]] || {
  echo "ERROR: expected one physical DPU CU, found ${#cu_names[@]}" >&2
  exit 5
}
printf '%s.fingerprint\t%s\n' "${cu_names[0]}" "$MODEL_FP_DEC" >"$OUT/fingerprint_env.tsv"

UDOCKER_BIN="$UDOCKER_BASE/udocker/maincmd.py"
[[ -f "$UDOCKER_BIN" ]] || { echo "ERROR: udocker entry point missing" >&2; exit 6; }
export UDOCKER_DIR="$UDOCKER_STORE"
export UDOCKER_DEFAULT_EXECUTION_MODE=P2
export PROOT_TMP_DIR="$ROOT/toolchain/proot_tmp"
mkdir -p "$PROOT_TMP_DIR"

container_script="$OUT/container_thread_sweep.sh"
cat >"$container_script" <<'INNER'
#!/usr/bin/env bash
set -Eeuo pipefail
set +u
source /opt/xilinx/xrt/setup.sh
source /opt/xilinx/xrm/setup.sh
set -u

xclbin="$1"
xmodel="$2"
fp_env="$3"
thread_list="$4"
repeats="$5"
timeout_s="$6"
out="$7"

live_xrmd=0
while read -r stat; do
  [[ "$stat" == Z* ]] || live_xrmd=1
done < <(ps -C xrmd -o stat= 2>/dev/null || true)
[[ "$live_xrmd" -eq 0 ]] || { echo "ERROR: live xrmd already present" >&2; exit 20; }
rm -f /dev/shm/xrm.data
xrmd >"$out/raw/xrmd.log" 2>&1 &
xrmd_pid=$!
cleanup() { kill "$xrmd_pid" 2>/dev/null || true; wait "$xrmd_pid" 2>/dev/null || true; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 20); do
  if kill -0 "$xrmd_pid" 2>/dev/null; then ready=1; break; fi
  sleep 0.5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: xrmd did not start" >&2; exit 21; }
sleep 2

cat >"$out/raw/xrm_load.json" <<JSON
{"request":{"name":"load","requestId":2,"parameters":[{"device":0,"xclbin":"$xclbin"}]}}
JSON
xrmadm "$out/raw/xrm_load.json" >"$out/raw/xrm_load.txt" 2>&1
grep -qi '"status"[[:space:]]*:[[:space:]]*"ok"' "$out/raw/xrm_load.txt" || {
  cat "$out/raw/xrm_load.txt"; exit 22;
}

printf 'schedule_index,repeat,software_runners,fps,frames,seconds\n' >"$out/thread_sweep_runs.csv"
schedule_index=0
for repeat in $(seq 1 "$repeats"); do
  for runners in $thread_list; do
    schedule_index=$((schedule_index + 1))
    log="$out/raw/xdputil_t${runners}_r${repeat}.txt"
    echo "RUN_BEGIN index=$schedule_index repeat=$repeat runners=$runners"
    set +e
    timeout "$timeout_s" env \
      "$(cut -f1 "$fp_env")=$(cut -f2 "$fp_env")" \
      XLNX_VART_FIRMWARE="$xclbin" XLNX_ENABLE_DEVICES=0 \
      /usr/bin/python3 -m xdputil benchmark "$xmodel" "$runners" >"$log" 2>&1
    rc=$?
    set -e
    cat "$log"
    [[ "$rc" -eq 0 ]] || { echo "ERROR: benchmark rc=$rc" >&2; exit 23; }
    grep -q 'Test PASS[.]' "$log" || {
      echo "ERROR: benchmark did not report Test PASS" >&2
      exit 24
    }
    if grep -Eqi 'Test Failed|CU timeout|error:' "$log"; then
      echo "ERROR: benchmark log contains a runtime failure" >&2
      exit 24
    fi
    parsed="$(/usr/bin/python3 - "$log" <<'PY'
from __future__ import print_function
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'FPS\s*=\s*([0-9]+(?:\.[0-9]+)?).*?number_of_frames\s*=\s*([0-9]+).*?time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*seconds', text, re.I | re.S)
if not m:
    raise SystemExit(1)
print(','.join(m.groups()))
PY
)"
    printf '%d,%d,%d,%s\n' "$schedule_index" "$repeat" "$runners" "$parsed" >>"$out/thread_sweep_runs.csv"
  done
done

/usr/bin/python3 - "$out/thread_sweep_runs.csv" "$out/thread_sweep_summary.json" <<'PY'
from __future__ import print_function
import csv, json, math, statistics, sys
src, dst = sys.argv[1:]
groups = {}
with open(src) as handle:
    for row in csv.DictReader(handle):
        groups.setdefault(int(row['software_runners']), []).append(float(row['fps']))
tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
summary = {'measurement': 'xdputil benchmark on one physical 500-MHz CU', 'groups': []}
for runners in sorted(groups):
    values = groups[runners]
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    ci = tcrit.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else None
    summary['groups'].append({
        'software_runners': runners, 'n': n, 'fps': values,
        'mean_fps': mean, 'sample_sd_fps': sd, 'ci95_half_width_fps': ci,
    })
with open(dst, 'w') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

sha256sum "$out/thread_sweep_runs.csv" "$out/thread_sweep_summary.json" >"$out/results.sha256"
echo "THREAD_SWEEP=PASS"
INNER
chmod +x "$container_script"

volumes=(
  "--volume=$ROOT:/workspace"
  "--volume=/dev/dri:/dev/dri"
  "--volume=/dev/shm:/dev/shm"
  "--volume=/sys/bus/pci:/sys/bus/pci"
)
[[ -d /sys/class/drm ]] && volumes+=("--volume=/sys/class/drm:/sys/class/drm")
[[ -d /opt/xilinx/dsa ]] && volumes+=("--volume=/opt/xilinx/dsa:/opt/xilinx/dsa")
[[ -d /opt/xilinx/overlaybins ]] && volumes+=("--volume=/opt/xilinx/overlaybins:/opt/xilinx/overlaybins")

c_xclbin="/workspace/${XCLBIN#$ROOT/}"
c_xmodel="/workspace/${XMODEL#$ROOT/}"
c_fp="/workspace/${OUT#$ROOT/}/fingerprint_env.tsv"
c_out="/workspace/${OUT#$ROOT/}"
c_script="/workspace/${OUT#$ROOT/}/container_thread_sweep.sh"

env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  PYTHONNOUSERSITE=1 UDOCKER_DIR="$UDOCKER_DIR" \
  UDOCKER_DEFAULT_EXECUTION_MODE="$UDOCKER_DEFAULT_EXECUTION_MODE" \
  PROOT_TMP_DIR="$PROOT_TMP_DIR" \
  /usr/bin/python3 "$UDOCKER_BIN" run "${volumes[@]}" \
  --env="HOME=/tmp" --workdir=/workspace "$UDOCKER_CONTAINER" \
  /bin/bash --noprofile --norc "$c_script" \
  "$c_xclbin" "$c_xmodel" "$c_fp" "$THREAD_LIST" "$REPEATS" \
  "$BENCH_TIMEOUT" "$c_out"

echo "UTC_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "OUT=$OUT"
