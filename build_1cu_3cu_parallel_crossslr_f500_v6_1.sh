#!/usr/bin/env bash
# ============================================================================
# DPUCADF8H U250 V6.1 — cross-SLR-aware build/certification for 1CU + 3CU @ F500
#
# Goals:
#   - Preserve the existing 1CU xclbin from the proven routed run untouched.
#   - Reuse the exact immutable Golden XO: external 300 MHz -> internal 500 MHz.
#   - Use worker-local Vitis-AI hooks (DPU_NUM/FREQ/BUILD_DIR correct per worker).
#   - Fix the known non-portable make_outputs glob /group/bcapps/... problem.
#   - DO NOT use --connectivity.slr, because the vendor physical floorplan for
#     SLR0 intentionally hard-LOCs some DPU URAM into physical SLR1.
#   - Re-create logical CU->SLR binding directly with pblock_dynamic_SLR{0,2,3}
#     in the worker-local opt_pre hook, after the vendor cross-SLR pblock resize.
#   - Keep DDR mapping explicit: CU1->DDR0, CU2->DDR2, CU3->DDR3.
#   - Preflight physical LOC overlap across SLR0/2/3 before any expensive build.
#   - Certify only routed designs with WNS>=0, WHS>=0, exact MMCM D=1/M=5/O=3,
#     300 MHz input, 500 MHz double-rate clock, 2 BUFGCE_DIV per CU,
#     320 URAM per CU, exact vendor URAM LOC set, correct pblock membership,
#     and no forbidden v++ 60-2334 error.
#
# No XO regeneration. No make gen_xclbin. No card programming.
#
# V6 ALWAYS builds 1CU and 3CU in parallel.
# ============================================================================

set -Eeuo pipefail
umask 022
IFS=$'\n\t'

ROOT="${ROOT:-/home/truongnh/u250_clean_2021_2}"
VITIS_SETTINGS="${VITIS_SETTINGS:-/home/truongnh/local_vitis/Vitis/2021.2/settings64.sh}"
VITIS_AI_ROOT="$ROOT/source/Vitis-AI"
EXPECTED_VAI_COMMIT="${EXPECTED_VAI_COMMIT:-d02dcb6041663dbc7ecbc0c6af9fafa087a789de}"

PLATFORM="xilinx_u250_gen3x16_xdma_3_1_202020_1"
PLATFORM_SHORT="xilinx_u250_gen3x16_xdma_3_1"
PLATFORM_DIR="$ROOT/platform/tmp_deb/dev_extract/opt/xilinx/platforms/$PLATFORM"
PLATFORM_REPO_ROOT="$(dirname "$PLATFORM_DIR")"
XPFM="$PLATFORM_DIR/$PLATFORM.xpfm"

GOLDEN_XO="${GOLDEN_XO:-$ROOT/09_results/golden_f500_xo_qualify_20260921_131323/GOLDEN_F500_XO/DPUCADF8H_U250_300MHz_to_500MHz_GOLDEN.xo}"
EXPECTED_GOLDEN_SHA="${EXPECTED_GOLDEN_SHA:-20d4737661c04c276cba8e81f639d8ff598ea1e6c37a6f9335bf7edc0e48b142}"

# Existing 1CU artifacts from the prior routed/timing-closed run.
# v3 intentionally renames an uncertified xclbin to *.REJECTED on worker exit.
# V6.1 preserves whichever prior artifacts actually exist, byte-for-byte.
PRESERVED_1CU_DIR="${PRESERVED_1CU_DIR:-$ROOT/05_dpu/parallel_golden_f500_v3_20260921_170841/1cu}"
PRESERVED_1CU_EXPLICIT="${PRESERVED_1CU:-}"
REQUIRE_PRESERVED_1CU="${REQUIRE_PRESERVED_1CU:-1}"

ML_SHELL_ROOT="$VITIS_AI_ROOT/dsa/DPUCADF8H-XO/projects/ml_shell"
HOOK_SOURCE_ROOT="$ML_SHELL_ROOT/scripts/syn_tcl"
USER_SETUP_SOURCE_ROOT="$ML_SHELL_ROOT/scripts/user_setup"
PROC_TCL_SOURCE_ROOT="$ML_SHELL_ROOT/scripts/proc_tcl"
PHYS_ROOT="$ML_SHELL_ROOT/src/constraints/physical"

CPUSET_1CU="${CPUSET_1CU:-0-34}"
CPUSET_3CU="${CPUSET_3CU:-35-69}"
VPP_JOBS="${VPP_JOBS:-8}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/05_dpu/crossslr_golden_f500_v6_1_$TS"
LOG="$ROOT/08_logs/crossslr_golden_f500_v6_1_$TS"
RES="$ROOT/09_results/crossslr_golden_f500_v6_1_$TS"
SHARED="$RUN/shared"

mkdir -p "$RUN" "$LOG" "$RES" "$SHARED"
exec > >(tee -a "$LOG/master.log") 2>&1

die(){
  echo "RESULT=FAIL" >&2
  echo "REASON=$1" >&2
  echo "RUN=$RUN" >&2
  echo "LOG=$LOG" >&2
  echo "RES=$RES" >&2
  exit "${2:-1}"
}

clean_python(){
  env -u LD_LIBRARY_PATH -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 "$@"
}

echo "===== 0. GLOBAL CONTRACT ====="
echo "PARALLEL_BUILD=1CU+3CU"
echo "RUN=$RUN"
echo "PRESERVED_1CU_DIR=$PRESERVED_1CU_DIR"
echo "GOLDEN_XO=$GOLDEN_XO"
echo "PLATFORM=$PLATFORM"

[[ -f "$XPFM" ]] || die "XPFM missing: $XPFM"
[[ -f "$VITIS_SETTINGS" ]] || die "Vitis settings missing: $VITIS_SETTINGS"
[[ -d "$VITIS_AI_ROOT" ]] || die "Vitis-AI root missing: $VITIS_AI_ROOT"
[[ -d "$ML_SHELL_ROOT/src" ]] || die "constraint tree missing: $ML_SHELL_ROOT/src"

for f in link_post.tcl synth_pre.tcl synth_post.tcl opt_pre.tcl opt_post.tcl place_pre.tcl place_post.tcl route_pre.tcl route_post.tcl; do
  [[ -f "$HOOK_SOURCE_ROOT/$f" ]] || die "required hook missing: $HOOK_SOURCE_ROOT/$f"
done
[[ -f "$USER_SETUP_SOURCE_ROOT/user_setup.tcl" ]] || die "user_setup.tcl missing"
[[ -f "$PROC_TCL_SOURCE_ROOT/proc_vivado.tcl" ]] || die "proc_vivado.tcl missing"

for s in 0 2 3; do
  [[ -f "$PHYS_ROOT/${PLATFORM_SHORT}_slr${s}_physical.tcl" ]] \
    || die "physical constraint missing for SLR$s"
done

# ----------------------------------------------------------------------------
# 1. Verify the exact Vitis-AI source commit WITHOUT using `git -C`.
# ----------------------------------------------------------------------------
echo "===== 1. VITIS-AI COMMIT ====="
if [[ ! -d "$VITIS_AI_ROOT/.git" ]]; then
  die "Vitis-AI .git metadata missing; exact commit cannot be proven"
fi
ACTUAL_VAI_COMMIT="$(
  cd "$VITIS_AI_ROOT"
  git rev-parse HEAD
)"
echo "VITIS_AI_COMMIT=$ACTUAL_VAI_COMMIT"
[[ "$ACTUAL_VAI_COMMIT" == "$EXPECTED_VAI_COMMIT" ]] \
  || die "Vitis-AI commit mismatch: got=$ACTUAL_VAI_COMMIT expected=$EXPECTED_VAI_COMMIT"
echo "VITIS_AI_COMMIT_CHECK=PASS"

# ----------------------------------------------------------------------------
# 2. Preserve prior 1CU artifacts by hash, and never touch their directory.
#
# Important: v3's RETURN trap renames an xclbin to *.REJECTED whenever the
# worker fails certification. Therefore both:
#   1cu_f500.xclbin
#   1cu_f500.xclbin.REJECTED
# are valid prior artifacts to preserve for later card/runtime evaluation.
# ----------------------------------------------------------------------------
echo "===== 2. PRESERVE EXISTING 1CU ARTIFACT(S) ====="

declare -a PRESERVED_1CU_FILES=()

if [[ -n "$PRESERVED_1CU_EXPLICIT" ]]; then
  [[ -f "$PRESERVED_1CU_EXPLICIT" ]] \
    || die "explicit PRESERVED_1CU file missing: $PRESERVED_1CU_EXPLICIT"
  PRESERVED_1CU_FILES+=("$PRESERVED_1CU_EXPLICIT")
else
  for cand in \
    "$PRESERVED_1CU_DIR/1cu_f500.xclbin" \
    "$PRESERVED_1CU_DIR/1cu_f500.xclbin.REJECTED"
  do
    [[ -f "$cand" ]] && PRESERVED_1CU_FILES+=("$cand")
  done

  # Also preserve any certification/copy variant from this exact old run.
  while IFS= read -r -d '' cand; do
    seen=0
    for x in "${PRESERVED_1CU_FILES[@]:-}"; do
      [[ "$x" == "$cand" ]] && seen=1 && break
    done
    [[ "$seen" -eq 1 ]] || PRESERVED_1CU_FILES+=("$cand")
  done < <(
    find "$PRESERVED_1CU_DIR" -maxdepth 1 -type f \
      \( -name '1cu*.xclbin' -o -name '1cu*.xclbin.REJECTED' \) \
      -print0 2>/dev/null
  )
fi

if [[ "${#PRESERVED_1CU_FILES[@]}" -eq 0 ]]; then
  echo "PRESERVED_1CU_FOUND=NO"
  if [[ "$REQUIRE_PRESERVED_1CU" == "1" ]]; then
    echo "Directory listing for diagnosis:"
    ls -lah "$PRESERVED_1CU_DIR" 2>/dev/null || true
    die "no existing 1CU xclbin artifact found in: $PRESERVED_1CU_DIR"
  fi
fi

PRESERVED_MANIFEST_BEFORE="$RES/PRESERVED_1CU_BEFORE.sha256"
: > "$PRESERVED_MANIFEST_BEFORE"

for f in "${PRESERVED_1CU_FILES[@]:-}"; do
  sha256sum "$f" >> "$PRESERVED_MANIFEST_BEFORE"
  {
    echo "PRESERVED_1CU_FOUND=YES"
    echo "PRESERVED_1CU_FILE=$f"
    echo "PRESERVED_1CU_SHA256=$(sha256sum "$f" | awk '{print $1}')"
    stat "$f" || true
    echo
  } >> "$RES/PRESERVED_1CU_MANIFEST.txt"
done

if [[ "${#PRESERVED_1CU_FILES[@]}" -gt 0 ]]; then
  echo "PRESERVED_1CU_ARTIFACT_COUNT=${#PRESERVED_1CU_FILES[@]}"
  cat "$PRESERVED_MANIFEST_BEFORE"
  echo "PRESERVED_1CU_SNAPSHOT=PASS"
fi

# ----------------------------------------------------------------------------
# 3. Lock the exact certified Golden F500 XO.
# ----------------------------------------------------------------------------
echo "===== 3. GOLDEN XO ====="
[[ -f "$GOLDEN_XO" ]] || die "GOLDEN_XO missing: $GOLDEN_XO"
ACTUAL_GOLDEN_SHA="$(sha256sum "$GOLDEN_XO" | awk '{print $1}')"
echo "SOURCE_GOLDEN_SHA256=$ACTUAL_GOLDEN_SHA"
[[ "$ACTUAL_GOLDEN_SHA" == "$EXPECTED_GOLDEN_SHA" ]] \
  || die "GOLDEN XO SHA mismatch: got=$ACTUAL_GOLDEN_SHA expected=$EXPECTED_GOLDEN_SHA"

LOCKED_XO="$SHARED/DPUCADF8H_U250_300_to_500_GOLDEN.xo"
cp -a "$GOLDEN_XO" "$LOCKED_XO"
chmod 0444 "$LOCKED_XO"
LOCKED_SHA="$(sha256sum "$LOCKED_XO" | awk '{print $1}')"
[[ "$LOCKED_SHA" == "$EXPECTED_GOLDEN_SHA" ]] || die "working Golden XO SHA mismatch"
echo "$LOCKED_SHA  $LOCKED_XO" > "$RES/GOLDEN_XO.sha256"
echo "IMMUTABLE_GOLDEN_XO=PASS"

# ----------------------------------------------------------------------------
# 4. Physical-constraint preflight.
#
# Parse every vendor set_property LOC assignment for SLR0/2/3 and prove that
# there is NO duplicate physical LOC across the three floorplans.
# This directly checks the concern that 3CU could target the same 8 URAM.
# ----------------------------------------------------------------------------
echo "===== 4. PHYSICAL LOC OVERLAP PREFLIGHT ====="
clean_python - \
  "$PHYS_ROOT/${PLATFORM_SHORT}_slr0_physical.tcl" \
  "$PHYS_ROOT/${PLATFORM_SHORT}_slr2_physical.tcl" \
  "$PHYS_ROOT/${PLATFORM_SHORT}_slr3_physical.tcl" \
  "$RES/physical_floorplan_preflight.txt" <<'PY_PREFLIGHT'
import re, sys
from pathlib import Path

files=[Path(x) for x in sys.argv[1:4]]
out=Path(sys.argv[4])

# Parse active Tcl commands only; ignore blank/comment lines.
loc_re=re.compile(r'^\s*set_property\s+LOC\s+(\S+)\s+\[get_cells\b')
data={}
for f in files:
    locs=[]
    for raw in f.read_text(errors='replace').splitlines():
        stripped=raw.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        m=loc_re.search(raw)
        if m:
            locs.append(m.group(1))
    urams=[x for x in locs if x.startswith('URAM288_')]
    dup=sorted({x for x in locs if locs.count(x)>1})
    if dup:
        raise SystemExit(f'duplicate active LOC inside {f.name}: {dup[:40]}')
    if len(set(urams)) != 320:
        raise SystemExit(f'{f.name}: expected exactly 320 unique active URAM LOCs, got {len(set(urams))}')
    data[f.name]={'all':set(locs),'uram':set(urams)}

names=list(data)
lines=['PHYSICAL_FLOORPLAN_PREFLIGHT=PASS_CANDIDATE']
bad=False

for n in names:
    ur=sorted(data[n]['uram'])
    lines.append(f'{n}: ACTIVE_LOC_COUNT={len(data[n]["all"])} UNIQUE_URAM_COUNT={len(ur)}')
    lines.append(f'{n}: URAM_MINMAX={ur[0]}..{ur[-1]}')

for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=names[i],names[j]
        common_all=sorted(data[a]['all'] & data[b]['all'])
        common_uram=sorted(data[a]['uram'] & data[b]['uram'])
        lines.append(f'OVERLAP {a} vs {b}: ALL_LOC={len(common_all)} URAM_LOC={len(common_uram)}')
        if common_all:
            bad=True
            lines.append('COMMON_ALL='+' '.join(common_all[:200]))
        if common_uram:
            bad=True
            lines.append('COMMON_URAM='+' '.join(common_uram[:200]))

# Previously proven SLR0 cross-URAM set must still be present in this exact source.
expected_slr0_cross={f'URAM288_X1Y{y}' for y in range(64,72)}
slr0_name=next(n for n in names if '_slr0_' in n)
missing=sorted(expected_slr0_cross-data[slr0_name]['uram'])
lines.append('SLR0_PROVEN_CROSS_URAM='+' '.join(sorted(expected_slr0_cross)))
lines.append('SLR0_PROVEN_CROSS_URAM_MISSING='+('NONE' if not missing else ' '.join(missing)))
if missing:
    bad=True

if not bad:
    lines.append('ALL_VENDOR_HARD_LOC_SETS_PAIRWISE_DISJOINT=YES')
    lines.append('ALL_VENDOR_URAM_LOC_SETS_PAIRWISE_DISJOINT=YES')
    lines.append('PHYSICAL_FLOORPLAN_PREFLIGHT=PASS')

out.write_text('\n'.join(lines)+'\n')
print(out.read_text(),end='')
if bad:
    raise SystemExit('physical floorplan preflight failed')
PY_PREFLIGHT
echo "PHYSICAL_LOC_OVERLAP_CHECK=PASS"

# Ensure worker pblock expansion logic contains the known cross-SLR regions.
grep -Fq 'URAM288_X0Y64:URAM288_X1Y79' "$HOOK_SOURCE_ROOT/opt_pre.tcl" \
  || die "expected SLR0 cross-URAM pblock expansion missing"
grep -Fq 'URAM288_X0Y112:URAM288_X1Y127' "$HOOK_SOURCE_ROOT/opt_pre.tcl" \
  || die "expected SLR2 cross-URAM pblock expansion missing"
echo "VENDOR_CROSS_URAM_PBLOCK_RULES=PASS"

# ----------------------------------------------------------------------------
# 5. CPU/toolchain precheck.
# ----------------------------------------------------------------------------
echo "===== 5. TOOLCHAIN / CPU ====="
taskset -c "$CPUSET_1CU" /bin/true || die "CPUSET_1CU=$CPUSET_1CU not allowed"
taskset -c "$CPUSET_3CU" /bin/true || die "CPUSET_3CU=$CPUSET_3CU not allowed"
clean_python - "$CPUSET_1CU" "$CPUSET_3CU" <<'PY_CPU'
import sys
def expand(s):
    r=set()
    for p in s.split(','):
        if '-' in p:
            a,b=map(int,p.split('-',1)); r.update(range(a,b+1))
        else:
            r.add(int(p))
    return r
a,b=expand(sys.argv[1]),expand(sys.argv[2])
if a & b:
    raise SystemExit('CPU sets overlap: %s' % sorted(a&b))
if not a or not b:
    raise SystemExit('CPU set must not be empty')
print('CPUSET_DISJOINT=PASS')
print('CPUSET_1CU_COUNT=%d' % len(a))
print('CPUSET_3CU_COUNT=%d' % len(b))
PY_CPU

unset LD_LIBRARY_PATH PYTHONHOME PYTHONPATH || true
# shellcheck disable=SC1090
source "$VITIS_SETTINGS"
export PLATFORM_REPO_PATHS="$PLATFORM_REPO_ROOT${PLATFORM_REPO_PATHS:+:$PLATFORM_REPO_PATHS}"

v++ --version | head -8 | tee "$RES/vpp_version.txt"
vivado -version | head -8 | tee "$RES/vivado_version.txt"
grep -q '2021\.2' "$RES/vpp_version.txt" || die "v++ is not 2021.2"
grep -q '2021\.2' "$RES/vivado_version.txt" || die "Vivado is not 2021.2"

# ----------------------------------------------------------------------------
# 6. Cross-SLR-aware routed auditor.
# ----------------------------------------------------------------------------
AUDITOR="$SHARED/audit_routed_crossslr.tcl"
cat > "$AUDITOR" <<'TCL_AUDIT'
set dcp [lindex $argv 0]
set out [lindex $argv 1]
set expected_cu [lindex $argv 2]
set slrs [lrange $argv 3 end]

open_checkpoint $dcp
set fd [open $out w]
proc P {fd s} { puts $fd $s; puts $s }
proc GP {obj prop} {
  if {[catch {get_property $prop $obj} v]} { return "<UNAVAILABLE>" }
  return $v
}
proc slr_of_loc {loc} {
  if {$loc eq ""} { return "NO_LOC" }
  set site [get_sites -quiet $loc]
  if {![llength $site]} { return "NO_SITE" }
  set s [get_slrs -quiet -of_objects $site]
  if {[llength $s]} { return [lindex $s 0] }
  set tile [get_tiles -quiet -of_objects $site]
  if {[llength $tile]} {
    set s [get_slrs -quiet -of_objects $tile]
    if {[llength $s]} { return [lindex $s 0] }
  }
  return "NO_SLR"
}

P $fd "PART=[get_property PART [current_design]]"
P $fd "EXPECTED_PART=xcu250-figd2104-2L-e"

set sp [get_timing_paths -quiet -delay_type max -max_paths 1 -nworst 1]
set hp [get_timing_paths -quiet -delay_type min -max_paths 1 -nworst 1]
P $fd "WNS=[expr {[llength $sp] ? [get_property SLACK [lindex $sp 0]] : "NO_PATH"}]"
P $fd "WHS=[expr {[llength $hp] ? [get_property SLACK [lindex $hp 0]] : "NO_PATH"}]"
report_timing_summary -delay_type min_max -check_timing_verbose -report_unconstrained -max_paths 10 -file "${out}.timing_summary.rpt"

set mmcms [get_cells -hier -quiet -filter {REF_NAME == MMCME4_ADV && NAME =~ *DPUCADF8H_*/inst/u_clk_wiz/inst/mmcme4_adv_inst}]
P $fd "MMCM_COUNT=[llength $mmcms]"
foreach m [lsort $mmcms] {
  P $fd "MMCM=$m D=[GP $m DIVCLK_DIVIDE] M=[GP $m CLKFBOUT_MULT_F] O=[GP $m CLKOUT0_DIVIDE_F] CLKIN1_PERIOD=[GP $m CLKIN1_PERIOD]"
}

set bufgs [get_cells -hier -quiet -filter {REF_NAME == BUFGCE_DIV && NAME =~ *DPUCADF8H_*}]
P $fd "DPU_BUFGCE_DIV_COUNT=[llength $bufgs]"
foreach b [lsort $bufgs] {
  P $fd "BUFGCE_DIV=$b DIVIDE=[GP $b BUFGCE_DIVIDE]"
}

foreach s $slrs {
  foreach suffix {ACLK_DR ACLK} {
    set n "${s}_${suffix}"
    set cs [get_clocks -quiet $n]
    P $fd "CLOCK_COUNT name=$n count=[llength $cs]"
    foreach c $cs {
      P $fd "CLOCK=$n PERIOD=[GP $c PERIOD] MASTER=[GP $c MASTER_CLOCK] DIVIDE_BY=[GP $c DIVIDE_BY] MULTIPLY_BY=[GP $c MULTIPLY_BY]"
    }
  }
}

# CU-specific physical audit.
for {set i 1} {$i <= $expected_cu} {incr i} {
  set logical_slr [lindex $slrs [expr {$i-1}]]
  set cu_pat "*DPUCADF8H_${i}"
  set cu [get_cells -hier -quiet -filter "NAME =~ $cu_pat"]
  # Prefer the kernel root whose name ends exactly in DPUCADF8H_i.
  set roots {}
  foreach c $cu {
    if {[regexp "DPUCADF8H_${i}$" [get_property NAME $c]]} {
      lappend roots $c
    }
  }
  P $fd "CU${i}_ROOT_COUNT=[llength $roots]"
  if {[llength $roots]} {
    set root [lindex $roots 0]
    P $fd "CU${i}_ROOT=[get_property NAME $root]"
    set pbs [get_pblocks -quiet -of_objects $root]
    P $fd "CU${i}_ROOT_PBLOCKS=[join $pbs ,]"
  }

  set urams [get_cells -hier -quiet -filter "NAME =~ *DPUCADF8H_${i}* && REF_NAME == URAM288"]
  P $fd "CU${i}_URAM_COUNT=[llength $urams]"
  array unset cnt
  array set cnt {}
  set locs {}
  foreach u $urams {
    set loc [GP $u LOC]
    lappend locs $loc
    set s [slr_of_loc $loc]
    if {![info exists cnt($s)]} { set cnt($s) 0 }
    incr cnt($s)
  }
  foreach s [lsort [array names cnt]] {
    P $fd "CU${i}_URAM_IN_${s}=$cnt($s)"
  }
  P $fd "CU${i}_URAM_LOCS=[join [lsort -unique $locs] ,]"
  set outside 0
  foreach s [array names cnt] {
    if {$s ne $logical_slr} { incr outside $cnt($s) }
  }
  P $fd "CU${i}_URAM_OUTSIDE_LOGICAL_SLR=$outside"
  set expected_pb "pblock_dynamic_${logical_slr}"
  set pb [get_pblocks -quiet $expected_pb]
  P $fd "CU${i}_EXPECTED_PBLOCK=$expected_pb"
  P $fd "CU${i}_EXPECTED_PBLOCK_EXISTS=[llength $pb]"
  if {[llength $pb]} {
    P $fd "CU${i}_PBLOCK_GRID_RANGES=[GP $pb GRID_RANGES]"
  }
}

close $fd
close_design
exit
TCL_AUDIT

# ----------------------------------------------------------------------------
# 7. Implementation strategy.
# ----------------------------------------------------------------------------
COMMON_IMPL_XP=(
  --xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore"
  --xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_HighUtilSLRs"
  --xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS=-debug_log"
  --xp "vivado_prop:run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore"
  --xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS=-tns_cleanup"
  --xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation"
  --xp "vivado_prop:run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore"
  --xp "param:compiler.interfaceWrBurstLen=16"
  --xp "param:compiler.interfaceRdBurstLen=16"
  --xp "param:compiler.interfaceWrOutstanding=32"
  --xp "param:compiler.interfaceRdOutstanding=32"
)
printf '%q ' "${COMMON_IMPL_XP[@]}" > "$RES/common_impl_xp_args.shell"
printf '\n' >> "$RES/common_impl_xp_args.shell"

! grep -Fq 'worstNegativeSlack' "$RES/common_impl_xp_args.shell" \
  || die "worstNegativeSlack override leaked into command"
echo "WORST_NEGATIVE_SLACK_OVERRIDE=ABSENT"

# ----------------------------------------------------------------------------
# 8. Worker.
# ----------------------------------------------------------------------------
worker(){
  local tag="$1" cpuset="$2" expected_cu="$3" slr_csv="$4"
  shift 4
  local -a conn=("$@")

  local wd="$RUN/$tag"
  local tmp="$wd/vpp_tmp"
  local outdir="$RES/$tag"
  local logdir="$LOG/$tag"
  local xclbin="$wd/${tag}_f500_crossslr.xclbin"
  local status="$outdir/FINAL_STATUS.txt"
  local audit="$outdir/routed_audit.txt"
  local valid=0

  mkdir -p "$wd" "$tmp" "$outdir" "$logdir"

  trap 'rc=$?; if [[ "$valid" -ne 1 && -f "$xclbin" ]]; then mv -f "$xclbin" "${xclbin}.REJECTED" || true; fi; return $rc' RETURN

  # ------------------------------------------------------------------------
  # Isolated worker-local ml_shell.
  # ------------------------------------------------------------------------
  local hook_ml="$wd/hooks/projects/ml_shell"
  local hook_syn="$hook_ml/scripts/syn_tcl"
  mkdir -p \
    "$hook_syn" \
    "$hook_ml/scripts/user_setup" \
    "$hook_ml/scripts/proc_tcl" \
    "$hook_ml/outputs/reports" \
    "$hook_ml/outputs/checkpoints" \
    "$hook_ml/outputs/logs"

  cp -a "$HOOK_SOURCE_ROOT/." "$hook_syn/"
  cp -a "$USER_SETUP_SOURCE_ROOT/user_setup.tcl" "$hook_ml/scripts/user_setup/user_setup.tcl"
  cp -a "$PROC_TCL_SOURCE_ROOT/proc_vivado.tcl" "$hook_ml/scripts/proc_tcl/proc_vivado.tcl"
  ln -s "$ML_SHELL_ROOT/src" "$hook_ml/src"

  # Fix known non-portable glob failure in make_outputs().
  clean_python - "$hook_ml/scripts/proc_tcl/proc_vivado.tcl" <<'PY_PATCH_PROC'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
old='foreach f [glob $BUILD_DIR/link/vivado/vpl/*.log] {file copy -force $f $LOG_DIR}'
new='foreach f [glob -nocomplain $BUILD_DIR/link/vivado/vpl/*.log] {file copy -force $f $LOG_DIR}'
if old not in s:
    raise SystemExit('expected make_outputs glob line not found')
p.write_text(s.replace(old,new,1))
PY_PATCH_PROC

  # Worker-specific environment: exact CU count, logical SLRs, FREQ=500,
  # and the real temp directory. No stock /group/bcapps path survives.
  {
    echo 'puts "Start to source [info script]"'
    echo 'set BOARD "u250"'
    echo "set DPU_NUM $expected_cu"
    echo 'set FREQ 500'
    echo "set PLATFORM \"$PLATFORM_SHORT\""
    local idx=0 slr
    IFS=',' read -r -a _slr_arr <<< "$slr_csv"
    for slr in "${_slr_arr[@]}"; do
      echo "set DPU_SLR_IDX($idx) ${slr#SLR}"
      idx=$((idx+1))
    done
    printf 'set BUILD_DIR {%s}\n' "$tmp"
  } > "$hook_ml/scripts/user_setup/env_config.tcl"

  # Append manual logical CU->pblock binding to opt_pre.
  # This reproduces the placement intent of --connectivity.slr without
  # emitting strict Vitis SLR metadata that rejects intentional cross-SLR LOCs.
  cat >> "$hook_syn/opt_pre.tcl" <<'TCL_MANUAL_PBLOCK'

# ---------------------------------------------------------------------------
# V6.1 manual CU -> pblock binding.
# Run after the vendor pblock resize above so intentional cross-SLR ranges
# are already present before constraining the kernel root.
# ---------------------------------------------------------------------------
for {set idx 0} {$idx < $DPU_NUM} {incr idx} {
    set slr $DPU_SLR_IDX($idx)
    set wrap_var "SLR${slr}_DPU_V3_WRAP_TOP"
    set wrap_top [set $wrap_var]
    regsub {/inst$} $wrap_top "" cu_top

    set pb [get_pblocks -quiet "pblock_dynamic_SLR${slr}"]
    set cu [get_cells -quiet $cu_top]

    if {[llength $pb] != 1} {
        error "V6.1: expected exactly one pblock_dynamic_SLR${slr}, got [llength $pb]"
    }
    if {[llength $cu] != 1} {
        error "V6.1: expected exactly one CU root $cu_top, got [llength $cu]"
    }

    set current_pbs [get_pblocks -quiet -of_objects $cu]
    if {[lsearch -exact $current_pbs "pblock_dynamic_SLR${slr}"] < 0} {
        add_cells_to_pblock $pb $cu
    }

    set verify_pbs [get_pblocks -quiet -of_objects $cu]
    if {[lsearch -exact $verify_pbs "pblock_dynamic_SLR${slr}"] < 0} {
        error "V6.1: CU $cu_top was not bound to pblock_dynamic_SLR${slr}"
    }
    puts "V6_1_MANUAL_PBLOCK_BIND CU=$cu_top PBLOCK=pblock_dynamic_SLR${slr}"
}
TCL_MANUAL_PBLOCK

  # Fail closed on known configuration regressions.
  if grep -RFn '/group/bcapps/fanzhang' "$hook_ml/scripts" > "$outdir/forbidden_hook_paths.txt" 2>/dev/null; then
    printf 'RESULT=FAIL\nREASON=forbidden_internal_BUILD_DIR_leaked_into_worker_hooks\n' > "$status"
    return 9
  fi
  grep -qx "set DPU_NUM $expected_cu" "$hook_ml/scripts/user_setup/env_config.tcl" \
    || { printf 'RESULT=FAIL\nREASON=worker_hook_DPU_NUM_mismatch\n' > "$status"; return 9; }
  grep -qx 'set FREQ 500' "$hook_ml/scripts/user_setup/env_config.tcl" \
    || { printf 'RESULT=FAIL\nREASON=worker_hook_FREQ_not_500\n' > "$status"; return 9; }
  grep -Fq 'V6_1_MANUAL_PBLOCK_BIND' "$hook_syn/opt_pre.tcl" \
    || { printf 'RESULT=FAIL\nREASON=manual_pblock_patch_missing\n' > "$status"; return 9; }

  cp -a "$hook_ml/scripts/user_setup/env_config.tcl" "$outdir/hook_env_config.tcl"
  cp -a "$hook_syn/opt_pre.tcl" "$outdir/hook_opt_pre.tcl"

  local -a WORKER_XP=(
    --xp "param:compiler.userPostSysLinkTcl=$hook_syn/link_post.tcl"
    --xp "vivado_prop:run.my_rm_synth_1.STEPS.SYNTH_DESIGN.TCL.PRE={$hook_syn/synth_pre.tcl}"
    --xp "vivado_prop:run.my_rm_synth_1.STEPS.SYNTH_DESIGN.TCL.POST={$hook_syn/synth_post.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.PRE={$hook_syn/opt_pre.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.POST={$hook_syn/opt_post.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE={$hook_syn/place_pre.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.POST={$hook_syn/place_post.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.PRE={$hook_syn/route_pre.tcl}"
    --xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST={$hook_syn/route_post.tcl}"
    "${COMMON_IMPL_XP[@]}"
  )

  printf '%q ' "${WORKER_XP[@]}" > "$outdir/worker_xp_args.shell"
  printf '\n' >> "$outdir/worker_xp_args.shell"

  {
    echo "TAG=$tag"
    echo "CPUSET=$cpuset"
    echo "EXPECTED_CU=$expected_cu"
    echo "LOGICAL_SLRS=$slr_csv"
    echo "GOLDEN_XO=$LOCKED_XO"
    echo "GOLDEN_SHA=$EXPECTED_GOLDEN_SHA"
    echo "EXTERNAL_KERNEL_FREQUENCY=0:300|1:300"
    echo "HOOK_FREQ=500"
    echo "HOOK_BUILD_DIR=$tmp"
    echo "VITIS_CONNECTIVITY_SLR=OMITTED_INTENTIONALLY"
    echo "MANUAL_CU_PBLOCK_BIND=YES"
    echo "VENDOR_PHYSICAL_CONSTRAINTS=YES"
    echo "PAIRWISE_VENDOR_LOC_PREFLIGHT=PASS"
  } > "$outdir/build_contract.txt"

  printf '%q ' "${conn[@]}" > "$outdir/connectivity_args.shell"
  printf '\n' >> "$outdir/connectivity_args.shell"
  if grep -Fq -- '--connectivity.slr' "$outdir/connectivity_args.shell"; then
    printf 'RESULT=FAIL\nREASON=connectivity_slr_must_be_omitted_for_crossslr_vendor_floorplan\n' > "$status"
    return 9
  fi

  local sha_before sha_after
  sha_before="$(sha256sum "$LOCKED_XO" | awk '{print $1}')"
  [[ "$sha_before" == "$EXPECTED_GOLDEN_SHA" ]] \
    || { printf 'RESULT=FAIL\nREASON=golden_sha_before_link_mismatch\n' > "$status"; return 10; }

  echo "[$tag] START $(date --iso-8601=seconds 2>/dev/null || date)"

  set +e
  (
    cd "$wd"
    taskset -c "$cpuset" v++ \
      -t hw \
      --link \
      --platform "$XPFM" \
      --save-temps \
      --jobs "$VPP_JOBS" \
      "${WORKER_XP[@]}" \
      --kernel_frequency "0:300|1:300" \
      "${conn[@]}" \
      --temp_dir "$tmp" \
      -o "$xclbin" \
      "$LOCKED_XO"
  ) > >(tee "$logdir/vpp_link.log") 2>&1
  local vpp_rc=$?
  set -e

  echo "VPP_RC=$vpp_rc" > "$outdir/vpp_status.txt"

  sha_after="$(sha256sum "$LOCKED_XO" | awk '{print $1}')"
  printf 'SHA_BEFORE=%s\nSHA_AFTER=%s\n' "$sha_before" "$sha_after" > "$outdir/golden_reuse.txt"
  [[ "$sha_after" == "$EXPECTED_GOLDEN_SHA" ]] \
    || { printf 'RESULT=FAIL\nREASON=golden_sha_changed_during_link\n' > "$status"; return 11; }

  # Known forbidden regressions.
  {
    grep -niE 'Synth 8-5809|Synth 8-6156|failed synthesizing module.*BUFGCE_DIV|v\+\+ 60-2334|span more than one SLR|ERROR:' "$logdir/vpp_link.log" 2>/dev/null || true
    find "$tmp" -type f \( -name '*.log' -o -name '*.rpt' -o -name '*.txt' \) -size -25M -print0 2>/dev/null \
      | xargs -0 -r grep -niE 'Synth 8-5809|Synth 8-6156|failed synthesizing module.*BUFGCE_DIV|v\+\+ 60-2334|span more than one SLR|ERROR:' 2>/dev/null || true
  } > "$outdir/error_focus.txt"

  if grep -qE 'Synth 8-5809|Synth 8-6156.*BUFGCE_DIV|failed synthesizing module.*BUFGCE_DIV' "$outdir/error_focus.txt" 2>/dev/null; then
    printf 'RESULT=FAIL\nREASON=BUFGCE_DIV_SYNTH_ERROR_RETURNED\n' > "$status"
    return 12
  fi

  if grep -qE 'v\+\+ 60-2334|span more than one SLR' "$logdir/vpp_link.log" 2>/dev/null; then
    printf 'RESULT=FAIL\nREASON=VITIS_STRICT_SLR_ERROR_60_2334_RETURNED\n' > "$status"
    return 13
  fi

  [[ "$vpp_rc" -eq 0 ]] \
    || { printf 'RESULT=FAIL\nREASON=vpp_link_nonzero_rc_%s\n' "$vpp_rc" > "$status"; return 14; }
  [[ -f "$xclbin" ]] \
    || { printf 'RESULT=FAIL\nREASON=xclbin_missing_after_vpp_success\n' > "$status"; return 15; }

  local dcp
  dcp="$(
    find "$tmp" -type f \( -name '*routed*.dcp' -o -name '*route*.dcp' \) \
      -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -1 | cut -d' ' -f2-
  )"
  [[ -n "$dcp" && -f "$dcp" ]] \
    || { printf 'RESULT=FAIL\nREASON=final_routed_dcp_missing\n' > "$status"; return 16; }

  echo "$dcp" > "$outdir/routed_dcp_path.txt"

  local slr_words="${slr_csv//,/ }"
  vivado -mode batch -nojournal -nolog \
    -source "$AUDITOR" \
    -tclargs "$dcp" "$audit" "$expected_cu" $slr_words \
    > "$logdir/routed_audit_vivado.log" 2>&1 \
    || { printf 'RESULT=FAIL\nREASON=vivado_routed_audit_failed\n' > "$status"; return 17; }

  # ------------------------------------------------------------------------
  # Strict routed certification, cross-SLR-aware.
  # ------------------------------------------------------------------------
  if ! env -u LD_LIBRARY_PATH -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - \
      "$audit" "$expected_cu" "$slr_csv" "$PHYS_ROOT" "$PLATFORM_SHORT" "$outdir/physical_certification.txt" <<'PY_CERT'
import re, sys, math
from pathlib import Path

audit=Path(sys.argv[1]).read_text(errors='replace')
n=int(sys.argv[2])
slrs=sys.argv[3].split(',')
phys_root=Path(sys.argv[4])
platform=sys.argv[5]
cert_out=Path(sys.argv[6])

def one(pattern,label):
    m=re.search(pattern,audit,re.M)
    if not m:
        raise SystemExit('missing '+label)
    return m.group(1)

part=one(r'^PART=(\S+)$','PART')
if part != 'xcu250-figd2104-2L-e':
    raise SystemExit(f'wrong FPGA part: {part}')

wns=float(one(r'^WNS=([-+]?[0-9]*\.?[0-9]+)$','WNS'))
whs=float(one(r'^WHS=([-+]?[0-9]*\.?[0-9]+)$','WHS'))
if wns < 0: raise SystemExit(f'WNS < 0: {wns}')
if whs < 0: raise SystemExit(f'WHS < 0: {whs}')

mc=int(one(r'^MMCM_COUNT=(\d+)$','MMCM_COUNT'))
if mc != n:
    raise SystemExit(f'MMCM_COUNT={mc} expected={n}')

rows=re.findall(
    r'^MMCM=.*? D=([0-9.]+) M=([0-9.]+) O=([0-9.]+) CLKIN1_PERIOD=([0-9.]+)$',
    audit,re.M)
if len(rows)!=n:
    raise SystemExit(f'MMCM detail count={len(rows)} expected={n}')
for d,m,o,p in rows:
    d,m,o,p=map(float,(d,m,o,p))
    if not math.isclose(d,1,abs_tol=1e-9): raise SystemExit('MMCM D != 1')
    if not math.isclose(m,5,abs_tol=1e-9): raise SystemExit('MMCM M != 5')
    if not math.isclose(o,3,abs_tol=1e-9): raise SystemExit('MMCM O != 3')
    if not math.isclose(p,3.333,abs_tol=.01): raise SystemExit('MMCM CLKIN1_PERIOD != 3.333ns')

bc=int(one(r'^DPU_BUFGCE_DIV_COUNT=(\d+)$','DPU_BUFGCE_DIV_COUNT'))
if bc != 2*n:
    raise SystemExit(f'BUFGCE_DIV count={bc} expected={2*n}')
buf_divs=[int(x) for x in re.findall(r'^BUFGCE_DIV=.*? DIVIDE=(\d+)$',audit,re.M)]
if len(buf_divs) != 2*n:
    raise SystemExit(f'BUFGCE_DIV detail count={len(buf_divs)} expected={2*n}')
if buf_divs.count(1) != n or buf_divs.count(2) != n:
    raise SystemExit(f'BUFGCE_DIV distribution invalid: {buf_divs}; expected {n}x/1 and {n}x/2')

for s in slrs:
    dr=re.search(r'^CLOCK=%s_ACLK_DR PERIOD=([0-9.]+)'%re.escape(s),audit,re.M)
    a =re.search(r'^CLOCK=%s_ACLK PERIOD=([0-9.]+)'%re.escape(s),audit,re.M)
    if not dr or not a:
        raise SystemExit('missing clocks for '+s)
    if not math.isclose(float(dr.group(1)),2.0,abs_tol=.01):
        raise SystemExit(s+'_ACLK_DR not 2.000ns / 500MHz')
    if not math.isclose(float(a.group(1)),4.0,abs_tol=.01):
        raise SystemExit(s+'_ACLK not 4.000ns / 250MHz')

loc_re=re.compile(r'\bset_property\s+LOC\s+(\S+)')

actual_sets=[]
physical_lines=[]
for i,s in enumerate(slrs, start=1):
    expected_pb=f'pblock_dynamic_{s}'
    pbs=one(rf'^CU{i}_ROOT_PBLOCKS=(.*)$',f'CU{i}_ROOT_PBLOCKS')
    if expected_pb not in [x for x in pbs.split(',') if x]:
        raise SystemExit(f'CU{i} root not bound to {expected_pb}: {pbs}')

    uram_count=int(one(rf'^CU{i}_URAM_COUNT=(\d+)$',f'CU{i}_URAM_COUNT'))
    if uram_count != 320:
        raise SystemExit(f'CU{i} URAM count={uram_count}, expected=320')

    loc_line=one(rf'^CU{i}_URAM_LOCS=(.*)$',f'CU{i}_URAM_LOCS')
    actual={x for x in loc_line.split(',') if x}
    if len(actual)!=320:
        raise SystemExit(f'CU{i} unique routed URAM LOC count={len(actual)}, expected=320')

    slr_num=s.replace('SLR','')
    phys=phys_root/f'{platform}_slr{slr_num}_physical.tcl'
    expected=set()
    for raw in phys.read_text(errors="replace").splitlines():
        stripped=raw.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        m=loc_re.search(raw)
        if m and m.group(1).startswith('URAM288_'):
            expected.add(m.group(1))
    if len(expected)!=320:
        raise SystemExit(f'{phys.name}: expected 320 URAM LOC declarations, got {len(expected)}')
    if actual != expected:
        miss=sorted(expected-actual)[:20]
        extra=sorted(actual-expected)[:20]
        raise SystemExit(f'CU{i} URAM LOC set differs from vendor physical floorplan; missing={miss} extra={extra}')
    # Capture physical distribution, without assuming every CU spills the same count.
    dist={}
    for phys_slr,count in re.findall(rf'^CU{i}_URAM_IN_(\S+)=(\d+)$',audit,re.M):
        dist[phys_slr]=int(count)
    outside=int(one(rf'^CU{i}_URAM_OUTSIDE_LOGICAL_SLR=(\d+)$',f'CU{i}_URAM_OUTSIDE_LOGICAL_SLR'))
    if sum(dist.values()) != 320:
        raise SystemExit(f'CU{i} physical URAM distribution does not sum to 320: {dist}')
    physical_lines.append(f'CU{i}_LOGICAL_SLR={s}')
    physical_lines.append(f'CU{i}_URAM_DISTRIBUTION={dist}')
    physical_lines.append(f'CU{i}_URAM_OUTSIDE_LOGICAL_SLR={outside}')
    physical_lines.append(f'CU{i}_VENDOR_URAM_LOC_MATCH=YES')
    actual_sets.append(actual)

for i in range(len(actual_sets)):
    for j in range(i+1,len(actual_sets)):
        common=actual_sets[i] & actual_sets[j]
        if common:
            raise SystemExit(f'CU{i+1}/CU{j+1} routed URAM LOC overlap: {sorted(common)[:20]}')
        physical_lines.append(f'CU{i+1}_CU{j+1}_URAM_LOC_OVERLAP=0')

physical_lines += [
    f'PART={part}',
    f'WNS={wns}',
    f'WHS={whs}',
    'MMCM_INPUT_MHZ=300',
    'MMCM_D=1',
    'MMCM_M=5',
    'MMCM_O=3',
    'MMCM_OUTPUT_MHZ=500',
    f'BUFGCE_DIV_1_COUNT={buf_divs.count(1)}',
    f'BUFGCE_DIV_2_COUNT={buf_divs.count(2)}',
    'CROSS_SLR_TIMING_CLOCK_PHYSICAL_AUDIT=PASS'
]
cert_out.write_text('\n'.join(physical_lines)+'\n')
print('CROSS_SLR_TIMING_CLOCK_PHYSICAL_AUDIT=PASS')
PY_CERT
  then
    printf 'RESULT=FAIL\nREASON=crossslr_timing_clock_physical_audit_failed\n' > "$status"
    return 18
  fi

  # ------------------------------------------------------------------------
  # XCLBIN metadata.
  # ------------------------------------------------------------------------
  xclbinutil --info --input "$xclbin" > "$outdir/xclbin_info.txt" 2>&1 \
    || { printf 'RESULT=FAIL\nREASON=xclbinutil_info_failed\n' > "$status"; return 19; }

  xclbinutil --force --input "$xclbin" \
    --dump-section IP_LAYOUT:JSON:"$outdir/ip_layout.json" \
    --dump-section CONNECTIVITY:JSON:"$outdir/connectivity.json" \
    --dump-section MEM_TOPOLOGY:JSON:"$outdir/mem_topology.json" \
    --dump-section CLOCK_FREQ_TOPOLOGY:JSON:"$outdir/clock_freq_topology.json" \
    > "$outdir/xclbin_dump_sections.log" 2>&1 \
    || { printf 'RESULT=FAIL\nREASON=xclbin_metadata_dump_failed\n' > "$status"; return 19; }

  local cu_count
  cu_count="$(grep -oE 'DPUCADF8H_[123]' "$outdir/xclbin_info.txt" | sort -u | wc -l | tr -d ' ')"
  [[ "$cu_count" == "$expected_cu" ]] \
    || { printf 'RESULT=FAIL\nREASON=xclbin_CU_count_%s_expected_%s\n' "$cu_count" "$expected_cu" > "$status"; return 20; }

  grep -Eq 'DATA_CLK[^0-9]*300|300[^0-9]*MHz[^A-Za-z]*DATA_CLK' "$outdir/xclbin_info.txt" \
    || { printf 'RESULT=FAIL\nREASON=xclbin_DATA_CLK_300_not_proven\n' > "$status"; return 21; }
  grep -Eq 'KERNEL_CLK[^0-9]*300|300[^0-9]*MHz[^A-Za-z]*KERNEL_CLK' "$outdir/xclbin_info.txt" \
    || { printf 'RESULT=FAIL\nREASON=xclbin_KERNEL_CLK_300_not_proven\n' > "$status"; return 22; }

  local certified="$outdir/${tag}_DPUCADF8H_300_to_500_CROSSSLR_TIMING_CLOSED.xclbin"
  cp -a "$xclbin" "$certified"
  chmod 0444 "$certified"

  local xsha
  xsha="$(sha256sum "$certified" | awk '{print $1}')"
  echo "$xsha  $certified" > "$certified.sha256"

  {
    echo "RESULT=PASS"
    echo "TAG=$tag"
    echo "VITIS_AI_COMMIT=$EXPECTED_VAI_COMMIT"
    echo "GOLDEN_XO_SHA256=$EXPECTED_GOLDEN_SHA"
    echo "GOLDEN_XO_REUSED_BYTE_IDENTICAL=YES"
    echo "EXTERNAL_KERNEL_CLOCKS_MHZ=300/300"
    echo "MMCM_INPUT_MHZ=300"
    echo "MMCM_D=1"
    echo "MMCM_M=5"
    echo "MMCM_O=3"
    echo "MMCM_OUTPUT_MHZ=500"
    echo "DPU_DERIVATIVE_ACLK_MHZ=250"
    echo "CU_COUNT=$expected_cu"
    echo "LOGICAL_SLRS=$slr_csv"
    echo "VITIS_CONNECTIVITY_SLR=OMITTED"
    echo "MANUAL_PBLOCK_BIND=PASS"
    echo "VENDOR_URAM_LOC_SET=PASS"
    echo "URAM_LOC_OVERLAP_BETWEEN_CUS=ABSENT"
    echo "DDR_MAPPING_1CU=CU1->DDR0"
    if [[ "$expected_cu" == "3" ]]; then
      echo "DDR_MAPPING_3CU=CU1->DDR0,CU2->DDR2,CU3->DDR3"
    fi
    echo "PHYSICAL_CERTIFICATION=$outdir/physical_certification.txt"
    echo "VPP_60_2334=ABSENT"
    echo "WNS_GE_0=YES"
    echo "WHS_GE_0=YES"
    echo "ROUTED_CLOCK_AUDIT=PASS"
    echo "XCLBIN=$certified"
    echo "XCLBIN_SHA256=$xsha"
    echo "CARD_PROGRAMMED=NO"
  } > "$status"

  valid=1
  echo "[$tag] PASS $(date --iso-8601=seconds 2>/dev/null || date)"
}

# ----------------------------------------------------------------------------
# 9. Launch BOTH builds in parallel.
#
# IMPORTANT: no --connectivity.slr arguments here.
# Placement remains controlled by worker-local pblocks + vendor physical LOCs.
# DDR mapping remains fixed.
# ----------------------------------------------------------------------------
echo "===== 9. BUILD 1CU + 3CU IN PARALLEL ====="
echo "PARALLEL_BUILD=YES"
echo "GOLDEN_SHA256=$EXPECTED_GOLDEN_SHA"
echo "VPP_JOBS_PER_BUILD=$VPP_JOBS"
echo "CONNECTIVITY_SLR=OMITTED_INTENTIONALLY"

set +e

worker \
  1cu "$CPUSET_1CU" 1 SLR0 \
  --connectivity.nk DPUCADF8H:1:DPUCADF8H_1 \
  --connectivity.sp 'DPUCADF8H_1.m00_axi:DDR[0]' \
  > >(tee "$LOG/1cu_driver.log") 2>&1 &
PID1=$!

worker \
  3cu "$CPUSET_3CU" 3 SLR0,SLR2,SLR3 \
  --connectivity.nk DPUCADF8H:3:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3 \
  --connectivity.sp 'DPUCADF8H_1.m00_axi:DDR[0]' \
  --connectivity.sp 'DPUCADF8H_2.m00_axi:DDR[2]' \
  --connectivity.sp 'DPUCADF8H_3.m00_axi:DDR[3]' \
  > >(tee "$LOG/3cu_driver.log") 2>&1 &
PID3=$!

echo "PID_1CU=$PID1"
echo "PID_3CU=$PID3"

wait "$PID1"
RC1=$?
wait "$PID3"
RC3=$?

set -e

# ----------------------------------------------------------------------------
# 10. Verify every preserved prior 1CU artifact is byte-identical after build.
# ----------------------------------------------------------------------------
echo "===== 10. PRESERVED 1CU RECHECK ====="
PRESERVED_MANIFEST_AFTER="$RES/PRESERVED_1CU_AFTER.sha256"
: > "$PRESERVED_MANIFEST_AFTER"

for f in "${PRESERVED_1CU_FILES[@]:-}"; do
  [[ -f "$f" ]] || die "preserved 1CU artifact disappeared during new build: $f"
  sha256sum "$f" >> "$PRESERVED_MANIFEST_AFTER"
done

if [[ "${#PRESERVED_1CU_FILES[@]}" -gt 0 ]]; then
  if ! cmp -s "$PRESERVED_MANIFEST_BEFORE" "$PRESERVED_MANIFEST_AFTER"; then
    echo "--- BEFORE ---" >&2
    cat "$PRESERVED_MANIFEST_BEFORE" >&2
    echo "--- AFTER ---" >&2
    cat "$PRESERVED_MANIFEST_AFTER" >&2
    die "one or more preserved 1CU artifacts changed during new build"
  fi
  echo "PRESERVED_1CU_UNCHANGED=PASS"
fi

FINAL_GOLDEN_SHA="$(sha256sum "$LOCKED_XO" | awk '{print $1}')"
[[ "$FINAL_GOLDEN_SHA" == "$EXPECTED_GOLDEN_SHA" ]] \
  || die "immutable Golden XO changed after builds"

STATUS1="$RES/1cu/FINAL_STATUS.txt"
STATUS3="$RES/3cu/FINAL_STATUS.txt"

echo "--- 1CU STATUS ---"
[[ -f "$STATUS1" ]] && cat "$STATUS1" || echo "NOT_RUN_OR_MISSING"
echo "--- 3CU STATUS ---"
[[ -f "$STATUS3" ]] && cat "$STATUS3" || echo "NOT_RUN_OR_MISSING"

FINAL=PASS
[[ "$RC1" == "0" && -f "$STATUS1" ]] && grep -qx 'RESULT=PASS' "$STATUS1" || FINAL=FAIL
[[ "$RC3" == "0" && -f "$STATUS3" ]] && grep -qx 'RESULT=PASS' "$STATUS3" || FINAL=FAIL

PKG="$ROOT/08_logs/crossslr_golden_f500_v6_1_$TS.tar.gz"
tar --exclude='*.xclbin' -C "$RES" -czf "$PKG" .

echo "================================================================"
echo "RESULT=$FINAL"
echo "PARALLEL_BUILD=1CU+3CU"
echo "VITIS_AI_COMMIT=$EXPECTED_VAI_COMMIT"
echo "GOLDEN_XO_SHA256=$EXPECTED_GOLDEN_SHA"
echo "GOLDEN_XO_REUSED_BYTE_IDENTICAL=YES"
echo "PRESERVED_1CU_DIR=$PRESERVED_1CU_DIR"
echo "PRESERVED_1CU_UNCHANGED=$([[ "${#PRESERVED_1CU_FILES[@]}" -eq 0 ]] && echo NOT_CHECKED || echo YES)"
echo "RC_1CU=$RC1"
echo "RC_3CU=$RC3"
echo "RESULT_DIR=$RES"
echo "LOG_DIR=$LOG"
echo "PACKAGE=$PKG"
echo "CARD_PROGRAMMED=NO"
echo "BOTH_1CU_3CU_REQUIRED=YES"
echo "================================================================"

[[ "$FINAL" == "PASS" ]] || exit 100
