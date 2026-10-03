#!/usr/bin/env bash
# Phase 3: Slurm orchestration only.
#
# Defaults reproduce the supplied A100/MPS cluster contract:
#   nodes=1, ntasks=1, cpus-per-task=4, mem=32G,
#   time=72:00:00, gres=mps:1, REQUIRED_VRAM=25000 MiB.
#
# Usage on the login node:
#   bash job.sh submit

set -Eeuo pipefail
umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
TRAIN_ENGINE="${TRAIN_ENGINE:-$SCRIPT_DIR/train_multimodel_cifar100_a100.sh}"

export PROJECT_ROOT="${PROJECT_ROOT:-/datastore/icc_fce/resnet_paper_multimodel}"
export CONDA_ROOT="${CONDA_ROOT:-/datastore/icc_fce/miniconda3}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export WHEELHOUSE="${WHEELHOUSE:-$PROJECT_ROOT/wheelhouse}"
export OFFLINE="${OFFLINE:-0}"
export REQUIRED_VRAM="${REQUIRED_VRAM:-25000}"
export GPU_CHECK_HELPER="${GPU_CHECK_HELPER:-}"
export LOCK_TIMEOUT="${LOCK_TIMEOUT:-600}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
WORKER_CPUS="${WORKER_CPUS:-4}"
WORKER_MEM="${WORKER_MEM:-32G}"
WORKER_TIME="${WORKER_TIME:-72:00:00}"
SLURM_GRES="${SLURM_GRES:-mps:1}"
LOG_ROOT="${LOG_ROOT:-/datastore/icc_fce/logs}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_uint() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer: $2"
}

site_args() {
  SITE_ARGS=()
  [[ -z "$SLURM_PARTITION" ]] || SITE_ARGS+=(--partition "$SLURM_PARTITION")
  [[ -z "$SLURM_ACCOUNT" ]] || SITE_ARGS+=(--account "$SLURM_ACCOUNT")
}

verify_prepared() {
  [[ -f "$TRAIN_ENGINE" ]] || die "training engine not found: $TRAIN_ENGINE"
  [[ -f "$PROJECT_ROOT/provenance/PREPARED.env" ]] \
    || die "data/environment not prepared; run prepare_data_env_a100.sh first"
  [[ -f "$PROJECT_ROOT/provenance/preparation_manifest.sha256" ]] \
    || die "preparation manifest is missing"
  (
    cd "$PROJECT_ROOT"
    sha256sum -c provenance/preparation_manifest.sha256 >/dev/null
  ) || die "prepared input manifest verification failed"
  cmp -s "$TRAIN_ENGINE" \
    "$PROJECT_ROOT/provenance/train_multimodel_cifar100_a100.sh" \
    || die "training engine changed after preparation; rerun prepare_data_env_a100.sh"
  cmp -s "$SELF" "$PROJECT_ROOT/provenance/job.sh" \
    || die "job.sh changed after preparation; rerun prepare_data_env_a100.sh"
}

submit_mode() {
  command -v sbatch >/dev/null 2>&1 || die "sbatch is required"
  require_uint "MAX_PARALLEL" "$MAX_PARALLEL"
  (( MAX_PARALLEL >= 1 )) || die "MAX_PARALLEL must be at least 1"
  verify_prepared
  mkdir -p "$LOG_ROOT"
  site_args

  echo "GPU_VALIDATION=deferred_to_slurm_compute_node"
  if [[ -n "$GPU_CHECK_HELPER" ]]; then
    echo "GPU_SELECTION_MODE=compute_node_helper:$GPU_CHECK_HELPER"
  else
    echo "GPU_SELECTION_MODE=slurm_allocation"
  fi

  local exported common pilot_raw pilot_id array_raw array_id aggregate_raw aggregate_id
  exported="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONDA_ROOT=$CONDA_ROOT,TORCH_HOME=$TORCH_HOME,WHEELHOUSE=$WHEELHOUSE,OFFLINE=$OFFLINE,REQUIRED_VRAM=$REQUIRED_VRAM,GPU_CHECK_HELPER=$GPU_CHECK_HELPER,LOCK_TIMEOUT=$LOCK_TIMEOUT,TRAIN_ENGINE=$TRAIN_ENGINE"
  common=(
    --parsable
    --export "$exported"
    --nodes 1
    --ntasks 1
    --cpus-per-task "$WORKER_CPUS"
    --mem "$WORKER_MEM"
    --time "$WORKER_TIME"
    --gres "$SLURM_GRES"
  )

  pilot_raw="$(sbatch ${SITE_ARGS[@]+"${SITE_ARGS[@]}"} "${common[@]}" \
    --job-name mmq1_a100_pilot \
    --output "$LOG_ROOT/mmq1_pilot_%j.out" \
    --error "$LOG_ROOT/mmq1_pilot_%j.err" \
    "$SELF" worker 0)"
  pilot_id="${pilot_raw%%;*}"
  [[ "$pilot_id" =~ ^[0-9]+$ ]] || die "invalid pilot job id: $pilot_raw"

  array_raw="$(sbatch ${SITE_ARGS[@]+"${SITE_ARGS[@]}"} "${common[@]}" \
    --dependency "afterok:$pilot_id" \
    --array "0-39%$MAX_PARALLEL" \
    --job-name mmq1_a100_train \
    --output "$LOG_ROOT/mmq1_train_%A_%a.out" \
    --error "$LOG_ROOT/mmq1_train_%A_%a.err" \
    "$SELF" worker)"
  array_id="${array_raw%%;*}"
  [[ "$array_id" =~ ^[0-9]+$ ]] || die "invalid array job id: $array_raw"

  aggregate_raw="$(sbatch ${SITE_ARGS[@]+"${SITE_ARGS[@]}"} --parsable \
    --export "$exported" \
    --dependency "afterok:$array_id" \
    --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 16G --time 02:00:00 \
    --job-name mmq1_a100_pack \
    --output "$LOG_ROOT/mmq1_aggregate_%j.out" \
    --error "$LOG_ROOT/mmq1_aggregate_%j.err" \
    "$SELF" aggregate)"
  aggregate_id="${aggregate_raw%%;*}"
  [[ "$aggregate_id" =~ ^[0-9]+$ ]] || die "invalid aggregate job id: $aggregate_raw"

  cat <<EOF
SLURM_SUBMISSION=PASS
PILOT_JOB_ID=$pilot_id
ARRAY_JOB_ID=$array_id
AGGREGATE_JOB_ID=$aggregate_id
DEPENDENCY_CHAIN=$pilot_id -> $array_id -> $aggregate_id
EOF
}

worker_mode() {
  local task_id="${1:-${SLURM_ARRAY_TASK_ID:-}}"
  [[ -n "$task_id" ]] || die "worker requires TASK_ID or SLURM_ARRAY_TASK_ID"
  exec bash "$TRAIN_ENGINE" worker "$task_id"
}

aggregate_mode() {
  exec bash "$TRAIN_ENGINE" aggregate
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [submit|worker [TASK_ID]|aggregate]

  submit     Verify prepared inputs, then submit pilot -> array -> aggregate.
  worker     Internal Slurm entry point for one model/seed task.
  aggregate  Internal Slurm entry point for statistics and packaging.
EOF
}

mode="${1:-submit}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  submit) submit_mode "$@" ;;
  worker) worker_mode "$@" ;;
  aggregate) aggregate_mode "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; die "unknown mode: $mode" ;;
esac
