#!/usr/bin/env bash
# Phase 1: prepare the exact A100/Vitis-AI 2.0 software contract, CIFAR-100,
# and all eight torchvision 0.9.0 ImageNet-pretrained artifacts.

set -Eeuo pipefail
umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENGINE="${TRAIN_ENGINE:-$SCRIPT_DIR/train_multimodel_cifar100_a100.sh}"

export PROJECT_ROOT="${PROJECT_ROOT:-/datastore/icc_fce/resnet_paper_multimodel}"
export CONDA_ROOT="${CONDA_ROOT:-/datastore/icc_fce/miniconda3}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export WHEELHOUSE="${WHEELHOUSE:-$PROJECT_ROOT/wheelhouse}"
export OFFLINE="${OFFLINE:-0}"
export LOCK_TIMEOUT="${LOCK_TIMEOUT:-600}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$TRAIN_ENGINE" ]] || die "training engine not found: $TRAIN_ENGINE"
[[ -x "$CONDA_ROOT/bin/conda" ]] || die "Conda not found: $CONDA_ROOT/bin/conda"

echo "PREPARE_PROJECT_ROOT=$PROJECT_ROOT"
echo "PREPARE_CONDA_ROOT=$CONDA_ROOT"
echo "PREPARE_OFFLINE=$OFFLINE"
echo "PREPARE_LOCK_TIMEOUT=$LOCK_TIMEOUT"
echo "PREPARE_ACTION=starting exact runtime and dataset preparation"

bash "$TRAIN_ENGINE" prepare-data

mkdir -p "$PROJECT_ROOT/provenance"
for source_file in \
  "$SCRIPT_DIR/prepare_data_env_a100.sh" \
  "$SCRIPT_DIR/train_multimodel_cifar100_a100.sh" \
  "$SCRIPT_DIR/job.sh"; do
  [[ -f "$source_file" ]] || die "campaign file missing: $source_file"
  cp -f "$source_file" "$PROJECT_ROOT/provenance/$(basename "$source_file")"
  sha256sum "$PROJECT_ROOT/provenance/$(basename "$source_file")" \
    >"$PROJECT_ROOT/provenance/$(basename "$source_file").sha256"
done

[[ -f "$PROJECT_ROOT/provenance/PREPARED.env" ]] \
  || die "PREPARED.env was not created"
[[ -f "$PROJECT_ROOT/provenance/preparation_manifest.sha256" ]] \
  || die "preparation manifest was not created"

(
  cd "$PROJECT_ROOT"
  sha256sum -c provenance/preparation_manifest.sha256
)

echo "DATA_ENV_PREPARATION=PASS"
echo "NEXT=bash $SCRIPT_DIR/job.sh submit"
