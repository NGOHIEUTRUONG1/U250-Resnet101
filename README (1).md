# U250 ResNet-101 training, deployment, and benchmark artifact

This repository accompanies the paper *Measured Scaling of INT8 ResNet-101
Inference on Alveo U250*. It contains the paper source, the measured
CSV/JSON inputs, the scripts used to build the XModel and routed designs, and
the throughput harnesses used in the study.

The repository also contains a hash-verified source snapshot collected from
the U250 build host on 2026-09-29. It covers dataset provenance, training,
quantization, XModel compilation, the v6.1 cross-SLR builds, runtime tests, and
the environment records used for the paper. Start with `PIPELINE_SOURCE.md`.

The conference manuscript is limited to six pages. The canonical LaTeX entry
point is the non-empty `main.tex` at the repository root. The compiled paper is
`Measured_Scaling_INT8_ResNet101_U250.pdf`. Supporting figures, bibliography,
fonts, generated macros, and measured inputs are under `paper/`. See
`LATEX_BUILD.md` for Tectonic, TeX Live, and Overleaf instructions, and
`GITHUB_PUBLISH.md` for the publication checklist.

## What can be reproduced from this repository

| Level | Task | Included inputs | Status |
| --- | --- | --- | --- |
| R1 | Compile the paper | LaTeX, bibliography, fonts, figures | Self-contained |
| R2 | Regenerate result macros and figures | Summary CSV/JSON and analysis scripts | Self-contained |
| R3 | Repeat the U250 measurements | Benchmark source and expected artifact hashes | Requires the XModel, XCLBINs, CIFAR-100, Vitis AI 2.0 runtime, and exclusive access to a U250 |
| R4 | Rebuild the deployed artifacts | XModel and 1CU/2CU/3CU build scripts | Requires the historical checkpoint/configuration, Golden XO, licensed Vitis/Vivado 2021.2 installation, U250 platform, and Vitis-AI source checkout |

The XModel, XCLBINs, Golden XO, model checkpoint, and CIFAR-100 archive are not
redistributed here. Their expected digests are recorded in
`config/artifacts.sha256`. The repository therefore does not claim a
self-contained rebuild from training through bitstream generation.

## Benchmark methods

Four accelerator-side synthetic-input throughput checks are reported. They do
not include image decoding or preprocessing.

1. **Threaded `xdputil benchmark`.** One, two, or three software runners are
   requested for the matching 1CU, 2CU, or 3CU bitstream. Each result is based
   on ten independent 60-s runs. The full validation script also records device
   identity, programmed UUID, XRM state, accuracy, latency, telemetry, and file
   hashes.
2. **Static VART runner pool.** Each worker owns one long-lived runner and a
   private input/output buffer pair. A barrier releases the workers together;
   there is no application-level round-robin dispatcher. The 2CU and 3CU tests
   include a one-worker baseline on the same bitstream and an XRM active-CU
   snapshot during every repetition.
3. **1CU software-runner sensitivity.** On the unchanged 1CU artifact, five
   alternating 60-s repetitions compare one and two `xdputil` runners. The
   exploratory three- and four-runner cases are retained as diagnostics. After
   stale processes were removed and the old 120-s outer limit was raised, they
   completed but remained severely imbalanced, so they are not throughput
   points and are not used in the paper.
4. **Balanced runner saturation.** The 1CU, 2CU, and 3CU artifacts use 2, 4,
   and 6 runners, respectively. Each CPU-pinned setting has five 60-s runs.
   The fail-closed wrapper checks completion, frame totals, wall-clock
   continuity, and a minimum per-runner fairness ratio of 0.90. The paper uses
   all five accepted values, including the measured 3CU host-state variation.

Accuracy is a separate traversal of all 10,000 CIFAR-100 test images. Latency
uses 1,000 calls to one runner with the measured runtime batch of four.

## Repository layout

```text
main.tex                    canonical IEEE LaTeX source
IEEEtran.cls                bundled IEEE conference class
LATEX_BUILD.md              build and Overleaf instructions
GITHUB_PUBLISH.md           GitHub publication checklist
PIPELINE_SOURCE.md           end-to-end source and provenance map
benchmarks/full_test/       unified 1CU/2CU/3CU validation and xdputil campaign
benchmarks/static_vart/     static runner-pool harness and campaign wrapper
benchmarks/thread_sweep/    1CU software-runner sensitivity wrapper
benchmarks/runner_saturation/ balanced 2-runners-per-CU fail-closed wrapper
scripts/build/              XModel and routed 1CU/2CU/3CU build scripts
paper/                      figures, bibliography, fonts, inputs, and generators
results/                    compact machine-readable measurements
evidence/manifests/         recorded campaign status and original manifests
evidence/recorded_source/   generated source retained from the 2CU campaign
config/                     expected artifact and recorded-source digests
pipeline/source_snapshot/   exact source snapshot collected from the build host
```

The original campaign manifests are retained as provenance. They refer to
larger evidence bundles that are not fully present in this compact Git history
and must not be used as a release-completeness check. `MANIFEST_SHA256.txt` at
the repository root covers every file actually shipped in this release.

## Regenerate the paper

Python 3.10 or newer, the packages in `requirements-figures.txt`, and Tectonic
are sufficient for the self-contained paper build.

```bash
python3 -m pip install -r requirements-figures.txt
make macros
make figures
make paper
make verify
```

Without `make`, the corresponding commands are:

```bash
python3 paper/scripts/update_results.py \
  --one-cu paper/data/ieee_results_1cu_10runs.json \
  --two-cu paper/data/ieee_results_2cu_10runs.json \
  --three-cu paper/data/ieee_results_3cu_10runs.json \
  --sensor-one-cu paper/data/sensor_correction_1cu_10runs.json \
  --sensor-two-cu paper/data/sensor_correction_2cu_10runs.json \
  --sensor-three-cu paper/data/sensor_correction_3cu_10runs.json \
  --certification-two-cu paper/data/certification_2cu.txt \
  --certification-three-cu paper/data/certification_3cu.txt \
  --output paper/generated_results.tex

python3 paper/scripts/generate_static_results.py \
  paper/data/static_runner_validation_20260925.json \
  paper/static_runner_results.tex
python3 paper/scripts/summarize_runner_saturation.py \
  --one paper/data/runner_saturation_evidence/runner_saturation_1cu2_cpuset_5run_20260927/thread_sweep_runs.csv \
  --two paper/data/runner_saturation_evidence/runner_saturation_2cu4_cpuset_5run_20260927/thread_sweep_runs.csv \
  --three paper/data/runner_saturation_evidence/runner_saturation_3cu6_cpuset_5run_20260927/thread_sweep_runs.csv \
  --json paper/data/runner_saturation_matched_20260927.json \
  --tex paper/runner_saturation_results.tex
python3 paper/scripts/make_figures.py
tectonic -X compile main.tex --outdir build
python3 scripts/analysis/generate_release_manifest.py . MANIFEST_SHA256.txt
python3 scripts/analysis/verify_release_manifest.py MANIFEST_SHA256.txt
```

## Repeat method A: full validation and `xdputil`

The script programs the card, manages XRM state, and writes new evidence. Run it
only on an exclusively reserved U250. `EXPECTED_XCLBIN_SHA` must be selected
from `config/artifacts.sha256`.

```bash
export ROOT=/path/to/u250_workspace
export EXCLUSIVE_DEVICE=1
export CU_MODE=2                         # 1, 2, or 3
export XCLBIN=/exact/path/to/2cu.xclbin
export EXPECTED_XCLBIN_SHA=e44d1630e590abfdd73426c10beae54ede0eba4c19ca5020997ce261ee0c5489
export PERF_REPEATS=10
bash benchmarks/full_test/full_test_u250_ieee_v18_utf8_aggregation.sh
```

The released script is a portable 1CU/2CU/3CU consolidation of the recorded v18
launcher. The exact 2CU host launcher was not retained in the downloaded
evidence; its generated container suite is preserved as
`evidence/recorded_source/2cu_container_ieee_suite.sh`. This distinction is
intentional and prevents the portable launcher from being presented as an
unchanged historical file.

## Repeat method B: static VART pool

First run method A for the selected bitstream so that its
`fingerprint_env.tsv` is available. Clone or copy this repository below `ROOT`,
because the udocker wrapper maps files below that directory into `/workspace`.

```bash
export ROOT=/path/to/u250_workspace
export REPO=$ROOT/u250-benchmark-artifact
export EXCLUSIVE_DEVICE=1
export XMODEL=$ROOT/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel
export BENCH=$REPO/benchmarks/static_vart/vart_static_shard_benchmark.py

bash "$REPO/benchmarks/static_vart/run_static_runner_campaign.sh" \
  3 /exact/path/to/3cu.xclbin /exact/path/to/fingerprint_env.tsv \
  "$ROOT/09_results/static_runner_repeat" 1
```

The wrapper restarts `xrmd` and clears its shared state. Do not use it on a
shared card or while another XRM client is active.

## Repeat method C: 1CU software-runner sensitivity

The wrapper verifies the XCLBIN and XModel hashes, programs the exact 1CU
artifact, checks that the target render node is idle, and rejects runtime
errors, a missing `Test PASS`, or timeout before recording a CSV row.

```bash
export ROOT=/path/to/u250_workspace
export EXCLUSIVE_DEVICE=1
export THREAD_LIST="1 2"
export REPEATS=5
bash benchmarks/thread_sweep/run_xdputil_1cu_thread_sweep.sh
```

The original measured campaign and rejected exploratory 3/4-runner attempts
are preserved under `paper/data/thread_sweep_evidence/`. Additional
configuration trials are archived under
`paper/data/clock_sensitivity_evidence/` for provenance. They use a different
routed implementation and are not reported or interpreted in the paper.

## Build scripts and missing licensed artifacts

The scripts in `scripts/build/` are path-parameterized copies of the study
scripts. Set both `ROOT` and `VITIS_SETTINGS` before use. They retain the exact
tool, source-commit, platform, checkpoint, dataset, Golden-XO, clock, placement,
and timing gates used in the study. The original source-file hashes, before path
parameterization, are recorded in `config/recorded_source.sha256`.

## Integrity and release scope

```bash
python3 scripts/analysis/verify_release_manifest.py MANIFEST_SHA256.txt
```

The compact static-runner JSON contains all reported per-run FPS values,
confidence intervals, active-CU counts, and source hashes. The raw static
campaign logs and XRM snapshots were not recovered into this workspace. They
should be attached as a separate immutable archive if they become available.

No password, private key, host address, or access token is stored in this
repository. Device programming and XRM reset commands are guarded by the
explicit `EXCLUSIVE_DEVICE=1` acknowledgement.

## Third-party files

`paper/IEEEtran.cls` and the TeX Gyre Termes fonts are third-party components.
See `LICENSES/THIRD_PARTY.md` before redistribution. A project license for the
study scripts should be selected by the authors before making the repository
public.
