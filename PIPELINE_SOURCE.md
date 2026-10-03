# End-to-end source map

This document maps the released files to the experimental pipeline used by the
paper *Measured Scaling of INT8 ResNet-101 Inference on Alveo U250*.

`pipeline/source_snapshot/` is a read-only historical snapshot collected from
the U250 build host. Paths below are relative to that directory. The portable
wrappers under `scripts/` and `benchmarks/` remain the recommended entry points
for a new machine.

## 1. Environment and revision records

- `00_versions/resnet101_cpu100_versions.txt`: training host, Python, PyTorch,
  torchvision, Pillow, and NumPy versions.
- `00_versions/resnet101_cpu100_pip_freeze.txt`: recorded Python packages.
- `00_versions/vitis_ai_git_model_pipeline.txt`: Vitis-AI source revision.
- `REMOTE_PROVENANCE_20260929.txt`: host snapshot date, OS, source root, and
  collection policy.
- `REMOTE_SOURCE_SHA256_20260929.txt`: SHA-256 for every included remote file.
- `u250_paper_source_skipped_20260929.txt`: one unreadable generated build log;
  no source file was skipped.

The Vitis-AI source revision recorded by the host is
`d02dcb6041663dbc7ecbc0c6af9fafa087a789de` on the v2.0 line.

## 2. Dataset provenance

The study uses the official CIFAR-100 Python archive. The archive and extracted
images are not committed. Their expected MD5 is
`eb9058c3a382ffc7106e4002c42a8d85`.

Dataset loading and preprocessing are defined in:

- `04_model/source/resnet101_cpu100/train_resnet101_cifar100_cpu.py`;
- `train/train.py` and `train/train_config.json`;
- `xmodel_build/resnet101_cifar100_vai20_20260922_125524/quantize_resnet101.py`;
- the accuracy programs retained in the final 1CU/2CU/3CU result directories.

## 3. Training

The original CPU training source and the source bundled for deployment are:

- `04_model/source/resnet101_cpu100/train_resnet101_cifar100_cpu.py`;
- `train/train.py`;
- `train/train_config.json`;
- `train/training_metrics.csv`;
- `train/float_test_metrics.json`;
- `train/resnet101_cifar100_best.pt.sha256`;
- `train/STACK_CONTRACT.env` and `train/TRAIN_ONCE_LOCK.txt`.

The checkpoint itself is intentionally omitted. Its SHA-256 file preserves the
identity required to verify a separately transferred checkpoint.

## 4. Quantization and XModel compilation

The launcher used for the successful Vitis-AI 2.0 flow is:

- `train/build_resnet101_xmodel_vai20_u250_udocker_v5_1_70cpu_fixed.sh`.

The final successful run is preserved under
`xmodel_build/resnet101_cifar100_vai20_20260922_125524/`, including:

- `container_build.sh`;
- `quantize_resnet101.py`;
- calibration, quantized-test, export, compile, and verification logs;
- `quantize_result/quant_info.json` and generated `ResNet.py`;
- calibration/test metrics and final XModel SHA-256;
- the pinned quantizer manifest and environment fingerprints.

The generated XModel and intermediate model binaries are omitted. Their hashes
and verification records remain in the snapshot.

## 5. Routed DPU builds (v6.1)

Primary build scripts:

- `build_1cu_3cu_parallel_crossslr_f500_v6_1.sh`;
- `build_2cu_crossslr_f500_v6_1_derived.sh`.

The corresponding link configurations, Tcl hooks, logs, topology extracts, and
certification records are retained under:

- `05_dpu/crossslr_golden_f500_v6_1_20260922_052031/`;
- `05_dpu/crossslr_golden_f500_v6_1_2cu_20260923_195425/`;
- `09_results/crossslr_golden_f500_v6_1_20260922_052031/`;
- `09_results/crossslr_golden_f500_v6_1_2cu_20260923_195425/`.

XCLBIN, XO, DCP, cache, and full generated timing-report files are not suitable
for GitHub and are omitted. Compact reports, topology JSON, build contracts,
artifact hashes, final status, and routed-audit records are included.

One generated log,
`05_dpu/crossslr_golden_f500_v6_1_20260922_052031/3cu/v++_3cu_f500_crossslr.log`,
returned an I/O error on the source host and is listed explicitly in the skipped
file record. The 3CU build script, hook, link configuration, certification, and
artifact metadata are unaffected.

## 6. Runtime measurement

The full validation launchers are:

- `full_test_u250_ieee_v18_utf8_aggregation.sh`;
- `full_test_u250_ieee_v18_utf8_aggregation_2cu.sh`.

Independent and saturation methods are under `runtime/benchmark_methods/`:

- `vart_static_shard_benchmark.py`;
- `run_static_runner_campaign.sh`;
- `run_xdputil_1cu_thread_sweep.sh` and its corrected wrapper;
- `run_xdputil_runner_saturation.sh`.

Final runtime evidence is retained under the timestamped 1CU, 2CU, 3CU,
static-runner, and saturation result directories in `09_results/`. The compact
paper-facing copies are also available under `results/` and `paper/data/`.

## 7. Paper and figures

- `main.tex`: canonical six-page paper source.
- `paper/references.bib`: bibliography.
- `paper/figures/`: publication figures.
- `paper/scripts/`: scripts that regenerate paper macros and figures.
- `Measured_Scaling_INT8_ResNet101_U250.pdf`: verified compiled manuscript.

See `LATEX_BUILD.md` for Tectonic, TeX Live, and Overleaf instructions.

## 8. Exclusion policy

The GitHub package excludes datasets, checkpoint tensors, XModels, XCLBINs,
XO/DCP products, licensed tool installations, caches, and other large generated
binaries. These are not source code and several cannot be redistributed. The
release preserves their expected digests and provenance so privately transferred
artifacts can be checked before use.

No password, private key, access token, or private host address is included.
