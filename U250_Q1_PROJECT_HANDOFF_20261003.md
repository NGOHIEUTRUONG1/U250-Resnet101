# Handoff kỹ thuật: ResNet-101 INT8 trên Alveo U250 và nền tảng cho project Q1

Ngày lập: 03-10-2026  
Máy U250: `192.168.26.26`  
Tài khoản: `truongnh`  
Workspace gốc trên máy U250: `/home/truongnh/u250_clean_2021_2`

> Tài liệu này tổng hợp các công việc đã thực hiện và đã kiểm chứng trên máy 26.
> Mật khẩu, private key và token không được ghi vào tài liệu. Các con số dưới
> đây lấy từ artifact, JSON/CSV, log và manifest đã sao chép về máy Windows,
> không phải số ước lượng.

## 1. Mục tiêu và trạng thái hiện tại

Pipeline đã hoàn tất cho một mô hình ResNet-101/CIFAR-100:

1. Huấn luyện mô hình float trên NVIDIA A100.
2. Kiểm tra checkpoint và khóa SHA-256.
3. Lượng tử hóa INT8 bằng Vitis-AI 2.0.
4. Biên dịch thành XModel cho `DPUCADF8H` trên Alveo U250.
5. Tạo thiết kế routed 1CU, 2CU và 3CU từ Golden XO.
6. Kiểm tra timing, clock, placement, URAM và topology của từng XCLBIN.
7. Chạy accuracy trên đủ 10.000 ảnh CIFAR-100.
8. Chạy benchmark theo hai phương pháp độc lập:
   - `xdputil benchmark` với số software runner khớp số CU;
   - static VART runner pool, mỗi worker giữ một runner và buffer riêng.
9. Chạy thêm runner-saturation với hai runner/CU để khảo sát giới hạn host.
10. Lưu raw trials, summary, telemetry, artifact hash và manifest.

Kết quả chính đã xác nhận:

| Cấu hình | `xdputil`, một runner/CU | Static VART, một worker/CU | Saturation, hai runner/CU |
| --- | ---: | ---: | ---: |
| 1CU | 162.230 ± 0.191 FPS, n=10 | 162.540 ± 0.136 FPS, n=10 | 188.442 ± 0.006 FPS, n=5 |
| 2CU | 324.688 ± 0.058 FPS, n=10 | 319.635 ± 0.760 FPS, n=10 | 376.830 ± 0.018 FPS, n=5 |
| 3CU | 487.325 ± 0.044 FPS, n=10 | 414.070 ± 4.377 FPS, n=10 | 534.146 ± 27.891 FPS, n=5 |

Các khoảng trên là 95% Student-t confidence interval. Với campaign n=10 dùng
`t(0.975,9)=2.262`; campaign n=5 dùng `t(0.975,4)=2.776`.

## 2. Vị trí source và bằng chứng đã sao chép về máy Windows

Gói nguồn đầy đủ hiện nằm tại:

```text
C:\Users\truongnh\Documents\ChatGPT\Baibao\output\
  U250_ResNet101_Full_Source_Clone_20260929\
```

Cấu trúc quan trọng:

```text
U250_ResNet101_Full_Source_Clone_20260929/
├── Vitis-AI/                         # working tree Vitis-AI đã sao chép
├── Vitis-AI_worktree_linux.tar       # bản Linux giữ symlink và executable bit
├── VITIS_AI_SOURCE_COMMIT.txt
├── VITIS_AI_WORKTREE_TAR_SHA256.txt
└── publication_repository/
    ├── README.md
    ├── PIPELINE_SOURCE.md
    ├── MANIFEST_SHA256.txt
    ├── config/
    ├── benchmarks/
    ├── scripts/
    ├── results/
    ├── evidence/
    ├── paper/
    └── pipeline/source_snapshot/     # snapshot từ máy 26, 533 file đã hash
```

Kiểm tra đã thực hiện:

- 533/533 file trong remote source snapshot khớp SHA-256.
- 929/929 file trong publication repository khớp manifest.
- Vitis-AI working tree tương ứng commit
  `d02dcb6041663dbc7ecbc0c6af9fafa087a789de`.
- SHA-256 của `Vitis-AI_worktree_linux.tar`:
  `800a1ca990637a9ff9334dab4fd1a1fdc9a25963f16f7a74d9b4b071564fc9f8`.

## 3. Phần cứng và toolchain trên máy 26

| Thành phần | Giá trị đã ghi nhận |
| --- | --- |
| FPGA board | AMD/Xilinx Alveo U250 |
| PCIe BDF | `0000:1b:00.0` |
| FPGA part | `xcu250-figd2104-2L-e` |
| OS | CentOS Linux 7, kernel `3.10.0-1160.119.1.el7.x86_64` |
| Vitis | 2021.2, SW build 3363252 |
| Vivado | 2021.2, SW build 3367213 |
| XRT | 2.12.427 |
| Platform | `xilinx_u250_gen3x16_xdma_3_1_202020_1` |
| DPU | `DPUCADF8H`, ISA target `DPUCADF8H_ISA0` |
| Vitis-AI | release/tag v2.0 |
| Vitis-AI commit | `d02dcb6041663dbc7ecbc0c6af9fafa087a789de` |
| Container image | `xilinx/vitis-ai-cpu:2.0.0.1103` |
| Container digest | `sha256:7a87fa94fd65710d3eab2713d4fb997fd7828ed627bedc71c8c0a1cea9eb4098` |
| Rootless runtime | udocker 1.3.17, P1/P2 depending on stage |

Thiết lập Vitis được source từ:

```bash
source /home/truongnh/local_vitis/Vitis/2021.2/settings64.sh
```

Biến platform khi build:

```bash
export PLATFORM_REPO_PATHS=/home/truongnh/u250_clean_2021_2/platform/tmp_deb/dev_extract/opt/xilinx/platforms
```

## 4. Hai lớp môi trường Python phải phân biệt

### 4.1. Môi trường huấn luyện checkpoint trên A100

Đây là môi trường sinh checkpoint float:

| Thư viện | Phiên bản |
| --- | --- |
| Python | 3.6.8 |
| PyTorch | 1.8.0+cu111 |
| torchvision | 0.9.0+cu111 |
| torchaudio | 0.8.0 |
| NumPy | 1.19.5 |
| Pillow | 8.4.0 |
| CUDA runtime | 11.1 |
| GPU | NVIDIA A100-SXM4-80GB |

Hợp đồng model:

```text
MODEL_IMPL=torchvision.models.resnet101(pretrained=False)
NUM_CLASSES=100
INPUT_SHAPE=1,3,224,224
CHECKPOINT_FORMAT=CPU_STATE_DICT
CHECKPOINT_LOAD_POLICY=STRICT_TRUE
TRAIN_POLICY=TRAIN_ONCE
```

Nguồn chuẩn:

```text
pipeline/source_snapshot/train/STACK_CONTRACT.env
pipeline/source_snapshot/train/train.py
pipeline/source_snapshot/train/train_config.json
pipeline/source_snapshot/train/training_metrics.csv
pipeline/source_snapshot/train/float_test_metrics.json
```

### 4.2. Môi trường lượng tử hóa/compile XModel thực tế thành công

Vitis-AI 2.0 container dùng:

| Thư viện | Phiên bản thực tế |
| --- | --- |
| Python | 3.6.15 |
| PyTorch | 1.7.1 |
| torchvision | 0.8.2 |
| NumPy | 1.17.5 |
| Pillow | 8.0.0 |
| Compiler | `/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/vai_c_xir` |
| Architecture | `/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json` |

Không thay các phiên bản trong container bằng bộ thư viện của môi trường A100.
Checkpoint được chuyển qua `CPU state_dict`, model được dựng lại với đúng topology,
sau đó nạp `strict=True`. Đây là ranh giới tương thích giữa hai môi trường.

Ngoài container, máy 26 từng có môi trường CPU kiểm tra với Python 3.8.5,
PyTorch 1.8.0+cpu, torchvision 0.9.0+cpu, NumPy 1.19.5 và Pillow 8.4.0.
Môi trường này dùng để audit/chuẩn bị input, không phải môi trường quantizer cuối.

## 5. Dataset, model và huấn luyện

Dataset: CIFAR-100 Python archive chính thức.

```text
MD5: eb9058c3a382ffc7106e4002c42a8d85
Train: 45.000 ảnh
Validation: 5.000 ảnh
Test chính thức: 10.000 ảnh
Split: stratified, mỗi lớp 450 train + 50 validation
```

Preprocessing/augmentation huấn luyện:

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip(p=0.5)
Resize(224, 224)
ToTensor()
Normalize(mean=[0.5071, 0.4867, 0.4408],
          std =[0.2675, 0.2565, 0.2761])
```

Hyperparameter đã dùng:

```text
epochs=100
batch_size=128
workers=8
seed=20260920
optimizer=SGD
base_lr=0.1
momentum=0.9
weight_decay=5e-4
schedule=5-epoch linear warm-up + cosine decay
selection=best validation Top-1
deterministic=true
AMP=false
```

Kết quả float:

```text
best_epoch=94
best_validation_top1=80.72%
official_test_top1=80.51%
official_test_top5=95.67%
checkpoint_sha256=331e2ea6720abf004154b1e58a74b675af2a20f306746ccfda91528533e3ddb3
```

## 6. Script tạo INT8 XModel

Script thành công cuối cùng:

```text
pipeline/source_snapshot/train/
  build_resnet101_xmodel_vai20_u250_udocker_v5_1_70cpu_fixed.sh
```

Script này thực hiện fail-closed theo thứ tự:

1. Kiểm tra file input, contract, checkpoint và dataset.
2. So checkpoint SHA-256 với giá trị cố định.
3. So CIFAR-100 MD5 với checksum chính thức.
4. Kiểm tra đúng Vitis-AI commit.
5. Kiểm tra/pull đúng OCI image digest.
6. Dựng udocker nếu host không có Docker daemon dùng được.
7. Gắn workspace vào container.
8. Cố định số luồng CPU cho MKL/OpenMP.
9. Chạy calibration.
10. Chạy INT8 test và chặn nếu Top-1 drop vượt 5 điểm phần trăm.
11. Export quantized XModel.
12. Compile bằng `vai_c_xir` cho kiến trúc U250/DPUCADF8H.
13. Mở lại XModel, xác minh DPU subgraph và tensor contract.
14. Ghi SHA-256/MD5 và chỉ publish khi mọi gate đều PASS.

Các biến có thể điều chỉnh nhưng mặc định đã dùng:

```bash
export CALIB_SAMPLES=200
export TEST_SAMPLES=1000
export CALIB_BATCH=32
export TEST_BATCH=32
export MAX_TOP1_DROP=0.05
export CPU_THREADS=70       # hoặc giá trị script suy ra từ host
export INTEROP_THREADS=1
export NUM_WORKERS=<giá trị được script giới hạn theo CPU>
```

Biến threading trong container:

```bash
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONNOUSERSITE=1
```

Lệnh tái chạy trên máy 26:

```bash
cd /home/truongnh/u250_clean_2021_2
bash train/build_resnet101_xmodel_vai20_u250_udocker_v5_1_70cpu_fixed.sh
```

Artifact thành công:

```text
/home/truongnh/u250_clean_2021_2/train/
  resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel
SHA-256:
  da6b56491525b3ccdf46f4aff5ba63ac4d8545ebf809d30fd024f7735c581007
```

Kết quả kiểm tra INT8 trên diagnostic subset 1.000 ảnh:

```text
float Top-1/Top-5 = 79.0% / 95.2%
INT8  Top-1/Top-5 = 78.6% / 94.1%
absolute Top-1 drop = 0.4 percentage point
```

## 7. Golden XO và script tạo 1CU/2CU/3CU XCLBIN

Golden XO:

```text
DPUCADF8H_U250_300MHz_to_500MHz_GOLDEN.xo
SHA-256: 20d4737661c04c276cba8e81f639d8ff598ea1e6c37a6f9335bf7edc0e48b142
```

Script chính:

```text
pipeline/source_snapshot/build_1cu_3cu_parallel_crossslr_f500_v6_1.sh
pipeline/source_snapshot/build_2cu_crossslr_f500_v6_1_derived.sh
```

Tên script đầu tiên phản ánh lịch sử phát triển. Bản v6.1 bảo toàn 1CU đã timing
closed và tạo/certify thiết kế 3CU. Script thứ hai suy ra từ cùng quy tắc để tạo
2CU. Không đổi Golden XO hoặc hook mà không tạo campaign mới.

Biến build quan trọng:

```bash
export ROOT=/home/truongnh/u250_clean_2021_2
export VITIS_SETTINGS=/home/truongnh/local_vitis/Vitis/2021.2/settings64.sh
export EXPECTED_VAI_COMMIT=d02dcb6041663dbc7ecbc0c6af9fafa087a789de
export GOLDEN_XO="$ROOT/09_results/golden_f500_xo_qualify_20260921_131323/GOLDEN_F500_XO/DPUCADF8H_U250_300MHz_to_500MHz_GOLDEN.xo"
export EXPECTED_GOLDEN_SHA=20d4737661c04c276cba8e81f639d8ff598ea1e6c37a6f9335bf7edc0e48b142
export CPUSET_1CU=0-34
export CPUSET_3CU=35-69
export VPP_JOBS=8
```

Đối với 2CU, kiểm tra giá trị `CPUSET_2CU` trong script và đặt theo CPU được
phép trên host trước khi chạy.

Topology đã dùng:

| CU | Logical SLR | DDR mapping | Routed URAM |
| --- | --- | --- | --- |
| CU1 | SLR0 | DDR0 | 312 ở SLR0 + 8 ở SLR1 |
| CU2 | SLR2 | DDR2 | 308 ở SLR2 + 12 ở SLR1 |
| CU3 | SLR3 | DDR3 | 320 ở SLR3 |

Các tập URAM giữa các CU không chồng lấn. Việc CU1/CU2 dùng một phần URAM ở
SLR1 là floorplan vật lý của vendor, không phải tràn tài nguyên runtime.

Clock/certification được script kiểm tra:

```text
external kernel/data clock = 300 MHz
MMCM input               = 300 MHz
MMCM D/M/O               = 1/5/3
MMCM output              = 500 MHz
derived ACLK             = 250 MHz
320 URAM mỗi CU
```

Timing routed:

| Thiết kế | WNS | WHS |
| --- | ---: | ---: |
| 1CU | +0.004 ns | +0.008 ns |
| 2CU | +0.003 ns | +0.010 ns |
| 3CU | +0.004 ns | +0.009 ns |

Lệnh build:

```bash
cd /home/truongnh/u250_clean_2021_2
bash build_1cu_3cu_parallel_crossslr_f500_v6_1.sh
bash build_2cu_crossslr_f500_v6_1_derived.sh
```

Không chạy hai build nặng đồng thời với benchmark. Build Vivado có thể chiếm
nhiều CPU/RAM và làm sai phép đo host/runtime.

XCLBIN đã dùng trong bài:

```text
1CU  156afe3e437241fa332ff817b5450c3069c460cf478e189c7eb01f12b0901928
2CU  e44d1630e590abfdd73426c10beae54ede0eba4c19ca5020997ce261ee0c5489
3CU  7efc8b0a5d60a1e49e339a52a233cf60a2db6e9a173c32de971fa06f3df91bba
```

## 8. Full validation trước benchmark

Script validation:

```text
pipeline/source_snapshot/full_test_u250_ieee_v18_utf8_aggregation.sh
pipeline/source_snapshot/full_test_u250_ieee_v18_utf8_aggregation_2cu.sh
```

Bản portable mới nằm ở:

```text
publication_repository/benchmarks/full_test/
  full_test_u250_ieee_v18_utf8_aggregation.sh
```

Các biến chính:

```bash
export ROOT=/home/truongnh/u250_clean_2021_2
export BDF=0000:1b:00.0
export CU_MODE=1                 # 1, 2 hoặc 3
export XCLBIN=/exact/path/to/design.xclbin
export EXPECTED_XCLBIN_SHA=<sha256 tương ứng>
export XMODEL="$ROOT/train/resnet101_cifar100_DPUCADF8H_U250_20260922_125524.xmodel"
export EXPECTED_XMODEL_SHA=da6b56491525b3ccdf46f4aff5ba63ac4d8545ebf809d30fd024f7735c581007
export EXPECTED_VAI_COMMIT=d02dcb6041663dbc7ecbc0c6af9fafa087a789de
export EXPECTED_CIFAR100_MD5=eb9058c3a382ffc7106e4002c42a8d85
export PERF_REPEATS=10
export BENCH_THREADS=$CU_MODE
export BENCH_TIMEOUT=120
export LAT_WARMUP=100
export LAT_ITERS=1000
export SENSOR_PERIOD=2
export EXCLUSIVE_DEVICE=1
```

Script thực hiện:

- kiểm tra hash của XCLBIN/XModel/dataset và Vitis-AI commit;
- ghi environment và device state trước test;
- program đúng BDF;
- xác nhận UUID và số physical CU sau program;
- khởi động XRM sạch, load đúng XCLBIN;
- kiểm tra từng CU mong đợi có trong XRM database;
- chạy full hardware accuracy 10.000 ảnh;
- đo latency 100 warm-up + 1.000 iteration;
- chạy throughput lặp lại;
- lấy telemetry trong đúng khoảng benchmark active;
- sinh JSON, CSV, Markdown report, `FINAL_STATUS.txt` và manifest.

Ví dụ 2CU:

```bash
cd /home/truongnh/u250_clean_2021_2
export ROOT=$PWD
export EXCLUSIVE_DEVICE=1
export CU_MODE=2
export XCLBIN=$ROOT/05_dpu/crossslr_golden_f500_v6_1_2cu_20260923_195425/2cu/2cu_f500_crossslr.xclbin
export EXPECTED_XCLBIN_SHA=e44d1630e590abfdd73426c10beae54ede0eba4c19ca5020997ce261ee0c5489
export PERF_REPEATS=10
bash full_test_u250_ieee_v18_utf8_aggregation.sh
```

## 9. Phương pháp benchmark A: `xdputil`, một runner/CU

Đây là phương pháp tạo kết quả scaling gần tuyến tính 1CU→3CU.

Protocol:

```text
synthetic input, accelerator-side
60 giây mỗi repetition
10 repetitions
software runners = physical CU count
1CU/2CU/3CU dùng 1/2/3 runners
không tính decode và preprocessing vào throughput
```

Kết quả:

| CU | Mean FPS | 95% CI | SD | CV | Scaling từ 1CU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 162.230 | ±0.191 | 0.267 | 0.164% | 1.000× |
| 2 | 324.688 | ±0.058 | 0.081 | 0.025% | 2.001× |
| 3 | 487.325 | ±0.044 | 0.062 | 0.013% | 3.004× |

Accuracy phần cứng giống nhau trên cả ba XCLBIN:

```text
Top-1: 78.97% = 7.897/10.000
Top-5: 94.95% = 9.495/10.000
runtime batch: 4
```

Latency một runner, batch 4:

| CU bitstream | Mean batch latency | Effective mean/image |
| --- | ---: | ---: |
| 1CU | 24.711 ms | 6.178 ms/image |
| 2CU | 24.710 ms | 6.177 ms/image |
| 3CU | 24.734 ms | 6.183 ms/image |

Power đã lọc lại theo đúng timestamp active và đúng label `Power/FPGA`:

| CU | Mean board power | FPS/W |
| ---: | ---: | ---: |
| 1 | 49.200 W | 3.297 img/J |
| 2 | 73.584 W | 4.412 img/J |
| 3 | 98.659 W | 4.939 img/J |

Không dùng trực tiếp trường power cũ trong `ieee_results_*.json`, vì parser ban
đầu đã trộn các dòng sensor không cùng ý nghĩa. Dùng file
`results/xdputil/sensor_correction_*_10runs.json`.

## 10. Phương pháp benchmark B: static VART runner pool

Source:

```text
publication_repository/benchmarks/static_vart/vart_static_shard_benchmark.py
publication_repository/benchmarks/static_vart/run_static_runner_campaign.sh
```

Đây không phải round-robin:

- mỗi worker tạo và giữ một VART runner riêng;
- mỗi worker có input/output buffers riêng;
- warm-up 20 call/worker;
- barrier giải phóng workers cùng lúc;
- mỗi worker submit trực tiếp vào runner của nó;
- chụp XRM active-CU trong từng repetition;
- 60 giây/run, 10 run.

Kết quả:

| Bitstream/cấu hình | Mean FPS | 95% CI | Speedup nội bộ | Hiệu suất lý tưởng |
| --- | ---: | ---: | ---: | ---: |
| 1CU, 1 worker | 162.540 | ±0.136 | 1.000× | 100% |
| 2CU, 1 worker | 162.636 | ±0.262 | baseline | — |
| 2CU, 2 workers | 319.635 | ±0.760 | 1.965× | 98.27% |
| 3CU, 1 worker | 162.327 | ±0.095 | baseline | — |
| 3CU, 3 workers | 414.070 | ±4.377 | 2.551× | 85.03% |

Điểm quan trọng cho project Q1: `xdputil` và static VART không tạo cùng một
loading pattern. Kết quả static VART cho thấy khi host điều khiển nhiều runner
dài hạn đồng thời, 3CU không còn scaling lý tưởng. Đây là bằng chứng host/runtime
có thể trở thành nút cổ chai; không được gộp hai phương pháp thành một series.

Ví dụ chạy 3CU:

```bash
export ROOT=/home/truongnh/u250_clean_2021_2
export EXCLUSIVE_DEVICE=1

bash "$ROOT/runtime/benchmark_methods/run_static_runner_campaign.sh" \
  3 \
  /exact/path/to/3cu.xclbin \
  /exact/path/to/fingerprint_env.tsv \
  "$ROOT/09_results/static_runner_repeat" \
  1
```

Wrapper reset `xrmd` và `/dev/shm/xrm.data`; chỉ chạy khi card được cấp độc quyền.

## 11. Test bổ sung: hai runner/CU và fairness gate

Source:

```text
publication_repository/benchmarks/runner_saturation/
  run_xdputil_runner_saturation.sh
```

Protocol:

```text
1CU -> 2 runners
2CU -> 4 runners
3CU -> 6 runners
5 runs x 60 s
CPU pinned
reject timeout/incomplete run
reject runner starvation
minimum fairness ratio used in accepted campaign: 0.90
```

Kết quả chấp nhận:

| CU | Runners | Mean FPS | 95% CI | Min–max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 188.442 | ±0.006 | 188.437–188.451 |
| 2 | 4 | 376.830 | ±0.018 | 376.805–376.842 |
| 3 | 6 | 534.146 | ±27.891 | 518.127–565.220 |

Hai runner/CU cải thiện throughput tuyệt đối, nhưng 3CU chỉ đạt 2.835× so với
1CU, tức 94.5% của ideal 3×. Dải 3CU rộng cho thấy trạng thái host ảnh hưởng
mạnh hơn khi concurrency tăng. Không suy diễn nguyên nhân nếu chưa có thêm
CPU/per-core utilization, memory bandwidth, scheduler trace và PCIe trace.

Ví dụ:

```bash
export ROOT=/home/truongnh/u250_clean_2021_2
export EXCLUSIVE_DEVICE=1
export BDF=0000:1b:00.0
export EXPECTED_CU_COUNT=3
export THREAD_LIST="6"
export REPEATS=5
export BENCH_TIMEOUT=180
export CPUSET=0-15
export MIN_FAIRNESS_RATIO=0.90
export XCLBIN=/exact/path/to/3cu.xclbin
export EXPECTED_XCLBIN_SHA=7efc8b0a5d60a1e49e339a52a233cf60a2db6e9a173c32de971fa06f3df91bba

bash "$ROOT/runtime/benchmark_methods/run_xdputil_runner_saturation.sh"
```

Các lần 3/4 runner thử nghiệm trước đó từng vượt wrapper timeout 120 s. Sau khi
dọn stale process và tăng outer timeout, chúng chạy xong nhưng mất cân bằng nặng.
Do không qua fairness gate, không dùng chúng làm điểm throughput.

## 12. Quy tắc an toàn khi chạy lại benchmark

1. Chỉ chạy khi được độc quyền U250; đặt `EXCLUSIVE_DEVICE=1`.
2. Kiểm tra không còn tiến trình `xdputil benchmark`, VART benchmark hoặc Vivado build.
3. Kiểm tra đúng BDF và device index; không mặc định card luôn là device 0.
4. So SHA-256 XCLBIN, XModel và MD5 dataset trước khi program.
5. Ghi `xbutil examine` trước và sau program.
6. Reset XRM có kiểm soát; không xóa shared state khi có user khác đang dùng card.
7. Xác nhận physical CU count bằng `xbutil`, không suy ra từ tên file.
8. Xác nhận tất cả expected CU xuất hiện trong XRM database.
9. Pin CPU cho các campaign so sánh concurrency cao.
10. Chạy warm-up trước cửa sổ đo.
11. Lưu từng raw run; không chỉ lưu mean.
12. Đo wall-clock thực, frame count, completion marker và fairness.
13. Telemetry phải có timestamp và chỉ lọc trong khoảng benchmark active.
14. Không chạy compiler đồng thời trừ khi đó là campaign shared-load được định nghĩa trước.
15. Mọi kết quả mới phải tạo thư mục timestamp mới, không ghi đè evidence cũ.

Preflight tối thiểu:

```bash
pgrep -af 'vart_static_shard_benchmark|xdputil.*benchmark|v\+\+|vivado'
/opt/xilinx/xrt/bin/xbutil examine --device 0000:1b:00.0
sha256sum "$XCLBIN" "$XMODEL"
md5sum "$DATASET_ARCHIVE"
```

## 13. Những lỗi đã gặp và cách xử lý

### 13.1. Metadata parser tạo false negative

Một số `FINAL_STATUS.txt` lịch sử của build 2CU/3CU ghi FAIL do parser so tên
clock/pblock không đúng, trong khi routed audit, timing report, runtime program,
CU count và full validation đều hợp lệ. Đây là lỗi của bước kiểm tra metadata,
không phải lỗi DPU. Bản công bố dùng certification đã kiểm lại và các manifest
runtime PASS.

Khi làm project mới, parser phải:

- parse JSON section của XCLBIN thay vì so chuỗi mong manh;
- phân biệt logical pblock với physical SLR distribution;
- không coi số entry toàn bộ XRM database là số physical CU;
- xác nhận UUID, IP layout, clock topology và CU names độc lập.

### 13.2. Timeout wrapper không đồng nghĩa CU timeout

Outer wrapper 120 s từng kết thúc phép thử concurrency cao. Chỉ được gọi là
hardware/CU timeout khi log runtime hoặc driver chứng minh điều đó. Project mới
cần ghi riêng:

```text
wrapper_timeout
subprocess_exit_code
runtime_error
driver_error
completion_marker
```

### 13.3. Power parser ban đầu trộn sensor fields

Phải chọn đúng label `Power` và `FPGA`, đồng thời chỉ lấy sample nằm trong
cửa sổ benchmark. Không dùng số `225 W` lặp lại từ parser cũ làm measured mean.

### 13.4. SCP/Git qua SSH bị shell startup text phá giao thức

`~/.bashrc` trên máy 26 từng gọi `/etc/bashrc` trong non-interactive shell và
in `XILINX_XRT` vào stdout. Đã thêm guard:

```bash
case $- in *i*) ;; *) return ;; esac
```

Bản cũ được sao lưu thành:

```text
~/.bashrc.before_noninteractive_guard_20260929
```

Không in bất kỳ nội dung nào trong non-interactive shell nếu dùng SCP/SFTP/Git.

### 13.5. Clone đầy đủ lịch sử Git làm máy 26 hết tài nguyên

`git pack-objects` từng bị signal 7. Working tree Vitis-AI đã được truyền bằng
TAR và giữ commit provenance. Khi cần clone lại, nên clone từ upstream theo
commit đã ghi nhận hoặc tạo bundle trên host có đủ RAM/swap; không ép máy 26
pack toàn bộ lịch sử trong lúc benchmark.

## 14. Nguồn dữ liệu kết quả

Kết quả `xdputil`:

```text
publication_repository/results/xdputil/
├── ieee_results_1cu_10runs.json
├── ieee_results_2cu_10runs.json
├── ieee_results_3cu_10runs.json
├── benchmark_runs_1cu_10.csv
├── benchmark_runs_2cu_10.csv
├── benchmark_runs_3cu_10.csv
├── latency_samples_1cu_10.csv
├── latency_samples_2cu_10.csv
├── latency_samples_3cu_10.csv
├── sensor_correction_1cu_10runs.json
├── sensor_correction_2cu_10runs.json
└── sensor_correction_3cu_10runs.json
```

Kết quả static VART:

```text
publication_repository/results/static_vart/
  static_runner_validation_20260925.json
```

Kết quả saturation và raw trials:

```text
publication_repository/pipeline/source_snapshot/09_results/
├── runner_saturation_1cu2_cpuset_5run_20260927/
├── runner_saturation_2cu4_cpuset_5run_20260927/
└── runner_saturation_3cu6_cpuset_5run_20260927/
```

Build evidence:

```text
publication_repository/pipeline/source_snapshot/09_results/
├── crossslr_golden_f500_v6_1_20260922_052031/
└── crossslr_golden_f500_v6_1_2cu_20260923_195425/
```

## 15. Khuyến nghị kiến trúc project Q1 mới

Không ghi kết quả đa mô hình vào thư mục ResNet-101 cũ. Tạo project mới, ví dụ:

```text
u250_q1_multimodel/
├── README.md
├── config/
│   ├── environment.lock
│   ├── datasets.yaml
│   ├── models.yaml
│   └── hardware.yaml
├── data/
│   └── manifests/
├── train/
│   ├── prepare_data_env_a100.sh
│   ├── train_multimodel.py
│   └── job.sh
├── quantize/
│   ├── adapters/
│   └── build_xmodels_vai20.sh
├── hardware/
│   ├── build_1cu_2cu_3cu.sh
│   └── constraints/
├── benchmark/
│   ├── full_validation.sh
│   ├── xdputil_matched_cu.sh
│   ├── static_vart_pool.py
│   ├── runner_saturation.sh
│   └── host_telemetry.sh
├── results/
│   ├── raw/
│   ├── summaries/
│   └── manifests/
└── paper/
```

Mỗi model/seed cần một immutable run ID và các artifact sau:

```text
config.json
environment.json
best_state_dict.pt
best_state_dict.pt.sha256
float_test_metrics.json
quant_test_metrics.json
compiled.xmodel.sha256
xclbin.sha256
accuracy_10000.json
latency_samples.csv
throughput_runs.csv
telemetry.csv
FINAL_STATUS.txt
MANIFEST_SHA256.txt
```

## 16. Thiết kế thí nghiệm phù hợp bài Q1

Để vượt khỏi một case study ResNet-101, nên triển khai ít nhất:

- nhiều họ mạng: ResNet, MobileNet, DenseNet, VGG, ShuffleNet, EfficientNet;
- ít nhất 5 seed/model cho training variation;
- cùng dataset split và preprocessing contract;
- cùng accuracy gate trước khi benchmark;
- 1CU/2CU/3CU cho mỗi model tương thích DPU;
- hai phương pháp đo độc lập như hiện tại;
- matched runners/CU và saturation runners/CU;
- CPU pinning, NUMA state, host utilization và memory bandwidth;
- board power/temperature được timestamp chính xác;
- cold-start, warm steady-state và sustained-duration tests;
- raw per-run values, Student-t CI và outlier policy định nghĩa trước;
- cross-model comparison theo FPS, latency, FPS/W, accuracy loss và resource use.

Không nên gọi scaling tuyến tính chỉ từ `xdputil` matched-CU. Câu hỏi nghiên cứu
mạnh hơn cho Q1 là: khi nào scaling multi-CU bị giới hạn bởi accelerator, runtime,
host scheduling, memory subsystem hoặc concurrency policy; và giới hạn đó thay
đổi thế nào giữa các topology mạng.

## 17. Checklist bắt đầu project mới

- [ ] Tạo repository mới, không ghi đè workspace ResNet-101.
- [ ] Chụp environment và hardware inventory ngay từ ngày đầu.
- [ ] Khóa dataset checksum và split indices.
- [ ] Khóa model factory, input shape và classifier contract.
- [ ] Chạy pilot một model/một seed trước job array.
- [ ] Kiểm tra checkpoint load `strict=True` trên cả training và Vitis-AI stack.
- [ ] Tạo adapter riêng cho model không tương thích torchvision/Vitis-AI 2.0.
- [ ] Quantization accuracy gate trước compile.
- [ ] XModel DPU-subgraph gate trước U250 test.
- [ ] XCLBIN hash/UUID/CU-count gate trước benchmark.
- [ ] Chạy đủ hai benchmark methods.
- [ ] Chạy ít nhất 10 repetitions cho số liệu chính.
- [ ] Pin CPU và ghi NUMA/CPU utilization ở concurrency cao.
- [ ] Lưu raw logs, không chỉ summary.
- [ ] Sinh manifest sau cùng và xác minh lại từ thư mục release.

## 18. Tệp nên đọc đầu tiên khi tiếp tục

Theo thứ tự:

1. `publication_repository/README.md`
2. `publication_repository/PIPELINE_SOURCE.md`
3. `publication_repository/config/artifacts.sha256`
4. `publication_repository/pipeline/source_snapshot/train/STACK_CONTRACT.env`
5. `publication_repository/pipeline/source_snapshot/train/build_resnet101_xmodel_vai20_u250_udocker_v5_1_70cpu_fixed.sh`
6. `publication_repository/pipeline/source_snapshot/build_1cu_3cu_parallel_crossslr_f500_v6_1.sh`
7. `publication_repository/pipeline/source_snapshot/build_2cu_crossslr_f500_v6_1_derived.sh`
8. `publication_repository/benchmarks/full_test/full_test_u250_ieee_v18_utf8_aggregation.sh`
9. `publication_repository/benchmarks/static_vart/run_static_runner_campaign.sh`
10. `publication_repository/benchmarks/static_vart/vart_static_shard_benchmark.py`
11. `publication_repository/benchmarks/runner_saturation/run_xdputil_runner_saturation.sh`

## 19. Kết luận handoff

Baseline ResNet-101 hiện đủ để dùng làm điểm xuất phát: artifact được khóa hash,
1CU/2CU/3CU đã chạy full validation, accuracy nhất quán, timing routed dương và
có hai phương pháp benchmark cho thấy hai chế độ scaling khác nhau. Project Q1
nên giữ nguyên baseline này, mở rộng sang nhiều model và tập trung vào cơ chế
host/runtime/memory làm thay đổi scaling, thay vì chỉ lặp lại phép đo FPS.

