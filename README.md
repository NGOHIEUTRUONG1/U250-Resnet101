# U250 Q1 reproducible pipeline bundle

Gói này tích hợp các script cuối cùng của baseline ResNet-101/U250 và bộ khung
training đa mô hình trên A100 cho project Q1. Không chứa mật khẩu, dataset,
checkpoint, Golden XO, XModel hay XCLBIN; checksum của các artifact bắt buộc nằm
trong `07_checksums_and_results/artifacts.sha256`.

## Chọn đúng script

### Tạo môi trường và train đa mô hình trên A100/Slurm

Chạy theo thứ tự:

```bash
bash 01_q1_a100_training/prepare_data_env_a100.sh
bash 01_q1_a100_training/job.sh submit
```

`job.sh` gọi `train_multimodel_cifar100_a100.sh`. Kiểm tra lại partition,
account, `PROJECT_ROOT`, GPU resource và đường dẫn Conda theo cluster mới trước
khi submit.

### Baseline ResNet-101 đã dùng trong nghiên cứu U250

Nguồn train và hợp đồng phiên bản nằm trong
`02_resnet101_training_recorded/`. Đây là bản ghi lịch sử đã tạo checkpoint
baseline; không trộn checkpoint này với campaign đa mô hình mới.

### Tạo INT8 XModel

- Canonical đã chạy trên máy 26:
  `03_xmodel_build/recorded_build_resnet101_xmodel_vai20_u250_udocker_v5_1_70cpu_fixed.sh`
- Bản đổi đường dẫn để tái sử dụng:
  `03_xmodel_build/portable_build_resnet101_xmodel_vai20_u250.sh`

Nếu cần tái lập đúng baseline, ưu tiên bản `recorded_*` trên đúng cấu trúc
`/home/truongnh/u250_clean_2021_2`. Với project mới, dùng bản portable sau khi
đặt đúng `ROOT` và kiểm tra lại toàn bộ hash/version gate.

### Tạo XCLBIN 1CU và 3CU

- Canonical máy 26:
  `04_xclbin_build/recorded_machine26/build_1cu_3cu_parallel_crossslr_f500_v6_1.sh`
- Portable:
  `04_xclbin_build/portable/build_1cu_3cu_crossslr_f500.sh`

Script canonical bảo toàn 1CU timing-closed đã có, đồng thời dựng/certify 3CU từ
Golden XO. Vì vậy tên “build 1CU+3CU” không có nghĩa mọi lần chạy đều route lại
1CU từ đầu.

### Tạo XCLBIN 2CU

- Canonical máy 26:
  `04_xclbin_build/recorded_machine26/build_2cu_crossslr_f500_v6_1_derived.sh`
- Portable:
  `04_xclbin_build/portable/build_2cu_crossslr_f500.sh`

### Benchmark

- Full validation + `xdputil`: `05_benchmark/portable/full_test_u250_ieee_v18_utf8_aggregation.sh`
- Static VART pool: `05_benchmark/portable/run_static_runner_campaign.sh`
- VART worker: `05_benchmark/portable/vart_static_shard_benchmark.py`
- Hai runner/CU và fairness gate: `05_benchmark/portable/run_xdputil_runner_saturation.sh`
- 1CU runner sensitivity: `05_benchmark/portable/run_xdputil_1cu_thread_sweep.sh`

Thư mục `recorded_machine26/` giữ nguyên các file lịch sử đã chạy. Thư mục
`portable/` dùng cho lần chạy mới.

## Tài liệu và môi trường

- `00_documentation/U250_Q1_PROJECT_HANDOFF_20261003.md`: handoff chi tiết.
- `00_documentation/PIPELINE_SOURCE.md`: ánh xạ source của pipeline cũ.
- `06_environment/`: version, pip freeze, provenance và môi trường XModel.
- `07_checksums_and_results/`: checksum artifact và summary đo đã xác nhận.

## Điều kiện bắt buộc trước khi chạy trên U250

1. Dùng quyền truy cập độc quyền card và đặt `EXCLUSIVE_DEVICE=1`.
2. Xác minh BDF `0000:1b:00.0` vẫn đúng trên host hiện tại.
3. Không chạy Vivado/v++ song song với benchmark.
4. So SHA-256 XModel/XCLBIN và MD5 CIFAR-100 trước khi program card.
5. Không thay phiên bản Vitis/Vivado/XRT/Vitis-AI trong baseline.
6. Không dùng kết quả nếu runtime không hoàn tất hoặc fairness gate thất bại.
7. Ghi kết quả vào thư mục timestamp mới; không ghi đè evidence cũ.

## Kiểm tra gói

```bash
sha256sum -c MANIFEST_SHA256.txt
```

Các script shell trong gói đã được kiểm tra cú pháp bằng `bash -n`. Kiểm tra cú
pháp không thay thế preflight artifact, toolchain và phần cứng.

