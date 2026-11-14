# Vietnamese Image Captioning

Hệ thống sinh câu mô tả tiếng Việt cho ảnh bằng deep learning: đầu vào là một ảnh, đầu ra là
một câu mô tả tiếng Việt tự nhiên.

> **Trạng thái:** đã có pipeline dữ liệu và mô hình baseline (ResNet-50 + LSTM) chạy được từ đầu
> đến cuối; chưa huấn luyện đầy đủ. Xem [lộ trình](docs/ROADMAP.md) cho các bước tiếp theo.

## Cài đặt

Yêu cầu: Windows hoặc Linux, Python 3.10, GPU NVIDIA có driver hỗ trợ CUDA 13.0 (đã thử trên
RTX 5070 Laptop 8 GB). Chạy lần lượt từ thư mục gốc của repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_env.py
```

- Phải nâng cấp `pip` trước: bản `pip` cũ đi kèm Python 3.10 không cài được PyTorch từ kho này.
- PyTorch phải cài riêng bằng `--index-url` như trên để có bản CUDA; GPU dòng RTX 50 cần bản
  build với CUDA 12.8 trở lên. `requirements.txt` không thay thế bản đã cài.
- `check_env.py` in phiên bản PyTorch, CUDA, tên GPU và chạy thử một phép tính trên GPU; dòng
  cuối phải là `GPU check      : OK`.

## Dữ liệu

Dự án dùng **UIT-OpenViIC**: 13.100 ảnh đời sống ở Việt Nam với 61.241 caption do người viết
(train 9.088 ảnh, val 2.011 ảnh, test 2.001 ảnh).

- Nguồn: https://github.com/nghiangh/UIT-OpenViIC-dataset
- Bài báo: Bui, Nguyen, Nguyen (2026), https://doi.org/10.1016/j.image.2025.117430
- Giấy phép: bộ dữ liệu không có file giấy phép chính thức; tác giả tuyên bố bộ dữ liệu
  *"publicly available for research purposes and freely accessible"* và bài báo ghi
  *"non-commercial research"*. Dự án chỉ dùng cho nghiên cứu phi thương mại.

Dữ liệu được tải về `data/` (khoảng 3,8 GB) và **không bao giờ được đưa lên repo**. Chi tiết
khảo sát, số liệu kiểm tra và cách đổi sang dataset khác: [docs/DATASETS.md](docs/DATASETS.md).

## Chạy từ đầu đến cuối

Mọi lệnh chạy từ thư mục gốc của repo. Mọi tham số nằm trong
[`configs/baseline_lstm.yaml`](configs/baseline_lstm.yaml) và có thể ghi đè bằng `key=value` ở
cuối lệnh (ví dụ `train.batch_size=64`).

### 1. Chuẩn bị dữ liệu (làm một lần)

```powershell
python scripts/download_data.py --config configs/datasets/uit_openviic.yaml
python scripts/verify_data.py --config configs/datasets/uit_openviic.yaml
python scripts/prepare_data.py --config configs/baseline_lstm.yaml
python scripts/extract_features.py --config configs/baseline_lstm.yaml
```

| Bước | Việc làm | Kết quả |
|---|---|---|
| `download_data` | Tải và giải nén, chuyển annotation về định dạng thống nhất | `data/raw/`, `annotations.json` |
| `verify_data` | Giải mã mọi ảnh, thống kê caption | `verify_report.json` |
| `prepare_data` | Chuẩn hóa, tách từ tiếng Việt, xây từ điển (3.482 token) | `captions.json`, `vocab.json` |
| `extract_features` | Chạy ResNet-50 một lần, lưu đặc trưng float16 (khoảng 2,5 GB, 1 phút) | `data/features/` |

Tiền xử lý văn bản: Unicode NFC, chữ thường, bỏ dấu câu, thống nhất cách đặt dấu thanh (`hoà`
thành `hòa`), tách từ bằng `underthesea` (âm tiết cùng từ nối bằng `_`, ví dụ `cây_xanh`).

### 2. Huấn luyện

```powershell
python scripts/train.py --config configs/baseline_lstm.yaml
```

Bị ngắt giữa chừng (tắt máy, Ctrl+C) thì tiếp tục đúng chỗ dở bằng:

```powershell
python scripts/train.py --config configs/baseline_lstm.yaml --resume auto
```

- Kết quả nằm ở `outputs/baseline_lstm/`: `checkpoints/last.pt`, `checkpoints/best.pt` (CIDEr
  cao nhất trên val), `train.log`, `metrics.jsonl`, `summary.json`, `config.yaml`.
- Checkpoint được lưu mỗi 200 bước và cuối mỗi epoch. Huấn luyện tự dừng sớm nếu CIDEr trên
  val không tăng sau 5 epoch.
- Thư mục đã có checkpoint sẽ không bị ghi đè: dùng `--resume auto`, đặt tên khác bằng
  `run_name=...`, hoặc thêm `--overwrite` nếu thật sự muốn chạy lại từ đầu.
- Theo dõi bằng TensorBoard: `tensorboard --logdir outputs`.

### 3. Đánh giá trên tập test

```powershell
python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split test
```

In BLEU-1..4, CIDEr, METEOR và ghi điểm cùng toàn bộ caption sinh ra vào
`outputs/baseline_lstm/eval_test_beam3.json`. Thêm `--beam-size 1` để giải mã greedy.

**BLEU và CIDEr là chỉ số chính.** METEOR ở đây chỉ khớp các từ giống hệt nhau vì tiếng Việt
không có tài nguyên từ đồng nghĩa mà METEOR cần, nên chỉ để tham khảo. Điểm được tính trên âm
tiết; CIDEr theo thang gốc 0–10. Chi tiết: [docs/DESIGN.md](docs/DESIGN.md).

### 4. Sinh caption cho ảnh bất kỳ

```powershell
python scripts/predict.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --image duong\dan\anh.jpg
```

Có thể truyền nhiều ảnh sau `--image`; thêm `--beam-size 5` để đổi độ rộng beam, `--device cpu`
để chạy không cần GPU.

## Kết quả chạy thử

Chưa có kết quả huấn luyện đầy đủ. Số liệu dưới đây là từ một lần chạy thử 400 bước (hơn một
epoch) trên RTX 5070 Laptop, chỉ để xác nhận pipeline hoạt động:

| | |
|---|---|
| Loss huấn luyện | 8,01 (50 bước đầu) giảm còn 3,46 (bước 351–400) |
| Thời gian mỗi bước | khoảng 27 ms (batch 128, mixed precision, đặc trưng đã cache) |
| VRAM đỉnh | 385 MB cấp phát, 524 MB PyTorch giữ |
| Một epoch | 323 bước, khoảng 9 giây huấn luyện cộng 7 giây validate |
| Sau 1 epoch, val (greedy) | BLEU-4 20,26; CIDEr 0,2248 |

Ví dụ câu sinh ra sau một epoch: *"một cô gái mặc áo màu trắng đang đứng trước một quầy hàng"*.

## Cấu trúc thư mục

```
configs/    cấu hình thí nghiệm và cấu hình dataset (YAML)
data/       dữ liệu, đặc trưng đã cache (không đưa lên git)
docs/       tài liệu: lộ trình, thiết kế, dữ liệu, hướng dẫn phát triển
notebooks/  notebook khám phá
outputs/    kết quả huấn luyện, checkpoint (không đưa lên git)
scripts/    script chạy từ dòng lệnh
src/vic/    mã nguồn chính
tests/      test đơn vị
```

## Tài liệu

- [Lộ trình dự án](docs/ROADMAP.md): các giai đoạn đã xong và việc còn lại.
- [Thiết kế](docs/DESIGN.md): kiến trúc, lý do lựa chọn, phương án đã loại, rủi ro.
- [Dữ liệu](docs/DATASETS.md): khảo sát, giấy phép, cách tải, cách đổi dataset.
- [Hướng dẫn phát triển](docs/DEVELOPMENT.md): môi trường, quy tắc làm việc, lệnh thường dùng.

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE). Giấy phép này **chỉ áp dụng cho mã nguồn**.
Bộ dữ liệu tuân theo điều khoản riêng của tác giả bộ dữ liệu (xem mục Dữ liệu) và không nằm
trong repo này.
