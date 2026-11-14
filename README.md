# Vietnamese Image Captioning

Hệ thống sinh câu mô tả tiếng Việt cho ảnh bằng deep learning: đầu vào là một ảnh, đầu ra là
một câu mô tả tiếng Việt tự nhiên.

> **Trạng thái:** đã có pipeline dữ liệu và mô hình baseline (ResNet-50 + LSTM) huấn luyện đầy
> đủ; kết quả ở [docs/RESULTS.md](docs/RESULTS.md). Các mô hình mạnh hơn nằm trong
> [lộ trình](docs/ROADMAP.md).

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
- Giấy phép: bộ dữ liệu không có file giấy phép chính thức. Căn cứ sử dụng là tuyên bố của
  tác giả trong README: *"publicly available for research purposes and freely accessible"*.

**Điều kiện sử dụng trong dự án này:**

- Bộ dữ liệu **chỉ được dùng cho nghiên cứu phi thương mại**.
- Repo này **không phân phối dữ liệu**: ảnh và caption do người dùng tự tải từ nguồn của tác
  giả về thư mục `data/` (khoảng 3,8 GB), thư mục này không bao giờ được đưa lên repo.
- Repo này **không phân phối checkpoint**: mô hình đã huấn luyện không được tải lên GitHub hay
  bất kỳ nơi công khai nào. Muốn có mô hình, hãy tự huấn luyện bằng các lệnh bên dưới.

Chi tiết khảo sát, số liệu kiểm tra và cách đổi sang dataset khác:
[docs/DATASETS.md](docs/DATASETS.md).

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
`outputs/baseline_lstm/eval_test_beam3.json`. Thêm `--beam-size 1` để giải mã greedy. Có thể
đánh giá nhiều tập và nhiều độ rộng beam trong một lệnh, rồi tổng hợp thành báo cáo Markdown
(đường cong học, bảng điểm, ví dụ ngẫu nhiên theo seed cố định):

```powershell
python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split val test --beam-size 1 3 5
python scripts/summarize_run.py --run outputs/baseline_lstm --output outputs/baseline_lstm/report.md
```

**BLEU và CIDEr là chỉ số chính.** METEOR ở đây chỉ khớp các từ giống hệt nhau vì tiếng Việt
không có tài nguyên từ đồng nghĩa mà METEOR cần, nên chỉ để tham khảo. Điểm được tính trên âm
tiết; CIDEr theo thang gốc 0–10. Chi tiết: [docs/DESIGN.md](docs/DESIGN.md).

### 4. Sinh caption cho ảnh bất kỳ

```powershell
python scripts/predict.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --image duong\dan\anh.jpg
```

Có thể truyền nhiều ảnh sau `--image`; thêm `--beam-size 5` để đổi độ rộng beam, `--device cpu`
để chạy không cần GPU.

## Kết quả

Baseline ResNet-50 đóng băng + LSTM, huấn luyện 30 epoch (khoảng 8 phút trên RTX 5070 Laptop,
VRAM đỉnh 385 MB), checkpoint tốt nhất ở epoch 28, giải mã beam 3:

| Tập | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | METEOR (tham khảo) |
|---|---:|---:|---:|---:|---:|---:|
| val | 68,06 | 51,43 | 38,54 | 28,74 | 0,4274 | 22,10 |
| test | 66,26 | 50,36 | 37,65 | 27,91 | 0,4479 | 21,76 |

BLEU và METEOR nhân 100, CIDEr theo thang gốc 0–10, tính trên âm tiết. Baseline còn yếu: câu
sinh ra thường chung chung, từ vựng nghèo và hay bịa ra người trong ảnh không có người. Bảng
đầy đủ (greedy, beam 3, beam 5), đường cong học, 10 ví dụ trên tập test và phân tích lỗi:
[docs/RESULTS.md](docs/RESULTS.md).

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

- [Kết quả](docs/RESULTS.md): điểm số, đường cong học, ví dụ và phân tích lỗi của từng mô hình.
- [Lộ trình dự án](docs/ROADMAP.md): các giai đoạn đã xong và việc còn lại.
- [Thiết kế](docs/DESIGN.md): kiến trúc, lý do lựa chọn, phương án đã loại, rủi ro.
- [Dữ liệu](docs/DATASETS.md): khảo sát, giấy phép, cách tải, cách đổi dataset.
- [Hướng dẫn phát triển](docs/DEVELOPMENT.md): môi trường, quy tắc làm việc, lệnh thường dùng.

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE). Giấy phép này **chỉ áp dụng cho mã nguồn**.
Bộ dữ liệu tuân theo điều khoản riêng của tác giả bộ dữ liệu, chỉ dùng cho nghiên cứu phi
thương mại (xem mục Dữ liệu). Repo không chứa và không phân phối dữ liệu hay checkpoint.
