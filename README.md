# Vietnamese Image Captioning

Hệ thống sinh câu mô tả tiếng Việt cho ảnh bằng deep learning: đầu vào là một ảnh, đầu ra là
một câu mô tả tiếng Việt tự nhiên.

> **Trạng thái:** đang phát triển. Xem [lộ trình](docs/ROADMAP.md) để biết giai đoạn hiện tại.

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

```powershell
python scripts/download_data.py --config configs/datasets/uit_openviic.yaml
python scripts/verify_data.py --config configs/datasets/uit_openviic.yaml
```

Dữ liệu được tải về `data/` (khoảng 3,8 GB) và **không bao giờ được đưa lên repo**. Chi tiết
khảo sát, số liệu kiểm tra và cách đổi sang dataset khác: [docs/DATASETS.md](docs/DATASETS.md).

## Tiền xử lý

```powershell
python scripts/prepare_data.py --config configs/baseline_lstm.yaml
```

- Chuẩn hóa caption: Unicode NFC, chữ thường, bỏ dấu câu, thống nhất cách đặt dấu thanh
  (`hoà` thành `hòa`, `thuỷ` thành `thủy`).
- Tách từ tiếng Việt bằng `underthesea`; âm tiết của cùng một từ nối bằng dấu gạch dưới
  (`cây_xanh`). Thêm `text.level=syllable` vào cuối lệnh để dùng mức âm tiết.
- Từ điển chỉ xây từ tập train. Với ngưỡng tần suất 3: 3.482 token, tỉ lệ từ ngoài từ điển
  1,25% trên val và 0,96% trên test.
- Kết quả nằm ở `data/processed/uit_openviic/word_minfreq3/`: `captions.json`, `vocab.json`,
  `stats.json`.

## Cấu trúc thư mục

```
configs/    cấu hình thí nghiệm và cấu hình dataset (YAML)
data/       dữ liệu, đặc trưng đã cache (không đưa lên git)
docs/       tài liệu: lộ trình, dữ liệu, hướng dẫn phát triển
notebooks/  notebook khám phá
scripts/    script chạy từ dòng lệnh
src/vic/    mã nguồn chính
tests/      test đơn vị
```

## Tài liệu

- [Lộ trình dự án](docs/ROADMAP.md)
- [Dữ liệu: khảo sát, giấy phép, cách tải](docs/DATASETS.md)
- [Hướng dẫn phát triển](docs/DEVELOPMENT.md)

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE). Giấy phép này **chỉ áp dụng cho mã nguồn**.
Bộ dữ liệu tuân theo điều khoản riêng của tác giả bộ dữ liệu (xem mục Dữ liệu) và không nằm
trong repo này.
