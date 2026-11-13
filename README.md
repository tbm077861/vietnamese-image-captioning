# Vietnamese Image Captioning

Hệ thống sinh câu mô tả tiếng Việt cho ảnh bằng deep learning: đầu vào là một ảnh, đầu ra là
một câu mô tả tiếng Việt tự nhiên.

> **Trạng thái:** đang phát triển. Xem [lộ trình](docs/ROADMAP.md) để biết giai đoạn hiện tại.

## Cài đặt

Yêu cầu: Python 3.10, GPU NVIDIA có driver hỗ trợ CUDA 13.0 (đã thử trên RTX 5070 Laptop 8 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_env.py
```

`check_env.py` in phiên bản PyTorch, CUDA, tên GPU và chạy thử một phép tính trên GPU.

## Cấu trúc thư mục

```
configs/    cấu hình thí nghiệm (YAML)
data/       dữ liệu, đặc trưng đã cache (không đưa lên git)
docs/       tài liệu: lộ trình, hướng dẫn phát triển
notebooks/  notebook khám phá
scripts/    script chạy từ dòng lệnh
src/vic/    mã nguồn chính
tests/      test đơn vị
```

## Dữ liệu và giấy phép

Sẽ cập nhật sau giai đoạn khảo sát dataset: nguồn, giấy phép và cách tải.

## Tài liệu

- [Lộ trình dự án](docs/ROADMAP.md)
- [Hướng dẫn phát triển](docs/DEVELOPMENT.md)
