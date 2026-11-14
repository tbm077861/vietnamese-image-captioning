# Lộ trình dự án

Mục tiêu cuối: hệ thống sinh câu mô tả tiếng Việt cho ảnh, gồm mô hình đã huấn luyện, kết quả
đánh giá (BLEU-1..4, CIDEr, METEOR tham khảo), script suy luận, demo Gradio và README hoàn chỉnh.

Trạng thái: ✅ xong · 🔄 đang làm · ⬜ chưa làm

| # | Giai đoạn | Nhánh | Trạng thái |
|---|-----------|-------|------------|
| 1 | Khởi tạo dự án | `main` | ✅ |
| 2 | Khảo sát và chọn dataset | `feat/dataset-survey` | ✅ |
| 3 | Tiền xử lý tiếng Việt và pipeline dữ liệu | `feat/preprocessing` | ✅ |
| 4 | Mô hình baseline, huấn luyện, đánh giá, suy luận | `feat/baseline-model` | ⬜ |
| 5 | Mô hình nâng cao (attention, Transformer) | `feat/advanced-model` | ⬜ |
| 6 | Huấn luyện đầy đủ và tinh chỉnh | `feat/training` | ⬜ |
| 7 | Đánh giá tổng hợp và phân tích lỗi | `feat/evaluation` | ⬜ |
| 8 | Demo Gradio và hoàn thiện README | `feat/demo` | ⬜ |

## Giai đoạn 1 – Khởi tạo dự án ✅

- Repo, cấu trúc thư mục, `.gitignore`, môi trường ảo, PyTorch bản CUDA, `scripts/check_env.py`.

**Tiêu chí hoàn thành:** `python scripts/check_env.py` in đúng tên GPU và báo `GPU check: OK`;
`pytest` chạy qua; `main` đã được đẩy lên GitHub.

## Giai đoạn 2 – Khảo sát và chọn dataset ✅

- Khảo sát các bộ image captioning tiếng Việt, xác minh giấy phép tại nguồn gốc.
- Chọn UIT-OpenViIC (13.100 ảnh, 61.241 caption do người viết, tải tự do, dùng cho nghiên cứu).
- `docs/DATASETS.md`, `scripts/download_data.py`, `scripts/verify_data.py`.

**Tiêu chí hoàn thành:** dữ liệu tải về `data/` và qua kiểm tra (số ảnh, số caption, ảnh hỏng,
thống kê độ dài); README ghi nguồn và giấy phép; đổi dataset chỉ cần sửa YAML.

## Giai đoạn 3 – Tiền xử lý tiếng Việt và pipeline dữ liệu ✅

- Chuẩn hóa Unicode (NFC), chữ thường, dấu câu; tách từ tiếng Việt (tùy chọn mức âm tiết).
- Từ điển có ngưỡng tần suất và token đặc biệt; chia train/val/test cố định theo seed.
- `Dataset` / `DataLoader` có padding; test đơn vị.

**Tiêu chí hoàn thành:** `pytest` qua; một batch mẫu có shape đúng và giải mã ngược ra đúng caption.

## Giai đoạn 4 – Mô hình baseline

- Encoder ảnh pretrained (đóng băng) + decoder LSTM; giải mã greedy và beam search.
- Trích xuất và cache đặc trưng ảnh (fp16) ra `data/`; vẫn giữ tùy chọn chạy encoder trực tiếp.
- Vòng huấn luyện: mixed precision, checkpoint, resume, log, seed, cấu hình YAML.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`.

**Tiêu chí hoàn thành:** chạy thử vài trăm bước thấy loss giảm và GPU được dùng; resume từ
checkpoint hoạt động; script đánh giá và suy luận chạy được trên checkpoint thử.

## Giai đoạn 5 – Mô hình nâng cao

- Decoder LSTM có attention trên lưới đặc trưng (kiểu Show, Attend and Tell).
- Decoder Transformer trên đặc trưng lưới; cân nhắc encoder ViT.
- So sánh với baseline trên cùng dữ liệu, cùng seed, cùng cách đánh giá.

**Tiêu chí hoàn thành:** mỗi mô hình có file YAML riêng, huấn luyện được trong 8 GB VRAM và có
bảng so sánh với baseline.

## Giai đoạn 6 – Huấn luyện đầy đủ và tinh chỉnh

- Huấn luyện đầy đủ từng mô hình; early stopping theo CIDEr trên tập val.
- Tinh chỉnh: learning rate, dropout, label smoothing, kích thước beam, fine-tune encoder.

**Tiêu chí hoàn thành:** mỗi thí nghiệm có cấu hình, log và checkpoint tốt nhất; kết quả tái lập
được với cùng seed.

## Giai đoạn 7 – Đánh giá tổng hợp và phân tích lỗi

- Bảng BLEU-1..4, CIDEr (chính) và METEOR (tham khảo) trên tập test cho mọi mô hình.
- Phân tích lỗi định tính: ví dụ đúng, sai, các kiểu lỗi thường gặp.

**Tiêu chí hoàn thành:** bảng kết quả và phần phân tích có trong README / `docs/`.

## Giai đoạn 8 – Demo và hoàn thiện

- Demo Gradio: tải ảnh lên, nhận câu mô tả tiếng Việt.
- README hoàn chỉnh: cài đặt, dữ liệu, huấn luyện, đánh giá, suy luận, demo, kết quả, giấy phép.

**Tiêu chí hoàn thành:** người mới làm theo README chạy được suy luận và demo từ checkpoint.
