# Lộ trình dự án

Mục tiêu cuối: hệ thống sinh câu mô tả tiếng Việt cho ảnh, gồm mô hình đã huấn luyện, kết quả
đánh giá (BLEU-1..4, CIDEr, METEOR tham khảo), script suy luận, demo Gradio và README hoàn chỉnh.

Trạng thái: ✅ xong · 🔄 đang làm · ⬜ chưa làm

| # | Giai đoạn | Nhánh | Trạng thái |
|---|-----------|-------|------------|
| 1 | Khởi tạo dự án | `main` | ✅ |
| 2 | Khảo sát và chọn dataset | `feat/dataset-survey` | ✅ |
| 3 | Tiền xử lý tiếng Việt và pipeline dữ liệu | `feat/preprocessing` | ✅ |
| 4 | Mô hình baseline, huấn luyện, đánh giá, suy luận | `feat/baseline-model` | ✅ |
| 5 | Mô hình nâng cao (attention, Transformer) | `feat/advanced-model` | ⬜ |
| 6 | Huấn luyện đầy đủ và tinh chỉnh | `feat/training` | ⬜ |
| 7 | Đánh giá tổng hợp và phân tích lỗi | `feat/evaluation` | ⬜ |
| 8 | Demo Gradio và hoàn thiện README | `feat/demo` | ⬜ |

Baseline đã huấn luyện đầy đủ: CIDEr val 0,4274 và BLEU-4 val 28,74 (beam 3); chi tiết và phân
tích lỗi trong [RESULTS.md](RESULTS.md). Đây là mốc so sánh cho mọi mô hình sau.

**Quy tắc đánh giá:** mọi lựa chọn mô hình, siêu tham số và cách giải mã chỉ dựa trên tập val.
Tập test chỉ được dùng thêm một lần, để báo cáo mô hình cuối cùng.

## Giai đoạn 1 – Khởi tạo dự án ✅

Repo, cấu trúc thư mục, `.gitignore`, môi trường ảo, PyTorch bản CUDA, `scripts/check_env.py`.

## Giai đoạn 2 – Khảo sát và chọn dataset ✅

Chọn UIT-OpenViIC (13.100 ảnh, 61.241 caption do người viết). `docs/DATASETS.md`,
`scripts/download_data.py`, `scripts/verify_data.py`, định dạng annotation thống nhất.

## Giai đoạn 3 – Tiền xử lý tiếng Việt và pipeline dữ liệu ✅

Chuẩn hóa, tách từ, từ điển 3.482 token, `Dataset` / collate / sampler tái lập được,
`scripts/prepare_data.py`.

## Giai đoạn 4 – Mô hình baseline ✅

ResNet-50 đóng băng + LSTM, cache đặc trưng float16, vòng huấn luyện (mixed precision,
checkpoint, resume chính xác, log), BLEU / CIDEr / METEOR, `train.py`, `evaluate.py`,
`predict.py`, `summarize_run.py`. Baseline đã huấn luyện đầy đủ 30 epoch; kết quả trong
[RESULTS.md](RESULTS.md).

---

## Giai đoạn 5 – Mô hình nâng cao ⬜

Mục tiêu: hai decoder mạnh hơn baseline, dùng lại nguyên cache đặc trưng 49 vùng và toàn bộ
vòng huấn luyện. Mỗi decoder là một lớp mới trong `src/vic/models/decoder.py`, hiện thực bốn
hàm `forward`, `init_state`, `step`, `reorder_state` rồi đăng ký vào `DECODERS`. Không cần sửa
`trainer.py`, `decoding.py` hay các script.

### 5a. LSTM có attention (kiểu Show, Attend and Tell)

1. Thêm lớp `AttentionLSTMDecoder`, đăng ký `DECODERS["lstm_attention"]`.
   - `init_state(features)`: chiếu đặc trưng `(B, 49, 2048)` xuống `attention_dim`; khởi tạo
     `h`, `c` từ trung bình các vùng. Trạng thái là bộ `(h, c, đặc_trưng_đã_chiếu)`.
   - `step(tokens, state)`: trọng số `alpha = softmax(w · tanh(W_f f + W_h h))` trên 49 vùng;
     ngữ cảnh là tổng có trọng số; đưa `[embedding; ngữ cảnh]` vào `nn.LSTMCell`; trả logits.
   - `forward`: lặp `step` theo thời gian (tối đa 31 bước).
   - `reorder_state`: chọn lại theo chiều batch cho cả ba phần tử của trạng thái.
2. Tạo `configs/attention_lstm.yaml` từ `baseline_lstm.yaml`: đổi `run_name`,
   `model.decoder.type: lstm_attention`, thêm `attention_dim: 512`.
3. Test: sao chép các test hợp đồng trong `tests/test_model.py` (teacher forcing trùng với
   chạy từng bước; beam 1 trùng greedy; `reorder_state`) cho decoder mới.

### 5b. Decoder Transformer

1. Thêm lớp `TransformerCaptionDecoder`, đăng ký `DECODERS["transformer"]`.
   - Chiếu đặc trưng xuống `d_model` (512), cộng embedding vị trí học được cho 49 vùng.
   - `nn.TransformerDecoder` 3 lớp, 8 đầu, feed-forward 2048, dropout 0,1; embedding vị trí cho
     token; mặt nạ nhân quả và mặt nạ padding.
   - Giải mã từng bước: trạng thái là `(memory, các_token_đã_sinh)`; mỗi `step` chạy lại decoder
     trên tiền tố và lấy logits ở vị trí cuối (câu tối đa 30 token nên chi phí chấp nhận được).
2. `configs/transformer.yaml`: `train.lr: 2.0e-4`, `warmup_steps: 1500`,
   `label_smoothing: 0.1`, `batch_size: 64`.
3. Cân nhắc chuyển sang mức âm tiết (`text.level=syllable`) cho Transformer và so sánh.

### 5c. (Tùy chọn) Encoder khác

Thêm một mục vào `BACKBONES` trong `src/vic/models/encoder.py` (ví dụ ViT-B/16), chạy lại
`extract_features.py`; cache mới nằm ở thư mục riêng theo tên encoder nên không ghi đè cache cũ.

**Tiêu chí hoàn thành:** mỗi mô hình có file YAML riêng, test hợp đồng qua, chạy thử vài trăm
bước thấy loss giảm trong 8 GB VRAM.

## Giai đoạn 6 – Huấn luyện đầy đủ và tinh chỉnh ⬜

1. **Baseline đầy đủ: đã xong** (30 epoch, khoảng 8 phút; kết quả trong `RESULTS.md`). Lệnh
   dùng cho mọi mô hình sau cũng theo mẫu này, chỉ đánh giá trên val:

   ```powershell
   python scripts/train.py --config configs/<thí nghiệm>.yaml
   python scripts/evaluate.py --checkpoint outputs/<run>/checkpoints/best.pt --split val --beam-size 1 3
   python scripts/summarize_run.py --run outputs/<run> --sample-split val
   ```

2. Huấn luyện đầy đủ các mô hình của Giai đoạn 5 với cùng seed.
3. Tinh chỉnh, mỗi lần đổi một yếu tố và đặt `run_name` riêng:
   - `train.lr`: 5e-4, 1e-3, 2e-3; `model.decoder.dropout`: 0,3 và 0,5;
   - `train.label_smoothing`: 0 và 0,1; `text.min_freq`: 3 và 5 (phải chạy lại `prepare_data.py`);
   - `text.level`: word và syllable;
   - khi đánh giá: `--beam-size` 1, 3, 5 và `eval.length_penalty` 0 / 0,7 / 1,0.
4. Fine-tune encoder (sau khi decoder đã hội tụ):
   - cần thêm tùy chọn nạp trọng số từ checkpoint mà không nạp optimizer (ví dụ `--init-from`);
     hiện `--resume` khôi phục cả optimizer và tiến độ nên không dùng cho việc này được;
   - chạy với `data.use_cached_features=false model.encoder.trainable=true
     train.batch_size=32 train.num_workers=4 train.encoder_lr=1e-5`;
   - chỉ mở khóa khối cuối của ResNet nếu thiếu VRAM;
   - **không lật ngang ảnh** khi tăng cường dữ liệu: caption của bộ này dùng nhiều "bên trái",
     "bên phải".
5. Ghi mọi thí nghiệm vào `docs/EXPERIMENTS.md`: cấu hình, seed, điểm val, điểm test, thời gian.

**Tiêu chí hoàn thành:** mỗi thí nghiệm có cấu hình, log và `best.pt`; chạy lại cùng seed cho
kết quả tương đương; có bảng so sánh các mô hình trên val.

## Giai đoạn 7 – Đánh giá tổng hợp và phân tích lỗi ⬜

1. Chạy `evaluate.py --split test` cho checkpoint tốt nhất của từng mô hình (beam 3), đưa bảng
   BLEU-1..4, CIDEr (chính) và METEOR (tham khảo) vào README.
2. Phân tích lỗi trong `notebooks/error_analysis.ipynb`:
   - `cider_d` trả về điểm từng ảnh: xem 20 ảnh điểm cao nhất và 20 ảnh thấp nhất;
   - phân loại lỗi: đếm sai số lượng, sai màu, sai trái / phải, bịa vật thể, câu lặp, câu cụt;
   - so độ dài câu sinh ra với câu tham chiếu; đếm số từ khác nhau mô hình thực sự dùng.
3. (Tùy chọn) đối chiếu một checkpoint với `pycocoevalcap`; dùng phần tiếng Việt của XM3600 làm
   tập kiểm tra ngoài miền.

**Tiêu chí hoàn thành:** bảng kết quả test và phần phân tích lỗi có trong README hoặc `docs/`.

## Giai đoạn 8 – Demo và hoàn thiện ⬜

1. `scripts/demo.py` dùng Gradio: tải ảnh lên, chọn độ rộng beam, nhận câu mô tả. Lớp
   `CaptionPredictor` (`src/vic/engine/inference.py`) đã lo việc nạp checkpoint và sinh câu.
2. Ghim `gradio` trong `requirements.txt`; thêm lệnh chạy demo vào README.
3. README hoàn chỉnh: bảng kết quả, vài ví dụ đúng và sai (ảnh nhỏ đặt trong `docs/assets/`),
   hạn chế đã biết, trích dẫn bộ dữ liệu.
4. **Không công bố checkpoint** (đã chốt): mô hình học từ dữ liệu chỉ cho phép nghiên cứu phi
   thương mại. Demo chạy cục bộ với checkpoint do người dùng tự huấn luyện.

**Tiêu chí hoàn thành:** người mới làm theo README chạy được suy luận và demo từ checkpoint.
