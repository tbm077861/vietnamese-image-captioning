# Lộ trình dự án

Mục tiêu cuối: hệ thống sinh câu mô tả tiếng Việt cho ảnh, gồm mô hình đã huấn luyện, kết quả
đánh giá (BLEU-1..4, CIDEr, METEOR tham khảo), script suy luận, demo Gradio và README hoàn chỉnh.

Trạng thái: ✅ xong · 🔄 đang làm · ⬜ chưa làm

| # | Giai đoạn | Trạng thái |
|---|-----------|------------|
| 1 | Khởi tạo dự án | ✅ |
| 2 | Khảo sát và chọn dataset | ✅ |
| 3 | Tiền xử lý tiếng Việt và pipeline dữ liệu | ✅ |
| 4 | Mô hình baseline, huấn luyện, đánh giá, suy luận, kết quả baseline | ✅ |
| 5 | Mô hình mạnh hơn: bước 1–4 của kế hoạch bên dưới | ✅ |
| 6 | Tinh chỉnh, tối ưu CIDEr, hướng tùy chọn: bước 5–7 | 🔄 bước 5–6 đang làm |
| 7 | Đánh giá tổng hợp và phân tích lỗi | ⬜ |
| 8 | Demo Gradio và hoàn thiện README | ⬜ |

**Việc kế tiếp:** làm năm việc của mục "Bước 5–6: phạm vi đã duyệt (28/11/2026)" bên dưới, rồi
dừng và báo cáo. Bảng so sánh mọi run: [EXPERIMENTS.md](EXPERIMENTS.md).

**Quy tắc đánh giá:** mọi lựa chọn mô hình, siêu tham số và cách giải mã chỉ dựa trên tập val.
Tập test chỉ được dùng thêm một lần, để báo cáo mô hình cuối cùng.

## Giai đoạn 1–4 ✅

- **1 – Khởi tạo:** repo, cấu trúc, môi trường ảo, PyTorch bản CUDA, `scripts/check_env.py`.
- **2 – Dataset:** UIT-OpenViIC (13.100 ảnh, 61.241 caption do người viết); `download_data.py`,
  `verify_data.py`, định dạng annotation thống nhất. Xem [DATASETS.md](DATASETS.md).
- **3 – Tiền xử lý:** chuẩn hóa, tách từ, từ điển 3.482 token, `Dataset` / collate / sampler tái
  lập được; `prepare_data.py`.
- **4 – Baseline:** ResNet-50 đóng băng + LSTM, cache đặc trưng float16, vòng huấn luyện (mixed
  precision, checkpoint, resume chính xác), BLEU / CIDEr / METEOR; `extract_features.py`,
  `train.py`, `evaluate.py`, `predict.py`, `summarize_run.py`. Đã huấn luyện đầy đủ 30 epoch.

Kết quả baseline (chi tiết trong [RESULTS.md](RESULTS.md)): CIDEr val 0,4274 và BLEU-4 val 28,74
với beam 3. Điểm yếu đo được trên val:

- câu khuôn mẫu: 2.011 ảnh chỉ nhận 975 câu khác nhau;
- từ vựng nghèo: 357 âm tiết khác nhau, câu tham chiếu dùng 2.358;
- bịa ra người ở ảnh không có người; sai màu, số lượng, hành động; bỏ sót vật thể hiếm;
- lặp cụm từ trong câu ("có nhiều hơn một tầng có nhiều hơn một tầng");
- overfit nhẹ: loss val đi ngang từ khoảng epoch 16, CIDEr val dao động 0,01–0,02 giữa các epoch.

---

## Giai đoạn 5–6: kế hoạch đã được chủ dự án duyệt (14/11/2026)

Một lần huấn luyện chỉ mất vài phút, nên thời gian còn lại dành cho việc nâng chất lượng mô hình.

### Phạm vi đã duyệt

- **Làm liên tục Bước 1 đến Bước 4, rồi dừng và báo cáo.** Bước 5–7 chưa được làm cho tới khi
  chủ dự án duyệt tiếp.
- Sau **mỗi** bước: commit, mở Pull Request, cập nhật `docs/EXPERIMENTS.md`, file này và
  `docs/DEVELOPMENT.md` (trạng thái hiện tại, việc kế tiếp), rồi tự merge Pull Request bằng
  merge commit khi `pytest` và `ruff` qua. Mục đích: nếu công việc bị ngắt giữa chừng, người
  tiếp theo đọc tài liệu là làm tiếp được ngay.

### Các hướng đã cân nhắc

Mức lợi ích là ước lượng trước khi làm, chưa đo.

| Hướng | Lợi ích kỳ vọng | Chi phí | Rủi ro với 9 nghìn ảnh train, 6 GB VRAM |
|---|---|---|---|
| Attention trên lưới đặc trưng | Rõ: nhắm thẳng vào câu chung chung, sai màu, sai số lượng | Thấp: một decoder mới, dùng lại cache | Thấp |
| Transformer decoder | Ngang hoặc hơn attention một chút; là nền cho các hướng sau | Thấp đến trung bình: cần chỉnh warmup, dropout | Trung bình: dễ overfit với 41 nghìn caption |
| Encoder mạnh hơn (CLIP, SigLIP), vẫn cache | Lớn nhất: vật thể hiếm, cảnh Việt Nam | Trung bình: thêm thư viện, tải trọng số, cache vài GB mỗi encoder | Thấp về overfit; phải xác minh giấy phép trọng số và tương thích PyTorch |
| Decoder tiếng Việt pretrained | Không chắc: từ vựng giàu hơn nhưng caption có văn phong hẹp | Cao: tokenizer subword, mô hình 100–400 triệu tham số | Cao: overfit khi fine-tune, khó bảo trì |
| Label smoothing, regularization | Nhỏ nhưng chắc | Rất thấp: chỉ đổi cấu hình | Thấp; quét quá nhiều sẽ khớp vào val |
| Tối ưu CIDEr kiểu SCST | Lớn trên CIDEr, nhưng câu kém tự nhiên và kém đa dạng hơn | Trung bình: giải mã lấy mẫu, CIDEr làm phần thưởng | Trung bình đến cao: khớp vào CIDEr của train |
| Fine-tune encoder | Vừa với ResNet, không chắc với CLIP | Cao: bỏ cache, VRAM sát giới hạn | Cao: overfit, mất đặc trưng pretrained |

DINOv2 đã bị loại khỏi danh sách encoder (không được học cùng ngôn ngữ).

### Bước 1 – Hạ tầng so sánh ✅ (nhánh `feat/experiment-infra`)

**Kết quả:** cả tám mục dưới đây đã xong. Baseline qua ba seed (42, 43, 44): CIDEr val
0,4199 ± 0,0164, BLEU-4 28,13 ± 0,80 (beam 3), nên một chênh lệch phải vượt khoảng 0,033 CIDEr
mới được coi là thật. Dừng sớm với ngưỡng 5 epoch góp phần vào độ nhiễu (seed 44 dừng ở epoch
19). Chi tiết và bảng: [EXPERIMENTS.md](EXPERIMENTS.md). Ngoài kế hoạch có thêm
`scripts/compare_runs.py` (caption của nhiều run trên cùng 50 ảnh val cố định, phục vụ kiểm tra
độ tự nhiên ở Bước 6) và `scripts/update_experiments.py` (sinh lại bảng).

1. **Bảng thí nghiệm `docs/EXPERIMENTS.md`**, sinh tự động từ `outputs/*/` (`summary.json`,
   `eval_val_beam*.json`), chỉ dùng số liệu val. Mỗi run một dòng: encoder, decoder, seed, CIDEr
   val (beam 3), BLEU-4 val, số câu khác nhau, số âm tiết khác nhau, số tham số, thời gian huấn
   luyện, VRAM đỉnh, epoch tốt nhất, commit, ghi chú. Run thất bại vẫn nằm trong bảng.
2. **Chỉ số canh độ tự nhiên** lưu cùng kết quả đánh giá: số câu khác nhau, số âm tiết khác
   nhau, độ dài trung bình, tỉ lệ câu có cụm từ lặp. `vic.engine.report.diagnostics` đã tính ba
   chỉ số đầu; chỉ số "âm tiết lặp liền nhau" hiện có không bắt được kiểu lặp cả cụm, cần thay
   bằng đếm n-gram (2 và 3 âm tiết) lặp trong câu.
3. **Khóa tập test:** `evaluate.py --split test` phải bị từ chối nếu thiếu cờ `--final`.
4. **`train.py --init-from <checkpoint>`:** nạp trọng số mô hình mà không nạp optimizer và tiến
   độ (cần cho Bước 6 và Bước 7; `--resume` hiện khôi phục tất cả nên không dùng được).
5. Ghi commit git và tổng thời gian huấn luyện vào `summary.json` (hiện chỉ có thời gian mỗi
   bước và VRAM đỉnh), kèm cờ cho biết run đã chạy xong.
6. **`scripts/run_sweep.py`:** chạy tuần tự một danh sách file YAML (huấn luyện, đánh giá trên
   val với greedy và beam 3, cập nhật bảng thí nghiệm); bỏ qua run đã xong, tiếp tục run dở bằng
   resume; một run lỗi không làm dừng cả lượt. Chủ dự án sẽ tự dùng script này cho Bước 5.
7. **Đo độ nhiễu:** chạy baseline với seed 43 và 44 (seed 42 đã có ở `outputs/baseline_lstm`),
   ghi trung bình ± độ lệch chuẩn của CIDEr val vào bảng.
8. Bộ 50 ảnh val cố định theo seed để đọc bằng mắt
   (`summarize_run.py --sample-split val --samples 50`).

### Bước 2 – LSTM có attention ✅ (nhánh `feat/attention-decoder`)

**Kết quả:** CIDEr val 0,4860 ± 0,0024 so với 0,4199 ± 0,0164 của baseline (beam 3, ba seed):
cải thiện thật (vượt ngưỡng 0,033). Câu đa dạng hơn rõ (1.496–1.600 câu khác nhau so với
918–1.036). Beam rộng kém vì mô hình ưa câu chung chung hoặc vòng lặp khi được tìm kỹ hơn, không
phải lỗi điều kiện dừng. Cách giải mã chốt cho mô hình cuối: beam 3, LP 0,7, dừng `first`, chặn
lặp 3-gram (+0,006 đến +0,009 CIDEr trung bình, ở cả sáu checkpoint). Bảng so sánh mô hình giữ
giao thức cũ (không chặn lặp). Chi tiết: [EXPERIMENTS.md](EXPERIMENTS.md).

Decoder mới trong `src/vic/models/decoder.py`, hiện thực bốn hàm `forward`, `init_state`, `step`,
`reorder_state` và đăng ký vào `DECODERS["lstm_attention"]`. Không cần sửa `trainer.py`.

- `init_state(features)`: chiếu đặc trưng `(B, vùng, kênh)` xuống `attention_dim`; khởi tạo `h`,
  `c` từ trung bình các vùng. Trạng thái là bộ `(h, c, đặc_trưng_đã_chiếu)`.
- `step(tokens, state)`: trọng số `alpha = softmax(w · tanh(W_f f + W_h h))` trên các vùng; ngữ
  cảnh là tổng có trọng số; đưa `[embedding; ngữ cảnh]` vào `nn.LSTMCell`; trả logits.
- `forward`: lặp `step` theo thời gian (tối đa 31 bước).
- `reorder_state`: chọn lại theo chiều batch cho cả ba phần tử của trạng thái.
- `configs/attention_lstm.yaml` tạo từ `baseline_lstm.yaml`: đổi `run_name`,
  `model.decoder.type: lstm_attention`, thêm `attention_dim: 512`.
- Test: sao chép các test hợp đồng trong `tests/test_model.py` cho decoder mới.

**Kiểm tra beam search trong bước này** (yêu cầu của chủ dự án, vì beam kém greedy trên test):

- Hiện trạng: `beam_search` đã chuẩn hóa theo độ dài (`log_prob / độ_dài ^ alpha`, cấu hình
  `eval.length_penalty: 0.7`) và dừng khi đã có đủ `beam_size` câu hoàn thành. Với baseline: trên
  val, beam 3 nhỉnh hơn greedy còn beam 5 kém hơn; trên test, greedy cao nhất.
- Cần làm: so độ dài câu của greedy và từng beam; quét `length_penalty` (0 / 0,7 / 1,0 / 1,3)
  với beam 1, 3, 5 trên val; xem lại điều kiện dừng sớm (có thể bỏ lỡ câu dài điểm cao hơn);
  thêm tùy chọn chặn lặp n-gram vì mô hình hay lặp cụm từ.
- Chốt cách giải mã **trên val**, ghi vào `docs/EXPERIMENTS.md`. Không dùng test để chốt.

### Bước 3 – Encoder mạnh hơn ✅ (nhánh `feat/strong-encoders`)

**Kết quả:** hai encoder `clip_vit_b16` (CLIP ViT-B/16, MIT) và `siglip_b16` (SigLIP ViT-B/16,
Apache-2.0), giấy phép xác minh tại nguồn trong [MODELS.md](MODELS.md), đặc trưng là lưới 196
vùng × 768 chiều. Với LSTM attention, CIDEr val (beam 3, ba seed) đạt 0,6512 ± 0,0046 (CLIP) và
0,6570 ± 0,0058 (SigLIP), so với 0,4860 ± 0,0024 của ResNet-50: tăng khoảng 0,17, vượt xa mức
nhiễu. CLIP và SigLIP chênh nhau trong mức nhiễu; **chọn SigLIP** vì trung bình cao hơn (cùng
chiều: giấy phép Apache-2.0, đặc trưng không phụ thuộc cỡ lô). Decoder LSTM không attention chỉ
tăng 0,03–0,06. Cấu hình CLIP / SigLIP đọc cache bằng memory-map
(`data.features_in_memory: false`) vì nạp cả cache vào RAM làm máy gần hết bộ nhớ. Ngoài kế
hoạch: `summary.json` ghi các lần chạy tiếp và ghi đè cấu hình; bảng thí nghiệm có thêm cột giải
mã chốt cho mọi run (`src/vic/engine/protocol.py`). Chi tiết, RAM từng run:
[EXPERIMENTS.md](EXPERIMENTS.md).

- **Đúng hai ứng viên: một mô hình CLIP và một mô hình SigLIP.** Bỏ DINOv2.
- **Giấy phép trọng số:** chỉ dùng trọng số có giấy phép cho phép nghiên cứu, ưu tiên MIT hoặc
  Apache-2.0. Phải xác minh tại nguồn (trang mô hình và file LICENSE của repo gốc) **trước khi
  tải**, rồi ghi vào `docs/MODELS.md`: tên mô hình, nguồn, câu giấy phép trích nguyên văn, ngày
  kiểm tra.
- **Thư viện:** `open_clip_torch` hoặc `transformers`; kiểm tra tương thích với PyTorch 2.14.1,
  ghim phiên bản trong `requirements.txt`.
- **Lưu đặc trưng dạng lưới patch** (không chỉ vector gộp) để attention và Transformer dùng được,
  dạng float16 như cache hiện tại. Ước lượng dung lượng cho 13.100 ảnh: ViT-B/16 ở 224 px (196
  vùng × 768 chiều) khoảng 3,9 GB; ViT-L/14 ở 224 px (256 × 1024) khoảng 6,9 GB. Mô hình độ phân
  giải cao hơn phải gộp lưới xuống tối đa khoảng 196 vùng. Cache lớn thì đặt
  `data.features_in_memory=false` để memory-map.
- **Code cần sửa:** `BACKBONES` / `ImageEncoder` hiện chỉ có ResNet; mỗi encoder mới cần hệ số
  chuẩn hóa và kích thước ảnh riêng, nên `preprocess_image` phải nhận tham số theo encoder. Số
  chiều đặc trưng của decoder đang lấy từ `BACKBONES[tên][2]`.
- **Trọng số và cache không được commit.** `data/` đã bị git bỏ qua; thư mục cache của thư viện
  phải nằm ngoài repo.
- **Thí nghiệm:** huấn luyện baseline LSTM và LSTM attention trên từng bộ đặc trưng (ResNet-50,
  CLIP, SigLIP), chọn encoder theo CIDEr val (beam 3).
- Việc trích cache có thể chạy nền trong lúc viết code Bước 2.

### Bước 4 – Transformer decoder ✅ (nhánh `feat/transformer-decoder`)

**Kết quả:** `DECODERS["transformer"]` chạy trên SigLIP, CLIP (hai encoder sát nhau ở Bước 3)
và ResNet-50 (đối chứng), một seed. Với cấu hình khởi điểm, Transformer kém LSTM attention cùng
encoder 0,028–0,041 CIDEr vì overfit sớm (epoch tốt nhất 6–8). Thêm một run dropout 0,3 trên
SigLIP (trong khoảng 0,1–0,3 của kế hoạch): CIDEr 0,6437, vẫn kém LSTM attention + SigLIP
(0,6570 ± 0,0058) và BLEU-4 thấp hơn 2,4 điểm. **Cặp đề xuất: LSTM attention + SigLIP
ViT-B/16.** Danh sách cấu hình cho Bước 5 nằm trong `configs/sweeps/` (chưa chạy). Chi tiết:
[EXPERIMENTS.md](EXPERIMENTS.md).

- Lớp `TransformerCaptionDecoder`, đăng ký `DECODERS["transformer"]`: chiếu đặc trưng xuống
  `d_model` (512), cộng embedding vị trí học được cho các vùng; `nn.TransformerDecoder` 3 lớp, 8
  đầu, feed-forward 2048, dropout 0,1 đến 0,3; mặt nạ nhân quả và mặt nạ padding.
- Giải mã từng bước: trạng thái là `(memory, các_token_đã_sinh)`; mỗi `step` chạy lại decoder
  trên tiền tố và lấy logits ở vị trí cuối (câu tối đa 30 token nên chi phí chấp nhận được).
- Cấu hình khởi điểm: `train.lr: 2.0e-4`, `warmup_steps: 1500`, `label_smoothing: 0.1`,
  `batch_size: 64`.
- Chạy trên encoder đã chọn ở Bước 3, kèm đối chứng trên ResNet-50. Nếu hai encoder sát nhau ở
  Bước 3 thì thử Transformer trên cả hai.
- **Cuối bước:** chuẩn bị sẵn danh sách cấu hình cho Bước 5 (tối đa 15 run, đặt trong
  `configs/sweeps/`) nhưng **không chạy**.

### Dừng và báo cáo sau Bước 4

Báo cáo gửi chủ dự án phải có:

1. Bảng so sánh tất cả mô hình trên val: CIDEr (beam 3), BLEU-4, số câu khác nhau, số âm tiết
   khác nhau, số tham số, thời gian huấn luyện, VRAM đỉnh; kèm độ nhiễu giữa các seed.
2. Cặp encoder – decoder được đề xuất chọn, và lý do.
3. Cùng 10 ảnh mẫu như baseline (danh sách mã ảnh trong `RESULTS.md`), đặt caption của baseline
   cạnh caption của mô hình tốt nhất. Lấy caption bằng `predict.py` trên 10 file ảnh đó; không
   chạy đánh giá trên tập test.
4. Đề xuất cụ thể cho Bước 5–6, và phần nào nên ưu tiên làm trước.

### Việc phải làm trước Bước 5

Phát hiện trong Bước 3:

1. ✅ **Tách tiến trình cho mỗi run** trong `scripts/run_sweep.py`. Trước đây mọi cấu hình chạy
   nối tiếp trong cùng một tiến trình Python. Nay mỗi run có tiến trình riêng, và tiến trình đó
   từ chối chạy nếu repo không phải bản checkout sạch.
2. **Cho encoder ViT chạy float32** khi trích cache và khi suy luận: chuyển sang Bước 7 (tùy
   chọn). Chủ dự án quyết định giữ cache float16 hiện có. Đặc trưng CLIP tính ở float16 phụ
   thuộc cỡ lô (lệch khoảng 0,5% theo L2); SigLIP thì không (chi tiết: [MODELS.md](MODELS.md)).

### Bước 5–6: phạm vi đã duyệt (28/11/2026)

- **Mô hình nền:** LSTM attention + SigLIP ViT-B/16 (`attention_siglip_b16`), giữ cache float16.
- **Cách chạy:** mọi lượt huấn luyện chạy nền, có đo RAM, dừng khi RAM trống dưới 1,0 GB. Không
  bao giờ chạy hai tiến trình huấn luyện cùng lúc. Không chạy gì trên tập test.

Năm việc, theo thứ tự:

1. Tách tiến trình cho mỗi run trong `run_sweep.py`, kèm kiểm tra cây làm việc sạch; có test.
2. Chạy seed 43 và 44 cho `transformer_siglip_b16_dropout03`; chọn decoder theo trung bình CIDEr
   val beam 3 của ba seed.
3. Lượt quét Bước 5 (`configs/sweeps/`), chạy từ repo chính ở một commit sạch, đọc kết quả theo
   luật ghi trong [EXPERIMENTS.md](EXPERIMENTS.md), mục "Tinh chỉnh (Bước 5)".
4. SCST (Bước 6), viết trong một git worktree riêng trong lúc lượt quét chạy, và không sửa repo
   chính khi lượt quét đang chạy. Chỉ chạy thử ngắn, sau khi lượt quét xong.
5. Dừng và báo cáo. Báo cáo gồm:
   - bảng Bước 5;
   - ba seed của Transformer;
   - cấu hình thắng và kết quả `study_decoding`;
   - kết quả chạy thử SCST kèm 10 ảnh mẫu;
   - các ngưỡng độ tự nhiên đề xuất;
   - số liệu RAM.

Chưa chạy SCST đầy đủ và chưa chạy tập test cho tới khi chủ dự án duyệt.

### Bước 5 – Tinh chỉnh 🔄

Trên cặp encoder – decoder tốt nhất, tối đa 15 run, mỗi run đổi một yếu tố: label smoothing (0 và
0,1), dropout, learning rate, số lớp và độ rộng, mức từ hay âm tiết, `text.min_freq` (đổi thì phải
chạy lại `prepare_data.py`), và cách giải mã (beam, length penalty, chặn lặp). Chạy bằng
`scripts/run_sweep.py` với danh sách đã chuẩn bị ở cuối Bước 4.

**Danh sách đã chuẩn bị** (`configs/sweeps/`, gốc là `attention_siglip_b16`, CIDEr val
0,6570 ± 0,0058): 10 run seed 42, mỗi run đổi một yếu tố so với gốc.

| # | Run | Yếu tố đổi (gốc → mới) |
|---:|---|---|
| 1 | `attention_siglip_b16_ls01` | `train.label_smoothing` 0 → 0,1 |
| 2 | `attention_siglip_b16_dropout03` | `model.decoder.dropout` 0,5 → 0,3 |
| 3 | `attention_siglip_b16_lr5e-4` | `train.lr` 1e-3 → 5e-4 |
| 4 | `attention_siglip_b16_lr2e-3` | `train.lr` 1e-3 → 2e-3 |
| 5 | `attention_siglip_b16_hidden1024` | `model.decoder.hidden_dim` 512 → 1024 |
| 6 | `attention_siglip_b16_wd001` | `train.weight_decay` 0 → 0,01 (AdamW) |
| 7 | `attention_siglip_b16_batch64` | `train.batch_size` 128 → 64 |
| 8 | `attention_siglip_b16_patience10` | `train.early_stopping_patience` 5 → 10 |
| 9 | `attention_siglip_b16_syllable` | `text.level` word → syllable |
| 10 | `attention_siglip_b16_minfreq5` | `text.min_freq` 3 → 5 |

- Decoder LSTM attention chỉ có một lớp nên không có run đổi số lớp; độ rộng thử ở run 5.
- Cách giải mã không cần huấn luyện lại: đo trên checkpoint tốt nhất bằng
  `scripts/study_decoding.py`, không tính vào 15 run.
- Run 9 và 10 cần chuẩn bị dữ liệu một lần trước (lệnh ghi ở đầu mỗi file cấu hình):
  `python -I scripts/prepare_data.py --config configs/sweeps/09_attention_siglip_b16_syllable.yaml`
  và tương tự với file `10_...`. Thiếu bước này thì run đó báo lỗi, các run khác vẫn chạy.
  Dữ liệu ghi vào thư mục riêng (`syllable_minfreq3/`, `word_minfreq5/`), không ghi đè dữ liệu
  của các run khác.
- Lệnh chạy cả danh sách: `python -I -X utf8 scripts/run_sweep.py configs/sweeps`. Ngân sách 15
  run còn 5, dành cho bước xác nhận.
- **Luật đọc kết quả**, ghi trước khi chạy trong [EXPERIMENTS.md](EXPERIMENTS.md):
  - so cặp với `attention_siglip_b16` seed 42;
  - ngưỡng đáng xác nhận là +0,012 CIDEr beam 3, bằng hai lần độ lệch chuẩn giữa các seed của
    chính cấu hình này;
  - một thay đổi vượt ngưỡng thì chạy seed 43/44 cho nó;
  - từ hai thay đổi trở lên thì chạy seed 43/44 cho thay đổi tốt nhất, cộng một cấu hình gộp hai
    thay đổi tốt nhất với ba seed;
  - không thay đổi nào vượt ngưỡng thì giữ cấu hình gốc;
  - mô hình thắng chọn theo trung bình ba seed, rồi chạy `study_decoding.py` (thêm chặn lặp
    2-gram).

### Bước 6 – Tối ưu CIDEr kiểu SCST 🔄 (viết code và chạy thử; chạy đầy đủ chờ duyệt)

- Bắt đầu từ checkpoint tốt nhất huấn luyện bằng cross-entropy (`--init-from`), learning rate
  thấp; phần thưởng là CIDEr-D của câu lấy mẫu trừ CIDEr-D của câu greedy, tính với câu tham
  chiếu và tần suất văn bản của tập train. Cần thêm đường giải mã lấy mẫu cho decoder và một bản
  CIDEr nhận tần suất văn bản từ ngoài. Cách đánh giá trên val giữ nguyên.
- Ghi các chỉ số độ tự nhiên theo từng epoch:
  - số câu khác nhau và số âm tiết khác nhau;
  - độ dài trung bình;
  - tỷ lệ câu lặp 2-gram và lặp 3-gram;
  - tỷ lệ câu kết thúc bằng hư từ ("và", "của", "với", "một", "có"…).
- Đề xuất ngưỡng chặn cho các chỉ số đó; kết quả SCST chỉ tính là cải thiện khi không vi phạm
  ngưỡng. Chủ dự án duyệt ngưỡng trước khi chạy đầy đủ.
- Chạy thử ngắn từ checkpoint `attention_siglip_b16` seed 44, sau khi lượt quét Bước 5 xong.
- **Tiêu chí đã duyệt:** bản SCST phải qua kiểm tra độ tự nhiên (các chỉ số canh ở Bước 1 và bộ
  50 ảnh val đọc bằng mắt không tệ hơn bản cross-entropy). Nếu không qua, mô hình cuối là bản
  cross-entropy và số SCST chỉ được báo cáo như một phép so sánh.

### Bước 7 – Hướng tùy chọn ⬜ (quyết định sau)

- Cho encoder ViT chạy float32 khi trích cache và khi suy luận (chuyển từ "Việc phải làm trước
  Bước 5"). Phải trích lại cache và huấn luyện lại mô hình được chọn sau khi đổi.
- Decoder tiếng Việt pretrained (cỡ BARTpho hoặc GPT-2 tiếng Việt nhỏ; mô hình hàng tỉ tham số
  không huấn luyện được trong 6 GB). Giấy phép phải xác minh như ở Bước 3.
- Fine-tune một phần encoder. **Mặc định là không làm.** Nếu làm: chạy với
  `data.use_cached_features=false model.encoder.trainable=true train.batch_size=32
  train.num_workers=4 train.encoder_lr=1e-5`, chỉ mở khóa các khối cuối, và **không lật ngang
  ảnh** khi tăng cường dữ liệu vì caption dùng nhiều "bên trái", "bên phải".

### So sánh công bằng

- Cùng split, cùng tiền xử lý, điểm tính trên âm tiết, cùng số epoch tối đa và luật dừng sớm.
- Mỗi thí nghiệm một file YAML và một `run_name` riêng; không sửa cấu hình của run đã chạy; mỗi
  lần chỉ đổi một yếu tố so với mô hình tốt nhất hiện tại.
- Chỉ số so sánh: CIDEr val với beam 3 là chính, BLEU-4 là phụ. Không dùng METEOR để so sánh.
- Sàng lọc bằng một seed (42). Các ứng viên sát nhau chạy lại ba seed (42, 43, 44) và báo trung
  bình ± độ lệch chuẩn. Một chênh lệch chỉ được coi là thật khi vượt hai lần độ lệch chuẩn (của
  cấu hình đang được so; với `attention_siglip_b16` là 0,012) và vượt 0,01 CIDEr.
- Luôn báo kèm số tham số, thời gian huấn luyện, VRAM đỉnh và các chỉ số canh độ tự nhiên.

### Chọn mô hình cuối

1. CIDEr val (beam 3) trung bình ba seed cao nhất. BLEU-4 là chỉ số phụ, luôn báo kèm, nhưng
   không dùng để loại một ứng viên khi chênh CIDEr còn nằm trong mức nhiễu.
2. Qua kiểm tra độ tự nhiên (xem Bước 6).
3. Suy luận chạy được trong 6 GB VRAM và trên CPU; tái lập được từ YAML và seed.
4. Sau đó mới chạy tập test **một lần** (`--final`) với cách giải mã đã chốt trên val. Điểm test
   của baseline trong `RESULTS.md` được giữ làm mốc.

---

## Giai đoạn 7 – Đánh giá tổng hợp và phân tích lỗi ⬜

1. Bảng BLEU-1..4, CIDEr (chính) và METEOR (tham khảo) trên test cho mô hình cuối, cạnh baseline.
2. Phân tích lỗi trong `notebooks/error_analysis.ipynb`: ảnh điểm cao nhất và thấp nhất theo
   CIDEr từng ảnh (`cider_d` trả về điểm từng ảnh); phân loại lỗi (số lượng, màu, trái / phải,
   bịa vật thể, lặp, câu cụt); so độ dài và độ đa dạng từ vựng với câu tham chiếu.
3. (Tùy chọn) đối chiếu một checkpoint với `pycocoevalcap`.

**Tiêu chí hoàn thành:** bảng kết quả test và phần phân tích lỗi có trong `RESULTS.md`.

## Giai đoạn 8 – Demo và hoàn thiện ⬜

1. `scripts/demo.py` dùng Gradio: tải ảnh lên, chọn độ rộng beam, nhận câu mô tả. Lớp
   `CaptionPredictor` (`src/vic/engine/inference.py`) đã lo việc nạp checkpoint và sinh câu.
2. Ghim `gradio` trong `requirements.txt`; thêm lệnh chạy demo vào README.
3. README hoàn chỉnh: bảng kết quả, vài ví dụ đúng và sai, hạn chế đã biết, trích dẫn bộ dữ liệu.
4. **Không công bố checkpoint** (đã chốt): mô hình học từ dữ liệu chỉ cho phép nghiên cứu phi
   thương mại. Demo chạy cục bộ với checkpoint do người dùng tự huấn luyện.

**Tiêu chí hoàn thành:** người mới làm theo README chạy được suy luận và demo từ checkpoint.
