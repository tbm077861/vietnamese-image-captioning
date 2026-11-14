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
| 5 | Mô hình mạnh hơn: bước 1–4 của kế hoạch bên dưới | ⬜ kế tiếp |
| 6 | Tinh chỉnh, tối ưu CIDEr, hướng tùy chọn: bước 5–7 | ⬜ |
| 7 | Đánh giá tổng hợp và phân tích lỗi | ⬜ |
| 8 | Demo Gradio và hoàn thiện README | ⬜ |

**Việc kế tiếp: Bước 1 của Giai đoạn 5** (hạ tầng so sánh). Chưa có dòng code nào của Giai đoạn 5.

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

### Bước 1 – Hạ tầng so sánh ⬜ (nhánh `feat/experiment-infra`)

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

### Bước 2 – LSTM có attention ⬜ (nhánh `feat/attention-decoder`)

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

### Bước 3 – Encoder mạnh hơn ⬜ (nhánh `feat/strong-encoders`)

- **Đúng hai ứng viên: một mô hình CLIP và một mô hình SigLIP.** Bỏ DINOv2.
- **Giấy phép trọng số:** chỉ dùng trọng số có giấy phép cho phép nghiên cứu, ưu tiên MIT hoặc
  Apache-2.0. Phải xác minh tại nguồn (trang mô hình và file LICENSE của repo gốc) **trước khi
  tải**, rồi ghi vào `docs/MODELS.md`: tên mô hình, nguồn, câu giấy phép trích nguyên văn, ngày
  kiểm tra. Hiện chưa mô hình nào được xác minh.
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

### Bước 4 – Transformer decoder ⬜ (nhánh `feat/transformer-decoder`)

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

### Bước 5 – Tinh chỉnh ⬜ (chưa được duyệt chạy)

Trên cặp encoder – decoder tốt nhất, tối đa 15 run, mỗi run đổi một yếu tố: label smoothing (0 và
0,1), dropout, learning rate, số lớp và độ rộng, mức từ hay âm tiết, `text.min_freq` (đổi thì phải
chạy lại `prepare_data.py`), và cách giải mã (beam, length penalty, chặn lặp). Chủ dự án tự chạy
bằng `scripts/run_sweep.py` với danh sách đã chuẩn bị ở cuối Bước 4.

### Bước 6 – Tối ưu CIDEr kiểu SCST ⬜ (chưa được duyệt chạy)

- Bắt đầu từ checkpoint tốt nhất huấn luyện bằng cross-entropy (`--init-from`), learning rate
  thấp; phần thưởng là CIDEr của câu lấy mẫu trừ CIDEr của câu greedy, tính với câu tham chiếu và
  tần suất văn bản của tập train. Cần thêm đường giải mã lấy mẫu cho decoder và một bản CIDEr
  nhận tần suất văn bản từ ngoài.
- **Tiêu chí đã duyệt:** bản SCST phải qua kiểm tra độ tự nhiên (các chỉ số canh ở Bước 1 và bộ
  50 ảnh val đọc bằng mắt không tệ hơn bản cross-entropy). Nếu không qua, mô hình cuối là bản
  cross-entropy và số SCST chỉ được báo cáo như một phép so sánh.

### Bước 7 – Hướng tùy chọn ⬜ (quyết định sau)

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
  bình ± độ lệch chuẩn. Một chênh lệch chỉ được coi là thật khi vượt hai lần độ lệch chuẩn và
  vượt 0,01 CIDEr.
- Luôn báo kèm số tham số, thời gian huấn luyện, VRAM đỉnh và các chỉ số canh độ tự nhiên.

### Chọn mô hình cuối

1. CIDEr val (beam 3) trung bình ba seed cao nhất, BLEU-4 không kém hơn ngoài mức nhiễu.
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
