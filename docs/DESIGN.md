# Thiết kế hệ thống

Tài liệu này giải thích kiến trúc hiện tại, lý do của từng lựa chọn, các phương án đã cân nhắc
nhưng loại bỏ, và các rủi ro còn lại. Đọc cùng với [ROADMAP.md](ROADMAP.md) (việc tiếp theo) và
[DEVELOPMENT.md](DEVELOPMENT.md) (cách chạy).

## 1. Luồng dữ liệu

```
download_data.py     data/raw/<dataset>/                      file gốc của tác giả
        |            data/processed/<dataset>/annotations.json  định dạng thống nhất
verify_data.py       .../verify_report.json                   kiểm tra ảnh và caption
prepare_data.py      .../<text setting>/captions.json, vocab.json, stats.json
extract_features.py  data/features/<dataset>/<encoder>_<size>/features.npy + index.json
train.py             outputs/<run>/checkpoints/{last,best}.pt, metrics.jsonl, train.log
evaluate.py          outputs/<run>/eval_<split>_beam<k>.json
predict.py           caption cho ảnh bất kỳ
```

Mỗi bước chỉ đọc kết quả của bước trước, nên có thể chạy lại riêng từng bước. Thư mục kết quả
mang tên cấu hình sinh ra nó (`word_minfreq3`, `resnet50_224`), vì vậy đổi cấu hình không ghi đè
kết quả cũ.

## 2. Mã nguồn

| Gói | Vai trò |
|---|---|
| `vic.config` | Đọc YAML, gộp cấu hình dataset, ghi đè `key=value`, quy ước đường dẫn dữ liệu |
| `vic.data` | Tải dữ liệu, adapter dataset, kiểm tra, tiền xử lý, `Dataset` / sampler, cache đặc trưng |
| `vic.text` | Chuẩn hóa, tách từ, từ điển |
| `vic.models` | Encoder ảnh, decoder, giải mã greedy / beam search |
| `vic.metrics` | BLEU, CIDEr-D, METEOR khớp chính xác |
| `vic.engine` | Vòng huấn luyện, checkpoint, đánh giá, suy luận một ảnh |

`scripts/` chỉ là lớp dòng lệnh mỏng gọi vào các gói trên.

## 3. Dữ liệu

- **Định dạng thống nhất.** Mỗi dataset có một adapter chuyển file gốc về cùng một JSON
  (`image_id`, `file_name`, `split`, `captions`). Mọi phần còn lại không biết dataset nào đang
  dùng. Đổi dataset là thêm một file YAML và sửa dòng `dataset_config`.
- **Split.** Dùng split chính thức nếu có (UIT-OpenViIC có đủ train / dev / test). Dataset thiếu
  split thì tách theo ảnh từ train, cố định theo seed.
- **Ảnh lỗi.** File thiếu vài byte cuối vẫn được đọc (bật chế độ chấp nhận ảnh cắt cụt) để
  giữ nguyên split chính thức; ảnh không giải mã được thì bước kiểm tra báo lỗi.

Lựa chọn dataset và giấy phép: xem [DATASETS.md](DATASETS.md).

## 4. Xử lý văn bản

- **Chuẩn hóa:** Unicode NFC, chữ thường, bỏ dấu câu, sửa ký tự dễ nhầm (`Ð` và `Đ`), thống nhất
  cách đặt dấu thanh (`hoà` thành `hòa`). Caption do nhiều người gõ nên hai kiểu đặt dấu lẫn
  nhau; không thống nhất thì cùng một từ thành hai mục từ điển.
- **Tách từ:** mặc định mức từ bằng `underthesea`, âm tiết cùng từ nối bằng `_`. Có thể chuyển
  sang mức âm tiết bằng `text.level=syllable`.
- **Từ điển:** chỉ xây từ train, ngưỡng tần suất 3 (3.482 token, khoảng 1% từ ngoài từ điển).
  Khi sinh câu, `<unk>` bị chặn nên đầu ra không bao giờ chứa `<unk>`.

Phương án đã loại:

| Phương án | Lý do loại |
|---|---|
| VnCoreNLP để tách từ | Cần Java và tải mô hình riêng; khó cài trên Windows hơn mà lợi ích nhỏ với caption ngắn |
| Subword (BPE / SentencePiece, PhoBERT) | Hợp với decoder Transformer hoặc mô hình ngôn ngữ pretrained; để dành cho Giai đoạn 5. Với LSTM và từ điển 3,5 nghìn từ thì chưa cần |
| Chỉ dùng mức âm tiết | Đơn giản nhất nhưng câu dài hơn khoảng 13% và mất ranh giới từ ghép; vẫn giữ làm tùy chọn để so sánh |

## 5. Mô hình baseline

- **Encoder:** ResNet-50 pretrained ImageNet, bỏ lớp pooling và phân loại, đóng băng. Ảnh được
  co về 224 x 224 **không cắt** (để không mất nội dung ở rìa), cho ra lưới 7 x 7 = 49 vùng,
  mỗi vùng 2048 chiều.
- **Decoder:** LSTM một lớp kiểu Show-and-Tell. Trung bình của 49 vùng khởi tạo trạng thái ẩn
  và trạng thái nhớ; mỗi bước nhận embedding của từ trước đó. Khoảng 7,8 triệu tham số.
- **Giao diện decoder chung:** `forward`, `init_state`, `step`, `reorder_state` (mô tả trong
  `src/vic/models/decoder.py`). Vòng huấn luyện, greedy và beam search chỉ dùng bốn hàm này,
  nên decoder attention hay Transformer ở Giai đoạn 5 chỉ cần hiện thực giao diện và đăng ký
  vào `DECODERS`.
- Baseline cố ý **không dùng attention** dù cache đã lưu cả lưới 49 vùng: nó là mốc so sánh
  cho mô hình attention sau này, trên cùng đặc trưng.

Phương án đã loại:

| Phương án | Lý do loại |
|---|---|
| Fine-tune encoder ngay từ đầu | Tốn VRAM và thời gian, dễ quá khớp với 9 nghìn ảnh; để Giai đoạn 6 sau khi decoder đã hội tụ |
| Encoder ViT / CLIP | Thường cho kết quả tốt hơn nhưng làm baseline khó so với các bài báo dùng CNN; cân nhắc ở Giai đoạn 5 |
| Tinh chỉnh mô hình vision-language lớn có sẵn | Mục tiêu dự án là tự xây và hiểu từng thành phần; có thể dùng làm mốc tham chiếu về sau |
| Cắt giữa ảnh (center crop) | Mất chủ thể ở rìa ảnh, trong khi caption mô tả toàn cảnh |

## 6. Cache đặc trưng

Encoder đóng băng nên đặc trưng của mỗi ảnh không đổi giữa các epoch. `extract_features.py`
chạy encoder một lần và lưu lưới đặc trưng ở dạng float16:

- `features.npy`: mảng `(số ảnh, 49, 2048)`, khoảng 2,5 GB cho 13.100 ảnh.
- `index.json`: thứ tự `image_id`, tên encoder, kích thước ảnh. File này được ghi sau cùng,
  nên cache dở dang không bao giờ bị coi là hợp lệ.

Huấn luyện đọc từ cache (`data.use_cached_features: true`): không cần encoder trên GPU, mỗi
epoch chỉ mất vài giây. Hạn chế: không tăng cường ảnh được, và phải trích lại khi đổi encoder
hay kích thước ảnh (cache mới nằm ở thư mục khác, không ghi đè).

Đường chạy trực tiếp vẫn giữ: đặt `data.use_cached_features=false` thì dataset trả ảnh và mô
hình tự chạy encoder; thêm `model.encoder.trainable=true` để fine-tune (encoder dùng learning
rate riêng `train.encoder_lr`, và trọng số encoder khi đó được lưu vào checkpoint).

## 7. Huấn luyện

- AdamW, cosine decay có warmup, cắt gradient, mixed precision (autocast float16 + GradScaler).
- **Checkpoint tự chứa:** cấu hình, từ điển, trọng số decoder, trạng thái optimizer / scheduler
  / scaler, trạng thái sinh số ngẫu nhiên và tiến độ. `evaluate.py` và `predict.py` chỉ cần file
  checkpoint. Ghi qua file tạm rồi đổi tên, nên bị ngắt khi đang lưu không làm hỏng file cũ.
- **Resume chính xác:** thứ tự batch chỉ phụ thuộc `(seed, epoch)`. Khi tiếp tục giữa epoch,
  sampler bỏ qua đúng số batch đã chạy mà không phải đọc lại dữ liệu. Test kiểm tra rằng chạy
  ngắt quãng rồi resume cho trọng số trùng với chạy liền một mạch (trên CPU).
- **Chọn mô hình:** cuối mỗi epoch tính loss trên val và sinh caption greedy cho ảnh val;
  `best.pt` là checkpoint có CIDEr cao nhất. Dừng sớm sau 5 lần đánh giá không cải thiện.
- **Log:** `train.log`, `metrics.jsonl` (mỗi dòng một bản ghi), TensorBoard, `summary.json`.

- **Không ghi đè âm thầm:** chạy mới vào thư mục đã có checkpoint sẽ bị từ chối, trừ khi dùng
  `--resume auto` hoặc `--overwrite`.

Ngân sách VRAM: hệ điều hành chiếm khoảng 1,5 GB trong 8 GB. Số đo khi chạy thử: huấn luyện từ
cache dùng đỉnh 385 MB (batch 128); chạy encoder đóng băng trực tiếp trên ảnh dùng đỉnh 486 MB
(batch 32). Fine-tune encoder sẽ tốn hơn nhiều vì phải giữ gradient của ResNet; khi đó bắt đầu
với batch 16–32 và giảm nếu thiếu bộ nhớ.

Ảnh có vùng trong suốt (PNG) được đặt lên nền trắng trước khi đưa vào encoder; ảnh có thông
tin xoay EXIF được xoay cho đúng chiều.

## 8. Giải mã

- Greedy theo batch (dùng khi validate) và beam search từng ảnh (mặc định beam 3).
- Beam search xếp hạng theo `log_prob / độ_dài ^ 0,7` để không thiên về câu quá ngắn.
- Độ dài tối đa bằng `text.max_length` (30 token).

## 9. Đánh giá

- **Chỉ số chính: BLEU-1..4 và CIDEr-D.** Cả hai được viết lại bằng Python thuần và đã đối
  chiếu với `pycocoevalcap` (sai khác dưới 1e-10 trên dữ liệu ngẫu nhiên); giá trị đối chiếu
  nằm trong `tests/test_metrics.py`. Không cần Java.
- **METEOR chỉ để tham khảo.** METEOR đầy đủ khớp cả gốc từ, từ đồng nghĩa và cụm diễn đạt lại,
  nhưng các tài nguyên đó không có cho tiếng Việt. Bản ở đây chỉ khớp từ giống hệt nhau, nên nó
  đo gần như cùng thứ với precision / recall unigram cộng một hệ số phạt trật tự từ. Không dùng
  METEOR để chọn mô hình hay kết luận mô hình nào tốt hơn.
- **Tính trên âm tiết.** Câu sinh ra được tách lại thành âm tiết (`sân_bóng` thành `sân bóng`),
  câu tham chiếu là caption đã chuẩn hóa. Nhờ vậy điểm số không phụ thuộc bộ tách từ và so
  sánh được giữa mô hình mức từ với mô hình mức âm tiết.
- CIDEr báo theo thang gốc 0–10 của bản tham chiếu (nhiều bài báo nhân 100).

## 10. Tái lập

Seed cố định cho Python, NumPy, PyTorch; mọi tham số nằm trong YAML; cấu hình đã gộp được lưu
vào thư mục chạy và vào checkpoint. Trên GPU, cuDNN vẫn có thể cho sai khác nhỏ giữa hai lần
chạy; đặt `train.deterministic=true` để giảm (chậm hơn một chút).

## 11. Rủi ro và hạn chế

| Rủi ro | Ảnh hưởng | Cách giảm |
|---|---|---|
| Dataset không có giấy phép chính thức, chỉ có tuyên bố "for research purposes" | Không dùng được cho mục đích thương mại; điều khoản có thể thay đổi | Chỉ dùng nghiên cứu, không phân phối lại, ghi nguồn; theo dõi repo của tác giả |
| Link Google Drive của tác giả có thể hết hạn mức hoặc bị gỡ | Không tải lại được dữ liệu | Giữ bản sao cục bộ của `data/raw`; script báo lỗi rõ và cho phép đặt file tay |
| Chỉ 9 nghìn ảnh train | Quá khớp, từ vựng hẹp | Dropout 0,5, dừng sớm theo CIDEr; Giai đoạn 6 thử label smoothing và fine-tune |
| Bộ tách từ sai | Từ điển nhiễu | Điểm số tính trên âm tiết nên không bị ảnh hưởng trực tiếp; có thể chuyển sang mức âm tiết |
| So sánh với bài báo | Cách tách từ và cách tính điểm khác nhau nên số không so trực tiếp được | Ghi rõ cách tính; nếu cần so sánh chặt thì chạy lại bằng công cụ của bài báo |
| Cache không cho tăng cường ảnh | Baseline kém bền với biến đổi ảnh | Dùng đường chạy encoder trực tiếp khi fine-tune |
| Huấn luyện bằng teacher forcing | Lệch giữa lúc học và lúc sinh câu | Beam search; Giai đoạn 6 cân nhắc tối ưu trực tiếp CIDEr |

## 12. Hướng mở rộng

Chi tiết từng giai đoạn nằm trong [ROADMAP.md](ROADMAP.md). Các điểm nối đã có sẵn:

- Lưới đặc trưng 49 vùng trong cache: dùng ngay cho decoder attention hoặc Transformer.
- `DECODERS` trong `src/vic/models/decoder.py`: nơi đăng ký decoder mới.
- `BACKBONES` trong `src/vic/models/encoder.py`: nơi thêm encoder mới.
- `CaptionPredictor` trong `src/vic/engine/inference.py`: lớp demo Gradio sẽ gọi.
