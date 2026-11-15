# Trọng số pretrained

Mọi trọng số pretrained dùng trong dự án phải có giấy phép cho phép nghiên cứu (ưu tiên MIT hoặc
Apache-2.0) và phải được **xác minh tại nguồn trước khi tải**. Trọng số, cache đặc trưng và
checkpoint không bao giờ được commit hay công bố; chúng nằm trong cache của thư viện (ngoài
repo) và trong `data/`, `outputs/` (bị git bỏ qua).

## Bảng tổng hợp

| Tên trong cấu hình | Mô hình | Nguồn tải | Giấy phép | Ngày kiểm tra |
|---|---|---|---|---|
| `resnet50`, `resnet101` | ResNet, ImageNet-1k (`IMAGENET1K_V2`) | `torchvision` | BSD-3-Clause (thư viện `torchvision`) | 15/11/2026 |
| `clip_vit_b16` | CLIP ViT-B/16 (OpenAI) | Hugging Face `openai/clip-vit-base-patch16`, commit `57c216476eefef5ab752ec549e440a49ae4ae5f3` | MIT (repo gốc `openai/CLIP`; trang Hugging Face không có thẻ giấy phép) | 15/11/2026 |
| `siglip_b16` | SigLIP ViT-B/16, 224 px (Google) | Hugging Face `google/siglip-base-patch16-224`, commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Apache-2.0 | 15/11/2026 |

Mã commit được ghim trong `src/vic/models/encoder.py`, nên lần tải nào cũng lấy đúng các file
đã kiểm tra.

## Chi tiết xác minh

### CLIP ViT-B/16 (`clip_vit_b16`)

Kiểm tra tại nguồn ngày 15/11/2026 bằng cách đọc trực tiếp file `LICENSE` và `model-card.md`
của repo gốc, cùng metadata của trang Hugging Face (API
`https://huggingface.co/api/models/openai/clip-vit-base-patch16`), không dựa vào tài liệu thứ
cấp.

- **Trang Hugging Face không có thẻ giấy phép.** Trang
  <https://huggingface.co/openai/clip-vit-base-patch16> do chính tổ chức `openai` đăng, nhưng
  metadata của nó không có mục `license` (không có thẻ `license:*`, `cardData` không có khóa
  `license`) và nội dung trang không ghi giấy phép.
- **Giấy phép lấy từ repo gốc `openai/CLIP`** (<https://github.com/openai/CLIP>), nơi OpenAI
  phát hành mô hình và trọng số. File `LICENSE`, trích nguyên văn: "MIT License" / "Copyright
  (c) 2021 OpenAI".
- **Phạm vi sử dụng do model card nêu** (`model-card.md` của repo `openai/CLIP`; trang Hugging
  Face có cùng nội dung), trích nguyên văn:
  - "The model is intended as a research output for research communities."
  - "**Any** deployed use case of the model - whether commercial or not - is currently out of
    scope."

  Dự án chỉ dùng mô hình để trích đặc trưng phục vụ nghiên cứu, không triển khai, nên nằm trong
  phạm vi này. Hệ quả cho dự án: mô hình dùng đặc trưng CLIP chỉ được dùng cho nghiên cứu và
  demo chạy cục bộ, không đưa vào sản phẩm (trùng với giới hạn đã có của bộ dữ liệu).
- **File tải về** (commit `57c216476eefef5ab752ec549e440a49ae4ae5f3`): `config.json`,
  `preprocessor_config.json`, `pytorch_model.bin`. Repo chỉ có trọng số dạng `.bin` (pickle),
  không có `.safetensors`. `transformers` 5.20.2 nạp file này bằng `torch.load(...,
  weights_only=True)` (mặc định của `from_pretrained`), tức chỉ đọc tensor, không chạy mã trong
  file.
- **Cách dùng:** chỉ nhánh ảnh (`CLIPVisionModel`). Đặc trưng là 196 token patch (lưới 14 × 14,
  768 chiều) của **lớp áp chót**, bỏ token `[CLS]`: lớp cuối của CLIP chỉ được huấn luyện qua
  token `[CLS]`, nên token patch của lớp áp chót mang nhiều thông tin vùng hơn.

### SigLIP ViT-B/16 224 px (`siglip_b16`)

Kiểm tra tại nguồn ngày 15/11/2026: đọc `README.md` (phần metadata) và metadata API của trang
Hugging Face, và file `LICENSE` của repo gốc.

- **Trang mô hình:** <https://huggingface.co/google/siglip-base-patch16-224>, do tổ chức
  `google` đăng. Metadata, trích nguyên văn: `license: apache-2.0` (thẻ API
  `license:apache-2.0`).
- **Repo gốc:** <https://github.com/google-research/big_vision>. File `LICENSE`, ba dòng đầu
  trích nguyên văn: "Apache License" / "Version 2.0, January 2004" /
  "http://www.apache.org/licenses/".
- **File tải về** (commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`): `config.json`,
  `preprocessor_config.json`, `model.safetensors`.
- **Cách dùng:** chỉ nhánh ảnh (`SiglipVisionModel`). Đặc trưng là 196 token patch (lưới
  14 × 14, 768 chiều) của lớp cuối; SigLIP không có token `[CLS]`.

### Kiểm tra cache đặc trưng (15/11/2026)

Hai cache (`data/features/uit_openviic/clip_vit_b16_224`, `.../siglip_b16_224`) được trích bằng
mã của nhánh `feat/strong-encoders` xuất ra một thư mục tạm, trước khi mã đó được commit. Để
chắc cache đúng là sản phẩm của mã đã commit, 16 ảnh val (chọn bằng `random.Random(42)`) được
trích lại bằng mã đã commit (commit `85339fd`) và so với cache:

| Phép so | Sai khác tuyệt đối lớn nhất | Sai số L2 tương đối | Cosine nhỏ nhất (theo vùng) |
|---|---:|---:|---:|
| SigLIP: cache so với trích lại (lô 16 ảnh) | 0 | 0 | 1,000000 |
| CLIP: cache so với trích lại (lô 16 ảnh) | 2,09 | 5,7e-3 | 0,943 |
| CLIP: cache so với trích lại **trong đúng các lô 64 ảnh đã dùng khi tạo cache** | 0 | 0 | 1,000000 |
| CLIP: trích lại hai lần liên tiếp (lô 16 ảnh) | 0 | 0 | 1,000000 |
| CLIP: cache so với tính bằng float32 | 1,60 | 5,5e-3 | 0,954 |
| CLIP: trích lại (lô 16 ảnh) so với tính bằng float32 | 2,21 | 5,1e-3 | 0,984 |

Kết luận (chủ dự án đã chấp nhận):

- Cả hai cache **trùng từng bit** với mã đã commit khi trích cùng cách chia lô.
- Đặc trưng CLIP tính ở float16 (autocast) **phụ thuộc cỡ lô**: GPU chọn kernel khác nhau theo
  cỡ lô, và lớp áp chót của CLIP có vài chiều trị tuyệt đối gần 90 nên rất nhạy với sai số làm
  tròn. Cache và bản lô 16 cách bản float32 ngang nhau (khoảng 0,5% theo L2). SigLIP không bị
  ảnh hưởng.
- Hệ quả: khi `predict.py` chạy CLIP trên từng ảnh, đặc trưng lệch khoảng 0,5% so với cache đã
  dùng để huấn luyện. Việc cần làm trước khi chốt mô hình cuối: cho encoder ViT chạy float32 khi
  trích cache và khi suy luận (xem danh sách việc trước Bước 5 trong `ROADMAP.md`).

### ResNet (`resnet50`, `resnet101`)

Trọng số ImageNet-1k do `torchvision` phân phối (`ResNet50_Weights.IMAGENET1K_V2`). Thư viện
`torchvision` dùng giấy phép BSD-3-Clause. Trọng số được huấn luyện trên ImageNet-1k, bộ dữ liệu
chỉ cho phép dùng cho nghiên cứu phi thương mại; dự án này cũng chỉ phục vụ nghiên cứu.

## Đã cân nhắc nhưng không dùng

- **DINOv2:** bị loại theo quyết định của chủ dự án (không được học cùng ngôn ngữ).
- **SigLIP 2** (`google/siglip2-base-patch16-224`, thẻ `license: apache-2.0`, kiểm tra
  15/11/2026): cùng kích thước với SigLIP và mới hơn, nhưng kế hoạch đã duyệt chỉ cho đúng hai
  ứng viên. Có thể đề xuất ở lượt sau; chưa tải.

## Thêm mô hình mới

1. Mở trang mô hình và file `LICENSE` của repo gốc; chép nguyên văn câu giấy phép vào file này,
   kèm ngày kiểm tra. Không dựa vào trí nhớ hay tài liệu thứ cấp.
2. Ghim commit của nguồn tải trong `src/vic/models/encoder.py`.
3. Sau đó mới tải và trích đặc trưng.
