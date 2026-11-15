# Trọng số pretrained

Mọi trọng số pretrained dùng trong dự án phải có giấy phép cho phép nghiên cứu (ưu tiên MIT hoặc
Apache-2.0) và phải được **xác minh tại nguồn trước khi tải**. Trọng số, cache đặc trưng và
checkpoint không bao giờ được commit hay công bố; chúng nằm trong cache của thư viện (ngoài
repo) và trong `data/`, `outputs/` (bị git bỏ qua).

## Bảng tổng hợp

| Tên trong cấu hình | Mô hình | Nguồn tải | Giấy phép | Ngày kiểm tra |
|---|---|---|---|---|
| `resnet50`, `resnet101` | ResNet, ImageNet-1k (`IMAGENET1K_V2`) | `torchvision` | BSD-3-Clause (thư viện `torchvision`) | 15/11/2026 |
| `clip_vit_b16` | CLIP ViT-B/16 (OpenAI) | Hugging Face `openai/clip-vit-base-patch16`, commit `57c216476eefef5ab752ec549e440a49ae4ae5f3` | MIT (repo gốc `openai/CLIP`) | 15/11/2026 |
| `siglip_b16` | SigLIP ViT-B/16, 224 px (Google) | Hugging Face `google/siglip-base-patch16-224`, commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Apache-2.0 | 15/11/2026 |

Mã commit được ghim trong `src/vic/models/encoder.py`, nên lần tải nào cũng lấy đúng các file
đã kiểm tra.

## Chi tiết xác minh

### CLIP ViT-B/16 (`clip_vit_b16`)

- **Repo gốc:** <https://github.com/openai/CLIP>. File `LICENSE` của repo, trích nguyên văn:
  "MIT License" / "Copyright (c) 2021 OpenAI".
- **Trang mô hình:** <https://huggingface.co/openai/clip-vit-base-patch16>, do chính tổ chức
  `openai` đăng. Trang này **không gắn thẻ giấy phép** (metadata không có mục `license`); giấy
  phép áp dụng là giấy phép MIT của repo gốc, nơi trọng số được phát hành.
- **Phạm vi sử dụng do tác giả nêu** (model card, trích nguyên văn): "The model is intended as a
  research output for research communities." và "**Any** deployed use case of the model -
  whether commercial or not - is currently out of scope." Dự án này chỉ dùng mô hình để trích
  đặc trưng cho nghiên cứu, không triển khai, nên nằm trong phạm vi đó.
- **File tải về:** `config.json`, `preprocessor_config.json`, `pytorch_model.bin`. Repo chỉ có
  trọng số dạng `.bin` (pickle), không có `.safetensors`; thư viện nạp file này ở chế độ chỉ
  đọc tensor (`weights_only`).
- **Cách dùng:** chỉ nhánh ảnh (`CLIPVisionModel`). Đặc trưng là 196 token patch (lưới 14 × 14,
  768 chiều) của **lớp áp chót**, bỏ token `[CLS]`: lớp cuối của CLIP chỉ được huấn luyện qua
  token `[CLS]`, nên token patch của lớp áp chót mang nhiều thông tin vùng hơn.

### SigLIP ViT-B/16 224 px (`siglip_b16`)

- **Trang mô hình:** <https://huggingface.co/google/siglip-base-patch16-224>, do tổ chức
  `google` đăng. Thẻ giấy phép trong metadata, trích nguyên văn: `license: apache-2.0`.
- **Repo gốc:** <https://github.com/google-research/big_vision>. File `LICENSE`, ba dòng đầu
  trích nguyên văn: "Apache License" / "Version 2.0, January 2004" /
  "http://www.apache.org/licenses/".
- **File tải về:** `config.json`, `preprocessor_config.json`, `model.safetensors`.
- **Cách dùng:** chỉ nhánh ảnh (`SiglipVisionModel`). Đặc trưng là 196 token patch (lưới
  14 × 14, 768 chiều) của lớp cuối; SigLIP không có token `[CLS]`.

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
