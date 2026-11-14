# Kết quả

Tài liệu này ghi kết quả chính thức của từng mô hình. Hiện có một mô hình: baseline.

## Baseline: ResNet-50 đóng băng + LSTM

| | |
|---|---|
| Cấu hình | [`configs/baseline_lstm.yaml`](../configs/baseline_lstm.yaml), seed 42 |
| Dữ liệu | UIT-OpenViIC: train 9.088 ảnh / 41.236 caption, val 2.011 ảnh, test 2.001 ảnh |
| Mô hình | 7,77 triệu tham số huấn luyện (decoder); encoder đóng băng, đặc trưng đã cache |
| Huấn luyện | 30 epoch (9.690 bước), batch 128, mixed precision, khoảng 8 phút trên RTX 5070 Laptop |
| VRAM đỉnh | 385 MB cấp phát |
| Ngày chạy | 14/11/2026 |

### Cách tái lập

```powershell
python scripts/train.py --config configs/baseline_lstm.yaml
python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split val test --beam-size 1 3 5
python scripts/summarize_run.py --run outputs/baseline_lstm --output outputs/baseline_lstm/report.md
```

Các bảng và 10 ví dụ bên dưới lấy nguyên từ `report.md` do lệnh cuối sinh ra.

### Điểm số

Checkpoint được đánh giá là `best.pt` (epoch 28). BLEU và METEOR nhân 100; CIDEr theo thang gốc
0–10. Mọi điểm tính trên âm tiết. **BLEU và CIDEr là chỉ số chính; METEOR chỉ khớp từ giống hệt
nhau nên chỉ để tham khảo.**

| Tập | Giải mã | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr |
|---|---|---:|---:|---:|---:|---:|---:|
| val | greedy | 68.11 | 51.11 | 38.01 | 28.25 | 22.41 | 0.4232 |
| val | beam 3 | 68.06 | 51.43 | 38.54 | 28.74 | 22.10 | 0.4274 |
| val | beam 5 | 67.28 | 50.44 | 37.75 | 28.13 | 21.70 | 0.4161 |
| test | greedy | 67.16 | 51.10 | 38.14 | 28.38 | 22.12 | 0.4531 |
| test | beam 3 | 66.26 | 50.36 | 37.65 | 27.91 | 21.76 | 0.4479 |
| test | beam 5 | 65.23 | 49.18 | 36.75 | 27.27 | 21.33 | 0.4314 |

Nhận xét về cách giải mã (chỉ dựa trên val):

- Beam 3 nhỉnh hơn greedy một chút trên val (CIDEr 0,4274 so với 0,4232; BLEU-4 28,74 so với
  28,25). Mức chênh này nhỏ và mới đo với một seed, chưa đủ để coi là khác biệt chắc chắn.
- Beam 5 kém hơn cả hai trên mọi chỉ số chính. Beam rộng hơn kéo mô hình về các câu chung
  chung, an toàn.
- Beam 3 được giữ làm mặc định cho `evaluate.py` và `predict.py`.

Điểm trên val và test không so trực tiếp với nhau được: hai tập có câu tham chiếu khác nhau về
độ dài, và CIDEr tính trọng số từ theo từng tập.

### Epoch tốt nhất và dấu hiệu overfit

- **Epoch tốt nhất: 28** (CIDEr val 0,4232, giải mã greedy). Huấn luyện chạy đủ 30 epoch, không
  dừng sớm: CIDEr vẫn nhích lên từng chút cho tới gần cuối.
- **Loss val đi ngang từ khoảng epoch 16** (2,754) đến cuối (2,746), trong khi loss train tiếp
  tục giảm từ 2,52 xuống 2,40. Khoảng cách train – val nới rộng dần. Loss train đo khi đang bật
  dropout 0,5 nên khoảng cách thật còn lớn hơn con số trong bảng.
- **Loss val không tăng trở lại và CIDEr không giảm**, nên đây là overfit nhẹ: mô hình ngừng
  tổng quát hóa thêm chứ chưa xấu đi. Learning rate giảm theo cosine về gần 0 ở các epoch cuối
  cũng góp phần làm các đường cong phẳng lại.
- CIDEr val dao động khoảng 0,01–0,02 giữa các epoch liền nhau (ví dụ epoch 12: 0,4065, epoch
  13: 0,3875). Chênh lệch nhỏ hơn mức này giữa hai mô hình không nên coi là có ý nghĩa.

Kết luận: huấn luyện thêm epoch không giúp gì. Giới hạn hiện tại nằm ở mô hình và đặc trưng ảnh,
không nằm ở thời gian huấn luyện.

| Epoch | Loss train | Loss val | BLEU-4 val | CIDEr val | |
|---:|---:|---:|---:|---:|---|
| 1 | 5.018 | 3.420 | 20.26 | 0.2248 |  |
| 2 | 3.389 | 3.064 | 23.75 | 0.2747 |  |
| 3 | 3.107 | 2.949 | 23.05 | 0.2964 |  |
| 4 | 2.965 | 2.890 | 26.03 | 0.3451 |  |
| 5 | 2.880 | 2.855 | 26.94 | 0.3656 |  |
| 6 | 2.809 | 2.830 | 26.05 | 0.3547 |  |
| 7 | 2.765 | 2.806 | 26.59 | 0.3686 |  |
| 8 | 2.718 | 2.797 | 27.82 | 0.3834 |  |
| 9 | 2.683 | 2.791 | 26.07 | 0.3733 |  |
| 10 | 2.649 | 2.780 | 26.88 | 0.3820 |  |
| 11 | 2.622 | 2.773 | 27.30 | 0.3929 |  |
| 12 | 2.594 | 2.766 | 28.43 | 0.4065 |  |
| 13 | 2.571 | 2.764 | 27.06 | 0.3875 |  |
| 14 | 2.552 | 2.762 | 27.74 | 0.3979 |  |
| 15 | 2.530 | 2.756 | 27.87 | 0.3989 |  |
| 16 | 2.516 | 2.754 | 28.24 | 0.4165 |  |
| 17 | 2.495 | 2.754 | 28.22 | 0.4098 |  |
| 18 | 2.481 | 2.752 | 27.74 | 0.4168 |  |
| 19 | 2.466 | 2.753 | 28.44 | 0.4207 |  |
| 20 | 2.452 | 2.750 | 27.54 | 0.4097 |  |
| 21 | 2.440 | 2.749 | 28.00 | 0.4215 |  |
| 22 | 2.432 | 2.750 | 28.19 | 0.4192 |  |
| 23 | 2.419 | 2.747 | 27.78 | 0.4145 |  |
| 24 | 2.419 | 2.747 | 28.15 | 0.4195 |  |
| 25 | 2.410 | 2.747 | 28.01 | 0.4227 |  |
| 26 | 2.406 | 2.748 | 28.35 | 0.4216 |  |
| 27 | 2.401 | 2.746 | 28.32 | 0.4227 |  |
| 28 | 2.395 | 2.746 | 28.25 | 0.4232 | tốt nhất |
| 29 | 2.397 | 2.746 | 28.22 | 0.4212 |  |
| 30 | 2.398 | 2.746 | 28.21 | 0.4212 |  |

### Chẩn đoán trên val (beam 3)

- Số ảnh: 2011.
- Độ dài trung bình (âm tiết): câu sinh ra 14.3, câu tham chiếu 18.6.
- Số câu sinh ra khác nhau: 975 trên 2011.
- Số âm tiết khác nhau được dùng: 357, trong khi câu tham chiếu dùng 2358.
- Câu có một âm tiết lặp liền nhau: 0 (0.0%).
- Ba câu xuất hiện nhiều nhất:
  - 37 lần: "nhiều người đang di chuyển trên một đoạn đường với hai bên là những hàng cây xanh"
  - 32 lần: "một cửa hàng có biển hiệu màu đen và dòng chữ màu trắng"
  - 21 lần: "nhiều người đang đứng trước một quầy hàng trong siêu thị"

Ba con số đáng chú ý nhất:

- **Từ vựng nghèo:** mô hình chỉ dùng 357 âm tiết khác nhau, câu tham chiếu dùng 2.358.
- **Câu lặp giữa các ảnh:** 2.011 ảnh chỉ nhận 975 câu khác nhau; có câu được dùng cho 37 ảnh.
- **Câu ngắn hơn:** 14,3 âm tiết so với 18,6 của câu tham chiếu.

### 10 ảnh test chọn ngẫu nhiên (seed 42, beam 3)

Chọn bằng `random.Random(42).sample(...)` trên danh sách mã ảnh test đã sắp xếp. Điểm CIDEr là
của riêng ảnh đó và được tính trên toàn bộ câu tham chiếu của ảnh. Ảnh không nằm trong repo; mã
ảnh là tên file trong bộ dữ liệu.

Mỗi ảnh chỉ trích **một** câu tham chiếu (câu đầu tiên trong các câu của ảnh) để minh họa. Câu
tham chiếu thuộc bộ dữ liệu **UIT-OpenViIC** (Bui, Nguyen, Nguyen, 2026;
https://github.com/nghiangh/UIT-OpenViIC-dataset), được trích ở đây cho mục đích nghiên cứu phi
thương mại. Repo này không phân phối bộ dữ liệu.

**Ảnh `00000009308`** (CIDEr 0.74)

- Sinh ra: *một người đàn ông mặc áo màu xanh đang đứng trước một tấm bảng màu đỏ*
- Tham chiếu: một người đàn ông mặc áo cam đội mũ đen đang đứng trước những rào chắn màu đỏ có dán những tờ giấy màu trắng chữ đen

**Ảnh `00000001046`** (CIDEr 0.16)

- Sinh ra: *một tòa nhà màu vàng có nhiều hơn một tầng có nhiều cửa sổ và ở trước có một chiếc xe máy đang đậu*
- Tham chiếu: phía sau căn nhà màu vàng là một căn nhà cao tầng có cửa kính

**Ảnh `00000000151`** (CIDEr 0.18)

- Sinh ra: *một người đàn ông mặc áo trắng đang đi trên đường*
- Tham chiếu: ba người chạy xe đạp giữa con đường đầy lá vàng và hai bên là hàng cây trải dài

**Ảnh `00000009817`** (CIDEr 0.29)

- Sinh ra: *một con đường có nhiều tòa nhà cao tầng và có nhiều cây xanh*
- Tham chiếu: có vài người đang di chuyển trên con đường mà bên phải là nhiều ngôi nhà có treo đèn ở ban công nằm san sát nhau

**Ảnh `00000005151`** (CIDEr 0.76)

- Sinh ra: *một người phụ nữ mặc áo đỏ đang cầm một hộp sữa trên tay*
- Tham chiếu: một cô gái mặc áo đỏ đứng trước một gian hàng có đèn vàng

**Ảnh `00000005089`** (CIDEr 0.44)

- Sinh ra: *một cửa hàng có biển hiệu màu đen và dòng chữ màu vàng*
- Tham chiếu: một cửa hàng bảng hiệu đen nền vàng đang đóng một cửa mở một cửa

**Ảnh `00000005045`** (CIDEr 0.10)

- Sinh ra: *một tấm bảng hiệu nền màu đỏ với nhiều dòng chữ màu vàng*
- Tham chiếu: nhiều người đang đứng thành hàng xem múa lân

**Ảnh `00000001103`** (CIDEr 0.96)

- Sinh ra: *một người đàn ông mặc áo màu xanh đang đứng trước một cửa hàng*
- Tham chiếu: một người đàn ông ngồi trên chiếc xe đỏ đậu trước một cửa hàng dùng cửa cuốn

**Ảnh `00000009807`** (CIDEr 0.46)

- Sinh ra: *một bức tranh vẽ một con phố với nhiều ngôi nhà được sơn màu vàng*
- Tham chiếu: có một cái máy lạnh màu trắng nằm phía trên bên trái của một bức tranh được lồng khung treo cạnh bộ công tắc màu trắng

**Ảnh `00000001027`** (CIDEr 0.24)

- Sinh ra: *nhiều người đang đứng trước một quầy hàng bán đồ trang trí giáng sinh*
- Tham chiếu: nhiều người mặc áo trắng có mang cà vạt đang đứng trước một kệ nhiều sách

### Các kiểu lỗi thường gặp

Nhận xét dưới đây rút ra từ tập val (số liệu chẩn đoán ở trên và 12 ảnh val có CIDEr thấp nhất),
đối chiếu thêm với 10 ảnh test ngẫu nhiên. Việc xem lỗi để định hướng cải tiến chỉ làm trên val.

1. **Câu khuôn mẫu, chung chung.** Mô hình hay rơi về vài mẫu câu quen thuộc như "nhiều người
   đang đứng trước một quầy hàng" hay "một cửa hàng có biển hiệu màu đen và dòng chữ màu trắng",
   gần như bất kể ảnh. Đây là hệ quả trực tiếp của việc decoder chỉ nhận một vector trung bình
   của cả ảnh.
2. **Bịa ra người.** Với ảnh không có người (nồi lẩu, bộ xương khủng long trong bảo tàng, bể
   cá), mô hình vẫn sinh "một người phụ nữ mặc áo màu trắng đang ngồi…" hoặc "một người đàn ông
   đang đứng trước…". Người là chủ thể phổ biến nhất trong dữ liệu nên thành lựa chọn mặc định.
3. **Sai thuộc tính.** Sai màu (áo cam thành áo xanh, ảnh `00000009308`), sai số lượng (ba người
   đi xe đạp thành "một người đàn ông… đang đi trên đường", ảnh `00000000151`), sai hành động
   (ngồi trên xe máy thành "đang đứng", ảnh `00000001103`).
4. **Bỏ sót vật thể hiếm, thay bằng khái niệm phổ biến.** Múa lân, kệ sách, cây cầu, bùng binh
   không xuất hiện trong câu sinh ra; thay vào đó là "quầy hàng", "tấm bảng hiệu", "tòa nhà".
   Ảnh `00000005045` (múa lân trước cửa hàng) chỉ được mô tả là "một tấm bảng hiệu nền màu đỏ".
5. **Thêm chi tiết cụ thể nhưng sai.** "quầy hàng bán đồ trang trí giáng sinh" cho một kệ sách
   (ảnh `00000001027`), "một tô phở bò có nhiều thịt và hành" cho một bát nước có trứng.
6. **Lặp cụm từ trong câu.** "có nhiều hơn một tầng có nhiều hơn một tầng", "và hành và hành".
   Chỉ số "âm tiết lặp liền nhau" ở trên bằng 0 vì nó không bắt được kiểu lặp cả cụm này.
7. **Loại ảnh lạ.** Ảnh ghép nhiều khung, ảnh chụp từ trên cao, ảnh chụp ban đêm thường bị mô tả
   sai hoàn toàn.

Một phần điểm thấp không phải lỗi của mô hình: các câu tham chiếu của cùng một ảnh thường mô tả
những vật khác nhau (người này tả cái ô ở góc phải, người kia tả cả con đường), nên một câu đúng
vẫn có thể trùng rất ít n-gram với phần lớn câu tham chiếu.

### Giới hạn của kết quả này

- Chỉ một seed; chưa có ước lượng độ dao động giữa các lần chạy.
- Điểm số không so trực tiếp với các bài báo: cách tách từ và công cụ tính điểm có thể khác.
- Điểm test của baseline được báo cáo ở đây làm mốc. Từ nay mọi lựa chọn (kiến trúc, siêu tham
  số, cách giải mã) chỉ dựa trên val; tập test chỉ dùng thêm một lần cho mô hình cuối cùng.
