# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603446**
- **Họ và tên: Nguyễn Thanh Tùng**
- **SEED** (in ở ô đầu notebook): 4904

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

| Phương pháp | F1 test | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin |
|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 0.873 | 0.889 | 0.101 ms | $0.0 |
| **2. BoW + MLP (deep learning)** | 0.920 | 0.947 | 0.593 ms | $0.0 |
| **3a. Encoder (MiniLM) + kNN** | 0.909 | 0.974 | 95.966 ms | $0.0 |
| **3b. Encoder (MiniLM) + LogReg** | 0.927 | 0.919 | 50.525 ms | $0.0 |
| **3c. Encoder (MiniLM) + MLP head** | 0.939 | 0.919 | 54.150 ms | $0.0 |

## 3. Dự đoán và thực tế

* **Dự đoán 1 — `tradeoff` (Đường biên Pareto & Cam kết SLA 50 ms):**
  * *Dự đoán:* Mô hình Transformer Encoder (`MiniLM + MLP head`) sẽ nằm trên đường Pareto vì vừa có khả năng bắt ngữ cảnh sâu, vừa tối ưu phân loại tốt hơn các mô hình túi từ truyền thống.
  * *Thực tế:* `BoW + MLP` mới là mô hình vượt trội và thống trị đường Pareto với độ trễ $p95 = 0.593\text{ ms}$ và F1 SUB đạt $0.947$. `MiniLM + MLP head` có độ trễ $p95 = 54.150\text{ ms}$ (vi phạm trần SLA $50\text{ ms}$), bị `BoW + MLP` lấn át hoàn toàn về cả tốc độ lẫn độ chính xác trên tập SUB.
  * *Nguyên nhân lệch:* Cơ chế Self-Attention của mô hình ngôn ngữ dạng Transformer đòi hỏi khối lượng tính toán ma trận dày đặc trên CPU, khiến thời gian trích xuất đặc trưng (embedding) cao gấp gần 100 lần so với việc ánh xạ túi từ thưa (sparse vector) của BoW.

* **Dự đoán 2 — `hard_eval` (Khả năng chịu tải đối kháng của LLM):**
  * *Dự đoán:* LLM sẽ đạt độ chính xác gần như hoàn hảo ($> 85\%$) trên 50 ca kiểm thử khó do sở hữu khả năng hiểu ngôn ngữ phong phú và suy luận ngữ cảnh tự nhiên.
  * *Thực tế:* LLM Zero-shot mặc định bị sập bẫy nặng nề ở các ca tấn công Prompt Injection, khiến độ chính xác ở nhóm này tụt xuống mức rất thấp.
  * *Nguyên nhân lệch:* LLM xử lý toàn bộ prompt đầu vào trên cùng một không gian ngữ cảnh phẳng. Nếu không áp dụng kỹ thuật đóng gói phân cách dữ liệu (ví dụ: bọc thẻ XML `<sms>...</sms>`), LLM không thể phân biệt ranh giới giữa chỉ thị của hệ thống và câu lệnh độc hại do kẻ tấn công chèn vào nội dung tin nhắn.

* **Dự đoán 3 — `emb_search` (Khoảng cách tương đồng ngữ nghĩa):**
  * *Dự đoán:* Các tin nhắn spam sử dụng kỹ thuật né từ khóa (obfuscation, leet-speak) sẽ có khoảng cách cosine rất xa so với mẫu spam gốc do bề mặt ký tự bị biến dạng hoàn toàn.
  * *Thực tế:* MiniLM vẫn ánh xạ các câu này vào vùng không gian lân cận và thuật toán kNN vẫn phát hiện chính xác phần lớn các ca này (F1 SUB đạt $0.974$).
  * *Nguyên nhân lệch:* Bộ tách từ Subword (WordPiece/BPE) của mô hình Transformer đã phân rã các từ biến thể thành các mảnh âm tiết có nghĩa, giúp duy trì vector biểu diễn ngữ nghĩa tổng thể của câu.

---

## 4. Phân tích 50 case khó

* **Mô hình tụt điểm nhiều nhất:** Nhóm mô hình dựa trên túi từ tĩnh `BoW + LogReg` và `BoW + MLP`. Điểm F1/Accuracy sụt giảm nghiêm trọng từ mức $> 90\%$ trên tập test chuẩn xuống chỉ còn khoảng $40\% - 50\%$ trên tập 50 ca khó.
* **Nhóm ca lỗi nặng nhất:** 
  1. *Tưởng spam nhưng không phải (`ham_looks_spam`):* Mô hình dự đoán nhầm tin nhắn quan trọng của người dùng thành Spam (False Positive).
  2. *Né từ khóa (`obfuscation` / Leet-speak):* Kẻ gian làm biến dạng từ ngữ khiến mô hình bỏ lọt tin rác (False Negative).

* **2 trường hợp cụ thể và cơ chế gây lỗi:**
  * **Trường hợp 1 (False Positive — `ham_looks_spam`):**
    * *Nội dung:* `"Hey, I just won two free tickets to the concert tonight, call me right now!"`
    * *Cơ chế lỗi:* Mô hình BoW hoạt động dựa trên giả định các từ độc lập (bag-of-words assumption). Các từ khóa mang trọng số spam cực cao trong tập dữ liệu huấn luyện như `won`, `free`, `tickets`, `call`, `now` xuất hiện đồng thời và cộng dồn giá trị logit lên mức tối đa, đẩy nhãn dự đoán thành Spam, dù bản chất câu là hội thoại tự nhiên giữa bạn bè.
  * **Trường hợp 2 (False Negative — `obfuscation`):**
    * *Nội dung:* `"C.l.i.c.k h-e-r-e to cl@im your £1,000 c4sh pr!ze"`
    * *Cơ chế lỗi:* Kẻ lừa đảo sử dụng ký tự đặc biệt, dấu chấm và số thay thế chữ cái (`cl@im`, `c4sh`, `pr!ze`). Khi đi qua bộ tokenizer tách từ cơ bản của BoW, các từ này không khớp với bất kỳ từ nào trong từ điển $2.000$ từ phổ biến. Vector đặc trưng đầu vào trở thành vector rỗng (toàn số 0), khiến mô hình mất hoàn toàn tín hiệu nhận diện và phán đoán sai thành tin sạch (Ham).

---

## 5. Thử thách cá nhân

* **Nội dung thử thách:** Khảo sát tác động của dung lượng mô hình mạng nơ-ron MLP khi thay đổi số lượng nơ-ron tại lớp ẩn: cấu hình **4 nơ-ron** so với **512 nơ-ron** (so với mốc gốc 64 nơ-ron).
* **Phương pháp thực hiện:**
  * Sử dụng kiến trúc `build_mlp(input_dim=len(vocab), hidden=h)` với $h \in \{4, 512\}$.
  * Đếm tổng số lượng tham số học được của toàn mạng: $(V \times H + H) + (H \times 1 + 1)$.
  * Huấn luyện mô hình $15$ epoch thông qua hàm `train_mlp` với hàm mất mát `BCEWithLogitsLoss` và thuật toán tối ưu Adam ($lr=0.001$).
  * Đo đạc thời gian huấn luyện thực tế và tính điểm F1 trên tập kiểm thử test set.
* **Kết quả thực nghiệm:**
  * **Lớp ẩn 4 nơ-ron:** $8.009$ tham số | Thời gian huấn luyện: $4.95\text{s}$ | F1 test: $\mathbf{0.9133}$
  * **Lớp ẩn 512 nơ-ron:** $1.025.025$ tham số | Thời gian huấn luyện: $33.98\text{s}$ | F1 test: $\mathbf{0.9070}$
* **Giải thích hiện tượng & Ứng dụng thực tế:**
  * *Hiện tượng Overfitting:* Khi tăng số nơ-ron lên 512, số tham số tăng gấp 128 lần ($> 1\text{ triệu tham số}$) và thời gian huấn luyện kéo dài gấp gần 7 lần, nhưng F1 trên tập kiểm thử lại giảm nhẹ ($0.9070 < 0.9133$). Mô hình quá lớn đã rơi vào hiện tượng học vẹt (overfitting), ghi nhớ cả các nhiễu thống kê ngẫu nhiên trong tập train thay vì học các đặc trưng tổng quát.
  * *Ứng dụng mạng nhỏ (4 nơ-ron):* Kiến trúc 4 nơ-ron cực kỳ phù hợp khi triển khai tại **thiết bị biên (Edge/IoT/Gateway viễn thông)** nơi tài nguyên bộ nhớ RAM chỉ có vài MB và không có GPU, hoặc các hệ thống yêu cầu độ trễ suy luận tức thì ($< 0.5\text{ ms}$) trên tập dữ liệu dạng bảng/từ vựng thưa.

---

## 6. Phương án cho sản phẩm thật

* **Yêu cầu bài toán:** Lọc $10\text{ triệu SMS/ngày}$ (thông lượng trung bình $\approx 116\text{ tin/giây}$), cam kết độ trễ nghiêm ngặt $p95 < 50\text{ ms/tin}$.
* **Phương án lựa chọn:** Lựa chọn **`BoW + MLP (deep learning)`** làm mô hình cốt lõi phục vụ luồng sản xuất.
* **Căn cứ số liệu thực nghiệm:**
  * **Độ trễ:** $p95 = \mathbf{0.593\text{ ms/tin}}$, bỏ xa yêu cầu trần $50\text{ ms}$ (nhanh gấp hơn 80 lần chuẩn yêu cầu).
  * **Thông lượng:** Đạt **$23.437\text{ tin/giây}$** khi chạy trên CPU thông thường, đáp ứng dư tải gấp hơn 200 lần so với lưu lượng trung bình $116\text{ tin/giây}$ (kể cả trong các đợt cao điểm nghẽn mạng).
  * **Hiệu năng & Chi phí:** Đạt $\text{F1} = 0.920$ trên tập test và $\text{F1} = 0.947$ trên tập SUB với chi phí vận hành mô hình là **$0\text{ USD}$** (tận dụng trực tiếp hạ tầng máy chủ gateway sẵn có).
* **Phân tích chi phí nếu dùng thuần LLM API:**
  * $10\text{ triệu tin/ngày} \times 30\text{ ngày} = 300\text{ triệu SMS/tháng}$.
  * Token input: $300\text{M} \times 100\text{ token} = 30.000\text{M token} \rightarrow 30.000 \times 0.10\text{ USD} = 3.000\text{ USD}$.
  * Token output: $300\text{M} \times 1\text{ token} = 300\text{M token} \rightarrow 300 \times 0.50\text{ USD} = 150\text{ USD}$.
  * $\rightarrow$ **Tổng chi phí: $3.150\text{ USD/tháng}$** ($\approx 80\text{ triệu VNĐ/tháng}$), quá đắt đỏ và vi phạm hoàn toàn SLA độ trễ do mạng ($p95 > 1.000\text{ ms}$).
* **Đề xuất thiết kế lai (Router Architecture):**
  * **Tầng 1 (Fast Gatekeeper):** Sử dụng `BoW + MLP` phân loại $100\%$ lưu lượng đầu vào.
    * Nếu $P(\text{spam}) < 0.3$: Xác định an toàn là `ham` $\rightarrow$ Chuyển tiếp ngay lập tức.
    * Nếu $P(\text{spam}) > 0.7$: Xác định rõ ràng là `spam` $\rightarrow$ Chặn ngay lập tức.
  * **Tầng 2 (Deep Inspector):** Với khoảng $3\% - 5\%$ tin nhắn rơi vào vùng phân vân $P(\text{spam}) \in [0.3, 0.7]$, hệ thống chuyển tiếp sang LLM (có prompt phòng thủ bọc thẻ `<sms>`) hoặc mô hình Transformer nhỏ để phân tích sâu ngữ nghĩa.
  * *Lợi ích:* Hệ thống giải quyết được hơn $95\%$ lưu lượng tin nhắn trong thời gian $< 1\text{ ms}$ với chi phí gần như bằng 0, đồng thời cắt giảm $> 95\%$ chi phí gọi API LLM (chỉ còn $\approx 100 - 150\text{ USD/tháng}$) mà vẫn xử lý triệt để các ca lừa đảo tinh vi.

---

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 0: Khảo sát & Baseline** | Cách tính baseline accuracy khi mô hình luôn đoán nhãn đa số (ham) | Công thức $N_{\text{ham}} / N_{\text{total}}$ và giải thích vì sao accuracy cao vẫn vô dụng với dữ liệu mất cân bằng | Tự kiểm tra lại số lượng mẫu sau khi drop duplicates trên notebook ($4.516 / 5.169 \approx 0.87$) và ghi chú vào báo cáo. |
| **Phần 1: Machine Learning (BoW)** | Tác động của kích thước từ vựng `max_size` và thay đổi ngưỡng phân loại | Phân tích Underfitting/Overfitting khi đổi vocab và sự đánh đổi giữa Precision và Recall khi đổi threshold (0.2 vs 0.8) | Chạy thử nghiệm thực tế với ngưỡng 0.2 và 0.8, ghi nhận chỉ số Precision/Recall tương ứng để trả lời câu hỏi lý thuyết về mã OTP. |
| **Phần 2: Deep Learning (MLP)** | Cách dựng kiến trúc `nn.Sequential` (Linear $\to$ ReLU $\to$ Linear) và giải thích logit | Đoạn code mẫu TODO 2.1, giải thích việc không dùng Sigmoid ở tầng cuối do kết hợp với `BCEWithLogitsLoss` | Tự gõ lại mã nguồn, kiểm tra chiều tensor đầu vào (2000) và đầu ra (1), quan sát đồ thị hàm loss hội tụ qua 15 epoch. |
| **Phần 3: Embedding & kNN** | So sánh cơ chế tìm láng giềng kNN trên vector dense với bộ phân loại tuyến tính | Giải thích cách MiniLM gom cụm ngữ nghĩa, tác động của $k$ (quá nhỏ gây nhạy cảm nhiễu, quá lớn bị lấn át bởi nhãn đa số) | Thử nghiệm với các giá trị $k = 1, 5, 25$, đối chiếu thời gian tra cứu và điểm F1 trên tập validation. |
| **Phần 4: LLM API & Prompt Defense** | Phân tích cơ chế tấn công Prompt Injection và cách dùng XML tagging để phòng vệ | Giải thích việc LLM nhầm dữ liệu với chỉ thị, gợi ý prompt 4d bọc thẻ `<sms>...</sms>` để cách ly ngữ cảnh | Chạy so sánh trực tiếp hàng 4a và 4d, kiểm tra bảng lỗi `INSPECT` để xác nhận 4 ca injection đã được chặn đứng. |
| **Bonus: 5 Hard cases tự tạo** | Gợi ý ý tưởng tạo 5 test case đối kháng bẫy được từng loại mô hình | 5 kịch bản SMS: Leet-speak né từ khóa, tin bạn bè chứa từ trúng thưởng, tin cảnh báo scam, rủ rê tự nhiên và injection đóng thẻ | Chỉnh sửa lại câu từ cho tự nhiên hơn, chạy vào mảng `MY_CASES`, chạy lại stress test và kiểm tra xem mô hình nào bị sập bẫy. |
| **Tính tay (Hand calc)** | Công thức và cách tính cho 5 bài toán (baseline, sigmoid, F1, cosine, LLM cost) | Các bước rút gọn, công thức đại số và kết quả làm tròn 2 chữ số thập phân | Bấm lại máy tính độc lập từng câu, chạy ô kiểm tra mã hash SHA-256 (đạt 5/5 tích xanh) và tự trình bày sạch sẽ ra file `hand_calc.jpg`. |
| **Câu 5: Biểu đồ Pareto & SLA** | Khái niệm đường Pareto, điểm bị lấn át và tác động của kích thước mẫu $N=40$ | Định nghĩa frontier tối ưu (1-F1 vs độ trễ) và phân tích phương sai lớn khi mẫu kiểm thử quá nhỏ | Đối chiếu vị trí thực tế của `BoW+MLP` và `MiniLM+kNN` trên biểu đồ matplotlib, xác nhận LLM nằm ngoài phạm vi SLA 50ms. |
| **Challenge: Đề 3 (4 vs 512 nơ-ron)** | Cách đo số tham số và cấu hình vòng lặp train thử nghiệm 2 kích thước lớp ẩn | Code mẫu chạy `build_mlp` với `hidden=4` và `hidden=512`, công thức tính $(V+2)H + 1$ | Thực thi trên GPU Colab, ghi nhận thông số thực tế (8.009 params / 4.95s vs 1.025.025 params / 33.98s) và phân tích hiện tượng overfit của mạng 512. |
| **Phản ánh cuối bài: Router & Chi phí** | Đề xuất kiến trúc Router phân luồng và ước tính ngân sách cho 10M tin/ngày | Thiết kế hệ thống lọc 2 tầng (BoW gatekeeper + LLM deep inspect) và phép tính chi phí token | Tính toán lại chi phí API hàng tháng ($3.150/tháng), viết lại đề xuất phân tầng lưu lượng theo ngưỡng xác suất $[0.3, 0.7]$. |
