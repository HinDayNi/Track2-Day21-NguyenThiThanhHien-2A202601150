# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | Nguyễn Thị Thanh Hiền|
| MSSV | 2A202601150 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/HinDayNi/Track2-Day21-NguyenThiThanhHien-2A202601150.git |
| Ngày nộp | 21/8/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.2 | 5 | 0.7032 | 0.870 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

**Lý do:** Bộ tham số này đạt `f1_score=0.7109`, cao nhất trong ba lần thử và vượt ngưỡng chất lượng 0.65. Đồng thời accuracy 0.878 cũng là giá trị cao nhất, nên lần chạy có F1 cao nhất và accuracy cao nhất trùng nhau. Bộ `50/0.05/2` cho F1 chỉ 0.6051, cho thấy cấu hình quá yếu đối với bài toán. Khi tăng lên 200 cây, learning rate 0.2 và max depth 5, F1 giảm nhẹ còn 0.7032. Kết quả cho thấy tăng số cây và learning rate không nhất thiết làm mô hình tốt hơn; cấu hình 100 cây với learning rate 0.1 tạo cân bằng tốt hơn giữa khả năng học và độ phức tạp.


<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Dữ liệu Adult có sự mất cân bằng lớp, trong đó nhóm thu nhập trên 50K chỉ chiếm khoảng một phần tư tổng số mẫu. Vì vậy, một mô hình luôn dự đoán “thu nhập thấp” vẫn có thể đạt accuracy khoảng 75% dù hoàn toàn không nhận diện được lớp thu nhập cao. Accuracy vì thế có thể tạo cảm giác mô hình hoạt động tốt trong khi bỏ sót lớp quan trọng. F1 của lớp dương kết hợp precision và recall, phản ánh đồng thời khả năng dự đoán đúng người có thu nhập cao và hạn chế bỏ sót các trường hợp này. Trong bài lab, `f1_score` mặc định được tính trực tiếp cho lớp dương `target=1`, thay vì dùng `average="weighted"` hay `average="macro"`, vì mục tiêu quality gate là đánh giá cụ thể năng lực nhận diện lớp thu nhập trên 50K.

___

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| GitHub Actions không SSH được vào EC2 | `SERVER_SSH_KEY` lưu sai định dạng và ban đầu public key chưa đúng | Tạo deploy key mới, thêm public key vào `authorized_keys` và cập nhật private key trong GitHub Secrets |
| Service `income-api` liên tục restart | Một tiến trình FastAPI chạy thủ công vẫn chiếm port 8080 | Xác định PID bằng `ss` và `ps`, dừng process cũ rồi quản lý API hoàn toàn bằng systemd |
| Pipeline ban đầu dùng cấu hình GCP | Lab scaffold dùng Cloud Storage của GCP trong khi môi trường triển khai là AWS | Chuyển DVC remote sang S3, dùng `boto3`, IAM Role và AWS Secrets trong GitHub Actions |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7109 | 0.8780 |
| Bước 3 (thêm `train_batch2`) | 0.7014 | 0.8740 |

**Nhận xét:** Sau khi bổ sung `train_batch2`, F1 giảm từ 0.7109 xuống 0.7014 và accuracy giảm từ 0.8780 xuống 0.8740. Điều này cho thấy việc tăng lượng dữ liệu không đảm bảo mô hình luôn cải thiện metric; dữ liệu mới có thể có phân phối tương tự hoặc bổ sung thêm các mẫu khó, làm kết quả trên tập holdout giảm nhẹ. Tuy vậy, F1 vẫn cao hơn ngưỡng chất lượng 0.65 nên pipeline tiếp tục cho phép triển khai.

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
