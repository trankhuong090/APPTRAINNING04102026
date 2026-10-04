# 📈 Ứng Dụng Web Định Lượng: Kiểm Định Chiến Lược SMA + OBV & Phân Bổ Danh Mục MPT

Ứng dụng web được xây dựng trên nền tảng **Streamlit** nhằm kiểm định tính hiệu quả của chiến lược giao dịch kỹ thuật kết hợp giữa chỉ báo xu hướng **SMA (Simple Moving Average)** và chỉ báo dòng tiền **OBV (On-Balance Volume)**, đồng thời áp dụng **Lý thuyết Danh mục Hiện đại (Modern Portfolio Theory - MPT)** của Harry Markowitz để tối ưu hóa phân bổ tỷ trọng trên thị trường chứng khoán Việt Nam (dữ liệu HOSE).

---

## 🌟 Tính Năng Nổi Bật

1. **Khắc phục triệt để Look-Ahead Bias:**
   - Tín hiệu tạo ra tại cuối phiên $t$ chỉ được kích hoạt và thực thi tại phiên tiếp theo $t+1$ (`executed_holding = holding.shift(1)`).
2. **Nguyên tắc Kiểm định Ngoài mẫu (Out-of-sample):**
   - **Tập Train (2020-01-01 → 2021-12-31):** Dùng để tối ưu tham số SMA/OBV (qua Hyperopt) và ước lượng vector tỷ trọng MPT.
   - **Tập Test (2022-01-01 → 2022-12-31):** Giữ nguyên toàn bộ tham số và tỷ trọng đã ước lượng từ Train để kiểm định ngoài mẫu khách quan, tránh hiện tượng Overfitting (học vẹt dữ liệu).
3. **Kiểm định Đa Chiến Lược:**
   - So sánh trực tiếp: `Buy & Hold`, `SMA`, `OBV`, `SMA + OBV (AND)`, `SMA + OBV (OR)`.
4. **Tối Ưu & Phân Bổ Danh Mục:**
   - **Equal Weight ($1/N$):** Chia đều tỷ trọng giữa các tài sản.
   - **MPT (Markowitz):** Tối đa hóa Sharpe Ratio với ràng buộc Long-only ($0 \le w_i \le 1, \sum w_i = 1$).
5. **Giao Diện Trực Quan & Tương Tác:**
   - Biểu đồ nến tương tác (Plotly), đường SMA, OBV, điểm Mua/Bán (Buy/Sell markers).
   - Biểu đồ tăng trưởng vốn (Equity Curve) và biểu đồ sụt giảm vốn (Underwater Drawdown).
   - Tự động xuất báo cáo và hỗ trợ tải dữ liệu kết quả dưới dạng file CSV.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách các thư viện Python cần thiết
├── README.md                   # Tài liệu hướng dẫn sử dụng và triển khai
├── HOSE_2020_2023_in.csv       # Bộ dữ liệu giao dịch các cổ phiếu niêm yết trên HOSE
└── HOSE_SMA_OBV_EqualWeight_MPT (1).ipynb # Notebook nghiên cứu gốc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### 1. Yêu cầu môi trường
- Python phiên bản `3.9` đến `3.12`.

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal / command prompt tại thư mục dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng Streamlit
Chạy lệnh sau để khởi động web app:

```bash
streamlit run app.py
```

Sau khi chạy lệnh, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Deploy Lên Streamlit Cloud Miễn Phí

Để chia sẻ ứng dụng với bạn bè, giảng viên hoặc đưa vào báo cáo môn học, bạn có thể triển khai miễn phí lên **Streamlit Community Cloud** theo các bước sau:

### Bước 1: Đẩy mã nguồn lên GitHub
1. Khởi tạo kho lưu trữ (repository) mới trên GitHub (ví dụ đặt tên: `stock-strategy-sma-obv-mpt`).
2. Tải toàn bộ các file: `app.py`, `requirements.txt`, `README.md` và file dữ liệu `HOSE_2020_2023_in.csv` lên repository đó.
   *(Lưu ý: File `HOSE_2020_2023_in.csv` có dung lượng ~5.7MB, hoàn toàn nằm trong giới hạn cho phép 100MB của GitHub).*

### Bước 2: Đăng nhập Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.

### Bước 3: Tạo ứng dụng mới
1. Nhấn nút **"New app"** (hoặc **"Create app"**).
2. Chọn Repository bạn vừa tải mã nguồn lên.
3. Chọn Branch: `main` (hoặc `master`).
4. Điền Main file path: `app.py`.
5. Nhấn **"Deploy!"**.

Hệ thống Streamlit Cloud sẽ tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy web app trong khoảng 1-2 phút. Bạn sẽ nhận được đường link web trực tiếp (URL) để chia sẻ.

---

## 📊 Phương Pháp Luận & Quy Tắc Giao Dịch

### 1. Chỉ báo Kỹ thuật & Tín hiệu
- **SMA Crossover:**
  - $BUY (+1)$ khi $SMA_{ngắn}$ cắt lên trên $SMA_{dài}$ (Golden Cross).
  - $SELL (-1)$ khi $SMA_{ngắn}$ cắt xuống dưới $SMA_{dài}$ (Death Cross).
- **OBV Crossover:**
  - $BUY (+1)$ khi $OBV$ cắt lên trên đường trung bình động $OBV\_MA$.
  - $SELL (-1)$ khi $OBV$ cắt xuống dưới đường trung bình động $OBV\_MA$.
- **Kết hợp AND:** Chỉ vào lệnh khi cả SMA và OBV cùng đồng thuận phát tín hiệu.
- **Kết hợp OR:** Vào lệnh khi có ít nhất một chỉ báo báo hiệu; nếu có tín hiệu mâu thuẫn (1 Mua, 1 Bán) thì giữ vị thế trung lập (0).

### 2. Quản lý Vị thế & Lợi nhuận
- **Trạng thái nắm giữ:** Duy trì vị thế $1.0$ (nắm giữ cổ phiếu) khi có tín hiệu BUY cho đến khi xuất hiện tín hiệu SELL để đóng vị thế về $0.0$ (tiền mặt).
- **Độ trễ 1 phiên (Lag 1):** Vị thế thực tế hôm nay được áp dụng từ tín hiệu phiên trước để phản ánh tính khả thi trong thực tế đặt lệnh.
- **Khấu trừ chi phí:** Tự động trừ phí giao dịch dựa trên tỷ lệ luân chuyển vị thế (turnover).

### 3. Phân bổ Danh mục MPT (Markowitz)
- Ước lượng vector lợi nhuận kỳ vọng $\mu$ và ma trận hiệp phương sai $\Sigma$ chỉ trên tập **Train**.
- Bài toán tối ưu hóa:
  $$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
  Ràng buộc: $\sum w_i = 1$ và $0 \le w_i \le 1$ (Long-only).
- Trọng số tối ưu sau đó được **giữ nguyên cố định** khi chạy kiểm định trên tập **Test** năm 2022.

---

## 📌 Ý Nghĩa Kết Quả Thực Nghiệm

- **Năm 2020-2021 (Train - Thị trường Tăng trưởng / Bull Market):** Danh mục MPT thường đạt hiệu suất vượt trội do mô hình tối ưu hóa bám sát dữ liệu lịch sử tăng giá mạnh.
- **Năm 2022 (Test - Thị trường Suy thoái / Bear Market):** VN-Index và đa số cổ phiếu sụt giảm từ 30% đến 60%. Chiến lược kết hợp SMA + OBV phát tín hiệu bán và chuyển về tiền mặt sớm, giúp giảm thiểu đáng kể mức sụt giảm tối đa (Max Drawdown) so với Buy & Hold.
- Hiện tượng phân bổ ngoài mẫu minh chứng cho bài học kinh nghiệm quan trọng: *Một danh mục tối ưu trên quá khứ không nhất thiết chiến thắng danh mục phân bổ đều (Equal Weight) trong tương lai khi chế độ thị trường (market regime) thay đổi.*
