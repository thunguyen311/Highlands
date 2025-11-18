# Highlands Interactive Dashboard 🏔️

Một bảng điều khiển tương tác động (Interactive Dashboard) được xây dựng bằng Streamlit để mô phỏng và phân tích dữ liệu theo thời gian thực.

## ✨ Tính năng

- **Dashboard tương tác động**: Giao diện web hiện đại với các biểu đồ tương tác
- **Nhiều tab phân tích**: 
  - 📅 Phân tích theo thời gian
  - 🗺️ Phân tích theo khu vực
  - 🛍️ Phân tích sản phẩm
  - 📊 Xem dữ liệu chi tiết
- **Bộ lọc động**: Lọc dữ liệu theo thời gian, danh mục, và số lượng hiển thị
- **Biểu đồ đa dạng**: Line charts, Bar charts, Pie charts, Scatter plots
- **Chỉ số thời gian thực**: Các metric cards với delta changes
- **Xuất dữ liệu**: Tải xuống dữ liệu dạng CSV

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8 trở lên
- pip (Python package manager)

### Các bước cài đặt

1. Clone repository:
```bash
git clone https://github.com/thunguyen311/Highlands.git
cd Highlands
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Chạy ứng dụng

Để chạy dashboard, sử dụng lệnh:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`

## 📊 Cấu trúc Dashboard

### 1. Sidebar (Thanh bên)
- **Bộ lọc thời gian**: Chọn khoảng thời gian để phân tích
- **Nút làm mới**: Cập nhật dữ liệu mới
- **Chọn danh mục**: Lọc theo doanh thu, khách hàng, sản phẩm, khu vực
- **Slider**: Điều chỉnh số lượng mục hiển thị

### 2. Chỉ số chính
- Tổng doanh thu với % thay đổi
- Tổng khách hàng với % thay đổi
- Tổng đơn hàng với % thay đổi
- Tỷ lệ chuyển đổi trung bình

### 3. Các Tab phân tích

#### Tab 1: Theo thời gian
- Biểu đồ đường: Doanh thu hàng ngày
- Biểu đồ cột: Số khách hàng hàng ngày
- Biểu đồ diện tích: Tỷ lệ chuyển đổi theo thời gian

#### Tab 2: Theo khu vực
- Biểu đồ cột: Doanh thu theo khu vực
- Biểu đồ tròn: Phân bố khách hàng
- Bảng dữ liệu khu vực chi tiết

#### Tab 3: Sản phẩm
- Biểu đồ ngang: Top sản phẩm bán chạy
- Scatter plot: Hiệu suất sản phẩm
- Bảng dữ liệu sản phẩm với sắp xếp động

#### Tab 4: Chi tiết
- Xem dữ liệu thô
- Thống kê tóm tắt
- Nút tải xuống CSV

## 🛠️ Công nghệ sử dụng

- **Streamlit**: Framework cho web app
- **Pandas**: Xử lý và phân tích dữ liệu
- **NumPy**: Tính toán số học
- **Plotly**: Tạo biểu đồ tương tác
- **Altair**: Visualization library

## 📝 Lưu ý

- Dữ liệu hiện tại được sinh ngẫu nhiên để mô phỏng
- Dashboard tự động cập nhật khi làm mới
- Có thể tùy chỉnh thêm các metric và biểu đồ theo nhu cầu

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License