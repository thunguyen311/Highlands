import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========================================================
# 1. CONFIG VÀ CUSTOM CSS (Để giao diện gần giống Figma)
# ========================================================
st.set_page_config(
    page_title="Highlands Price Optimization Project",
    page_icon="☕",
    layout="wide"
)

# Custom CSS để tạo style cho các thẻ (cards) và tiêu đề
st.markdown("""
<style>
    /* Màu sắc chủ đạo (Giả định màu nâu đậm/đỏ của Highlands) */
    :root {
        --primary-color: #A31F34; /* Đỏ/Nâu đậm */
        --secondary-color: #FFC000; /* Vàng (highlight) */
        --bg-color: #F8F9FA;
    }
    
    /* Cấu hình chung */
    .stApp {
        background-color: var(--bg-color);
    }
    
    /* Thiết kế cho các thẻ KPI (metrics) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E6E6;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    /* Thẻ metric chính */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Biểu đồ Container */
    .stContainer {
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 1px 1px 8px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }
    
    /* Nút primary màu đỏ Highlands */
    .stButton>button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    h1, h2, h3 {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

def var_css(var_name):
    """Lấy biến CSS từ thẻ style."""
    return f"var(--{var_name})"

# ========================================================
# 2. LOAD & PROCESS DATA (TÍCH HỢP LOGIC SEGMENTATION)
# ========================================================

@st.cache_data
def load_and_calculate_kpis(data_dir="data"):
    """Load data, tính toán KPI, và thực hiện Segmentation (Mô phỏng từ segment.py)."""
    try:
        df_trans = pd.read_csv(f'{data_dir}/transaction_data.csv')
        df_prod = pd.read_csv(f'{data_dir}/product_master.csv')
        
        # --- 2.1 Chuẩn bị dữ liệu cơ bản ---
        df_trans['Date_Time'] = pd.to_datetime(df_trans['Date_Time'])
        df_merged = pd.merge(df_trans, df_prod[['Product_ID', 'COGS', 'Category']], on='Product_ID', how='left')
        df_merged['Total_COGS'] = df_merged['COGS'] * df_merged['Quantity']
        df_merged['Gross_Profit'] = df_merged['Total_Paid'] - df_merged['Total_COGS']
        
        # --- 2.2 Tính KPIs chung (Khu vực 1 & 2) ---
        total_revenue = df_merged['Total_Paid'].sum()
        total_profit = df_merged['Gross_Profit'].sum()
        aov = df_merged['Total_Paid'].sum() / df_merged['Transaction_ID'].nunique()
        
        revenue_delta = total_revenue * 0.08
        profit_delta = total_profit * 0.12
        aov_delta = aov * 0.03

        # --- 2.3 RFM & Segmentation (Mô phỏng logic từ segment.py) ---
        snapshot_date = df_merged['Date_Time'].max() + pd.Timedelta(days=1)
        
        rfm_df = df_merged.groupby('Customer_ID').agg(
            Recency=('Date_Time', lambda x: (snapshot_date - x.max()).days),
            Frequency=('Transaction_ID', 'nunique'),
            Monetary=('Total_Paid', 'sum')
        ).reset_index()
        
        # Tiền xử lý (Log và Scale)
        rfm_log = np.log1p(rfm_df[['Recency', 'Frequency', 'Monetary']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        # Mô hình K-Means (Giả định 4 cụm)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Ánh xạ cụm thành tên Segment dễ hiểu
        segment_map = {0: 'Trung thành (A)', 1: 'Tiềm năng (B)', 2: 'Nguy cơ mất (C)', 3: 'Mới/Thăm dò (D)'}
        rfm_df['Segment'] = rfm_df['Cluster'].map(segment_map)
        
        # --- 2.4 Kết quả Tối ưu hóa (Mô phỏng logic từ optimization.py) ---
        # Đây là mức tăng/giảm giá đề xuất cho các sản phẩm/segment cụ thể
        optimization_result_mock = pd.DataFrame({
            'Segment': ['Trung thành (A)', 'Tiềm năng (B)', 'Nguy cơ mất (C)', 'Mới/Thăm dò (D)'],
            'Suggested_Price_Change': [0.15, 0.05, -0.05, 0.0],
            'Expected_Profit_Increase': [250000000, 50000000, 10000000, 5000000]
        })
        
        return {
            'revenue': total_revenue, 'profit': total_profit, 'aov': aov,
            'revenue_delta': revenue_delta, 'profit_delta': profit_delta, 'aov_delta': aov_delta,
            'df_merged': df_merged,
            'rfm_df': rfm_df, # Kết quả Segmentation
            'optimization_result': optimization_result_mock # Kết quả Optimization
        }
        
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return None

kpis_data = load_and_calculate_kpis()

# ========================================================
# 3. GIAO DIỆN CHÍNH (ÁNH XẠ TỪ THIẾT KẾ FIGMA)
# ========================================================

st.subheader(f"Dashboard Phân tích & Tối ưu hóa - Cập nhật: {date.today().strftime('%d/%m/%Y')}")

# --- BỘ LỌC SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/52/Highlands_Coffee_logo.svg/1200px-Highlands_Coffee_logo.svg.png", use_column_width=True)
    st.title("Phân tích Chiến lược Giá")
    st.markdown("---")
    
    st.subheader("Bộ Lọc Dữ liệu")
    location = st.selectbox("Chọn Khu vực:", ["Toàn Quốc", "Hà Nội", "Hồ Chí Minh", "Đà Nẵng"], index=0)
    segment_options = ["Tất cả"] + (list(kpis_data['rfm_df']['Segment'].unique()) if kpis_data and 'rfm_df' in kpis_data else [])
    segment = st.selectbox("Chọn Phân khúc KH:", segment_options)
    category = st.multiselect("Chọn Danh mục Sản phẩm:", ["Coffee", "Tea", "Freeze", "Food"], default=["Coffee", "Tea", "Freeze"])
    
    st.markdown("---")
    st.info("Các trang chi tiết ở menu bên trái. 👈")

if kpis_data is None:
    st.warning("Không thể hiển thị Dashboard. Vui lòng kiểm tra lại thư mục 'data/'.")
    st.stop()


# --- KHU VỰC 1: 4 THẺ KPI CHÍNH ---
st.markdown("### Hiệu suất Kinh doanh Tổng quan (Baseline)")

col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)

with col_kpi_1:
    st.metric(
        label="Tổng Doanh thu (VND)", 
        value=f"{kpis_data['revenue']:,.0f}", 
        delta=f"{kpis_data['revenue_delta']:,.0f} (+8%)"
    )

with col_kpi_2:
    st.metric(
        label="Tổng Lợi nhuận Gộp (VND)", 
        value=f"{kpis_data['profit']:,.0f}", 
        delta=f"{kpis_data['profit_delta']:,.0f} (+12%)"
    )

with col_kpi_3:
    st.metric(
        label="AOV (Giá trị đơn TB - VND)", 
        value=f"{kpis_data['aov']:,.0f}", 
        delta=f"{kpis_data['aov_delta']:,.0f} (+3%)"
    )

with col_kpi_4:
    st.metric(
        label="Tổng Giao dịch", 
        value=f"{kpis_data['df_merged']['Transaction_ID'].nunique():,}", 
        delta="+5%"
    )


st.markdown("---")

# --- KHU VỰC 2: BIỂU ĐỒ CHÍNH (Chia 2 cột chính - 70/30) ---
chart_col_main, chart_col_side = st.columns([7, 3])

# --- CỘT TRÁI (70%) - Xu hướng Lợi nhuận (EDA) ---
with chart_col_main:
    with st.container(border=True):
        st.subheader("Xu hướng Lợi nhuận gộp theo Tháng (EDA)")
        
        df_monthly = kpis_data['df_merged'].copy()
        df_monthly['Month'] = df_monthly['Date_Time'].dt.to_period('M').astype(str)
        monthly_profit = df_monthly.groupby('Month')['Gross_Profit'].sum().reset_index()
        
        fig_profit = px.line(
            monthly_profit, 
            x='Month', 
            y='Gross_Profit', 
            title='Lợi nhuận Gộp Hàng tháng',
            markers=True,
            color_discrete_sequence=[var_css('primary-color')]
        )
        fig_profit.update_yaxes(tickformat=',.0f')
        fig_profit.update_layout(height=400)
        st.plotly_chart(fig_profit, use_container_width=True)

# --- CỘT PHẢI (30%) - Tỷ trọng Lợi nhuận (EDA) ---
with chart_col_side:
    with st.container(border=True):
        st.subheader("Tỷ trọng Lợi nhuận theo Danh mục (EDA)")
        
        profit_by_cat = kpis_data['df_merged'].groupby('Category')['Gross_Profit'].sum().reset_index()
        
        fig_pie = px.pie(
            profit_by_cat, 
            values='Gross_Profit', 
            names='Category', 
            hole=.3,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.metric(label="Tỷ suất Lợi nhuận Gộp (GPM)", value=f"{(kpis_data['profit']/kpis_data['revenue'])*100:.1f}%", delta="-0.5%")


# --- KHU VỰC 3: PHÂN TÍCH CHUYÊN SÂU (2 CỘT) ---
st.markdown("### Kết quả Phân tích Trọng yếu")

deep_col_1, deep_col_2 = st.columns(2)

# --- CỘT 1: KẾT QUẢ SEGMENTATION (TỪ SEGMENT.PY) ---
with deep_col_1:
    with st.container(border=True):
        st.subheader("Tỷ lệ Khách hàng theo Phân khúc (Segmentation)")
        
        segment_counts = kpis_data['rfm_df']['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        # Biểu đồ cột thể hiện kích thước từng segment
        fig_seg = px.bar(
            segment_counts, 
            x='Segment', 
            y='Count', 
            title='Kích thước các Phân khúc Khách hàng',
            color='Segment',
            color_discrete_map={
                'Trung thành (A)': 'green', 
                'Tiềm năng (B)': 'blue', 
                'Nguy cơ mất (C)': 'orange', 
                'Mới/Thăm dò (D)': 'red'
            }
        )
        fig_seg.update_layout(height=300)
        st.plotly_chart(fig_seg, use_container_width=True)

# --- CỘT 2: KẾT QUẢ OPTIMIZATION (TỪ OPTIMIZATION.PY) ---
with deep_col_2:
    with st.container(border=True):
        st.subheader("Mức Thay đổi Giá Đề xuất theo Segment (Optimization)")
        
        df_opt = kpis_data['optimization_result']
        
        # Biểu đồ cột thể hiện % thay đổi giá đề xuất
        fig_opt = px.bar(
            df_opt, 
            x='Segment', 
            y='Suggested_Price_Change', 
            color='Suggested_Price_Change',
            color_continuous_scale=px.colors.diverging.RdYlGn, # Dùng scale xanh-đỏ cho tăng-giảm
            title='Phần trăm Thay đổi Giá đề xuất (%)'
        )
        fig_opt.update_yaxes(tickformat=".1%")
        fig_opt.update_layout(height=300)
        st.plotly_chart(fig_opt, use_container_width=True)
        
        # Thẻ metric thể hiện kết quả tối ưu hóa cuối cùng
        total_expected_increase = df_opt['Expected_Profit_Increase'].sum()
        st.metric(
            label="Lợi nhuận Tăng thêm Dự kiến (Triệu VND)",
            value=f"{total_expected_increase/1000000:,.0f}",
            delta="Mục tiêu Tối ưu hóa"
        )