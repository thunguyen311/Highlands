import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Highlands Interactive Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏔️ Highlands Interactive Dashboard")
st.markdown("**Bảng điều khiển tương tác động với dữ liệu thời gian thực**")

# Sidebar for filters and controls
st.sidebar.header("⚙️ Bộ lọc và điều khiển")

# Date range selector
date_range = st.sidebar.date_input(
    "Chọn khoảng thời gian",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# Data refresh button
if st.sidebar.button("🔄 Làm mới dữ liệu"):
    st.rerun()

# Category selector
categories = ["Doanh thu", "Khách hàng", "Sản phẩm", "Khu vực"]
selected_category = st.sidebar.selectbox("Chọn danh mục", categories)

# Metric slider
metric_count = st.sidebar.slider("Số lượng mục hiển thị", 5, 50, 20)

# Generate sample data
@st.cache_data
def generate_data(days=30):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = {
        'Ngày': dates,
        'Doanh thu': np.random.randint(1000000, 5000000, days),
        'Khách hàng': np.random.randint(50, 300, days),
        'Đơn hàng': np.random.randint(30, 200, days),
        'Tỷ lệ chuyển đổi': np.random.uniform(0.1, 0.5, days)
    }
    return pd.DataFrame(data)

# Generate regional data
@st.cache_data
def generate_regional_data():
    regions = ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng']
    data = {
        'Khu vực': regions,
        'Doanh thu': np.random.randint(5000000, 20000000, len(regions)),
        'Khách hàng': np.random.randint(100, 1000, len(regions)),
        'Tăng trưởng (%)': np.random.uniform(-10, 50, len(regions))
    }
    return pd.DataFrame(data)

# Generate product data
@st.cache_data
def generate_product_data():
    products = ['Cà phê Phin', 'Trà sữa', 'Bánh ngọt', 'Sinh tố', 'Nước ép', 
                'Soda', 'Freeze', 'Cappuccino', 'Latte', 'Espresso']
    data = {
        'Sản phẩm': products,
        'Số lượng bán': np.random.randint(50, 500, len(products)),
        'Doanh thu': np.random.randint(500000, 5000000, len(products)),
        'Đánh giá': np.random.uniform(3.5, 5.0, len(products))
    }
    return pd.DataFrame(data)

# Load data
df = generate_data(30)
regional_df = generate_regional_data()
product_df = generate_product_data()

# Key metrics row
st.header("📊 Các chỉ số chính")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = df['Doanh thu'].sum()
    revenue_change = ((df['Doanh thu'].iloc[-1] - df['Doanh thu'].iloc[0]) / df['Doanh thu'].iloc[0]) * 100
    st.metric(
        label="Tổng doanh thu",
        value=f"{total_revenue:,.0f} ₫",
        delta=f"{revenue_change:.1f}%"
    )

with col2:
    total_customers = df['Khách hàng'].sum()
    customer_change = ((df['Khách hàng'].iloc[-1] - df['Khách hàng'].iloc[0]) / df['Khách hàng'].iloc[0]) * 100
    st.metric(
        label="Tổng khách hàng",
        value=f"{total_customers:,}",
        delta=f"{customer_change:.1f}%"
    )

with col3:
    total_orders = df['Đơn hàng'].sum()
    order_change = ((df['Đơn hàng'].iloc[-1] - df['Đơn hàng'].iloc[0]) / df['Đơn hàng'].iloc[0]) * 100
    st.metric(
        label="Tổng đơn hàng",
        value=f"{total_orders:,}",
        delta=f"{order_change:.1f}%"
    )

with col4:
    avg_conversion = df['Tỷ lệ chuyển đổi'].mean()
    st.metric(
        label="Tỷ lệ chuyển đổi TB",
        value=f"{avg_conversion:.1%}",
        delta="Tốt" if avg_conversion > 0.3 else "Cần cải thiện"
    )

st.divider()

# Main charts section
st.header("📈 Phân tích dữ liệu")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📅 Theo thời gian", "🗺️ Theo khu vực", "🛍️ Sản phẩm", "📊 Chi tiết"])

with tab1:
    # Time series chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biểu đồ doanh thu theo thời gian")
        fig_revenue = px.line(
            df, 
            x='Ngày', 
            y='Doanh thu',
            title='Doanh thu hàng ngày',
            markers=True
        )
        fig_revenue.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Doanh thu (₫)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        st.subheader("Biểu đồ khách hàng theo thời gian")
        fig_customers = px.bar(
            df, 
            x='Ngày', 
            y='Khách hàng',
            title='Số khách hàng hàng ngày',
            color='Khách hàng',
            color_continuous_scale='Blues'
        )
        fig_customers.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Số khách hàng",
            showlegend=False
        )
        st.plotly_chart(fig_customers, use_container_width=True)
    
    # Additional metrics chart
    st.subheader("Tỷ lệ chuyển đổi theo thời gian")
    fig_conversion = go.Figure()
    fig_conversion.add_trace(go.Scatter(
        x=df['Ngày'],
        y=df['Tỷ lệ chuyển đổi'],
        mode='lines+markers',
        name='Tỷ lệ chuyển đổi',
        fill='tozeroy',
        line=dict(color='green', width=2)
    ))
    fig_conversion.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Tỷ lệ chuyển đổi",
        hovermode='x unified'
    )
    st.plotly_chart(fig_conversion, use_container_width=True)

with tab2:
    # Regional analysis
    st.subheader("Phân tích theo khu vực")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for regional revenue
        fig_regional = px.bar(
            regional_df,
            x='Khu vực',
            y='Doanh thu',
            title='Doanh thu theo khu vực',
            color='Doanh thu',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with col2:
        # Pie chart for customer distribution
        fig_pie = px.pie(
            regional_df,
            values='Khách hàng',
            names='Khu vực',
            title='Phân bố khách hàng theo khu vực'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Regional data table
    st.subheader("Bảng dữ liệu khu vực")
    st.dataframe(
        regional_df.style.format({
            'Doanh thu': '{:,.0f} ₫',
            'Khách hàng': '{:,}',
            'Tăng trưởng (%)': '{:.1f}%'
        }),
        use_container_width=True
    )

with tab3:
    # Product analysis
    st.subheader("Phân tích sản phẩm")
    
    # Sort products by selected metric
    sort_by = st.selectbox(
        "Sắp xếp theo",
        ['Số lượng bán', 'Doanh thu', 'Đánh giá']
    )
    
    product_df_sorted = product_df.sort_values(by=sort_by, ascending=False).head(metric_count)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Horizontal bar chart for products
        fig_products = px.bar(
            product_df_sorted,
            y='Sản phẩm',
            x='Số lượng bán',
            title=f'Top {metric_count} sản phẩm bán chạy',
            orientation='h',
            color='Số lượng bán',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    with col2:
        # Scatter plot for product performance
        fig_scatter = px.scatter(
            product_df,
            x='Số lượng bán',
            y='Doanh thu',
            size='Đánh giá',
            color='Đánh giá',
            hover_name='Sản phẩm',
            title='Hiệu suất sản phẩm',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Product data table
    st.subheader("Bảng dữ liệu sản phẩm")
    st.dataframe(
        product_df_sorted.style.format({
            'Số lượng bán': '{:,}',
            'Doanh thu': '{:,.0f} ₫',
            'Đánh giá': '{:.1f}⭐'
        }),
        use_container_width=True
    )

with tab4:
    # Detailed data view
    st.subheader("Dữ liệu chi tiết")
    
    # Show raw data with filters
    show_data = st.checkbox("Hiển thị dữ liệu thô", value=True)
    
    if show_data:
        st.subheader("Dữ liệu theo thời gian")
        st.dataframe(
            df.style.format({
                'Doanh thu': '{:,.0f} ₫',
                'Khách hàng': '{:,}',
                'Đơn hàng': '{:,}',
                'Tỷ lệ chuyển đổi': '{:.1%}'
            }),
            use_container_width=True
        )
        
        # Download button for data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải xuống dữ liệu CSV",
            data=csv,
            file_name=f'highlands_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
    
    # Statistical summary
    st.subheader("Thống kê tóm tắt")
    st.dataframe(df.describe(), use_container_width=True)

# Footer with real-time update
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("🏔️ Highlands Interactive Dashboard - Bảng điều khiển tương tác động")
with col2:
    st.caption(f"📅 Cập nhật: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    if st.button("ℹ️ Thông tin"):
        st.info("Dashboard được xây dựng bằng Streamlit với dữ liệu mô phỏng.")
