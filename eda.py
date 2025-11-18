import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ========================================================
# 1. CẤU HÌNH TRANG & STYLE
# ========================================================
st.set_page_config(page_title="Full EDA Analysis", page_icon="📊", layout="wide")

# Giữ nguyên style của bạn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Tăng kích thước mặc định lên một chút cho web
CURRENT_YEAR = 2024

st.title("📊 Phân tích Dữ liệu Chuyên sâu (Full EDA)")
st.markdown("""
Phiên bản chi tiết bao gồm:
- Xử lý dữ liệu thô & Feature Engineering.
- Phân tích đơn biến (Univariate): Khách hàng, Giao dịch.
- Phân tích đa biến (Bivariate): Quan hệ giữa Nghề nghiệp - Lợi nhuận - Sản phẩm.
""")

# ========================================================
# 2. LOAD DATA (Dùng Cache để không phải load lại 1.2tr dòng)
# ========================================================
@st.cache_data
def load_raw_data():
    """
    Load dữ liệu gốc và chuyển đổi kiểu dữ liệu cơ bản.
    """
    try:
        # Đường dẫn tương đối chuẩn cho Streamlit Cloud
        df_customer = pd.read_csv("data/customer_profile.csv")
        df_trans    = pd.read_csv("data/transaction_data.csv")
        df_product  = pd.read_csv("data/product_master.csv")
        df_macro    = pd.read_csv("data/macro_context.csv")
        return df_customer, df_trans, df_product, df_macro
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file dữ liệu. Hãy đảm bảo bạn đã upload folder 'data' lên GitHub.")
        return None, None, None, None

df_customer, df_trans, df_product, df_macro = load_raw_data()

if df_customer is None:
    st.stop()

# ========================================================
# 3. DATA PREPROCESSING (Giữ nguyên logic của bạn)
# ========================================================

# --- 3.1 Xử lý Customer ---
def preprocess_customer(df_customer):
    df = df_customer.copy()
    # Chuẩn hóa tên cột
    df = df.rename(columns={"Income level": "Income_Level"})
    
    # Tính tuổi
    df['Age'] = CURRENT_YEAR - df['YoB']
    
    # Phân nhóm tuổi (Binning)
    bins = [0, 18, 24, 34, 44, 54, 100]
    labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    return df

# --- 3.2 Xử lý Transaction ---
def preprocess_transaction(df_trans):
    df = df_trans.copy()
    # Convert datetime
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    
    # Extract features
    df['Date'] = df['Date_Time'].dt.date
    df['Hour'] = df['Date_Time'].dt.hour
    df['Month'] = df['Date_Time'].dt.month
    df['Year'] = df['Date_Time'].dt.year
    df['DayOfWeek'] = df['Date_Time'].dt.day_name()
    
    return df

with st.spinner('Đang xử lý dữ liệu...'):
    df_cust_clean = preprocess_customer(df_customer)
    df_trans_clean = preprocess_transaction(df_trans)

# ========================================================
# 4. HIỂN THỊ DASHBOARD (CHIA THEO TABS)
# ========================================================

# Tôi chia logic dài của bạn thành các Tabs để dễ xem hơn
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Hồ sơ Khách hàng", 
    "2. Phân tích Giao dịch", 
    "3. Lợi nhuận & Sản phẩm (Deep Dive)", 
    "4. Dữ liệu"
])

# --------------------------------------------------------
# TAB 1: CUSTOMER PROFILE
# --------------------------------------------------------
with tab1:
    st.header("1. Phân tích Hồ sơ Khách hàng")
    
    col1, col2 = st.columns(2)
    
    # Biểu đồ 1: Tuổi
    with col1:
        st.subheader("Phân bố Độ tuổi")
        fig, ax = plt.subplots()
        sns.histplot(data=df_cust_clean, x='Age', bins=30, kde=True, color='skyblue', ax=ax)
        ax.set_title("Histogram độ tuổi khách hàng")
        st.pyplot(fig)
        
        st.caption(f"Tuổi trung bình: {df_cust_clean['Age'].mean():.1f}")

    # Biểu đồ 2: Nhóm tuổi
    with col2:
        st.subheader("Nhóm tuổi (Age Groups)")
        fig, ax = plt.subplots()
        sns.countplot(data=df_cust_clean, x='Age_Group', palette='viridis', ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # Biểu đồ 3: Thu nhập
    st.subheader("Phân bố Mức thu nhập")
    income_order = ['< 2M', '2-5M', '5-10M', '10-20M', '20-50M', '> 50M'] # Sắp xếp cho chuẩn
    # Lọc chỉ những giá trị có trong data để tránh lỗi sort
    existing_order = [x for x in income_order if x in df_cust_clean['Income_Level'].unique()]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_cust_clean, y='Income_Level', order=existing_order, palette='magma', ax=ax)
    st.pyplot(fig)

    # Biểu đồ 4: Nghề nghiệp vs Membership (Cái này quan trọng)
    st.subheader("Quan hệ Nghề nghiệp & Hạng thành viên")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df_cust_clean, x='Occupation', hue='Membership_Tier', palette='Set2', ax=ax)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    st.pyplot(fig)

# --------------------------------------------------------
# TAB 2: TRANSACTION ANALYSIS
# --------------------------------------------------------
with tab2:
    st.header("2. Phân tích Giao dịch")

    # 2.1 Tổng quan
    total_rev = df_trans_clean['Total_Paid'].sum()
    total_txn = len(df_trans_clean)
    st.info(f"Tổng quan dữ liệu: {total_txn:,} giao dịch - Tổng doanh thu: {total_rev:,.0f} VND")

    # 2.2 Doanh thu theo tháng
    st.subheader("Xu hướng Doanh thu theo Tháng")
    monthly_revenue = df_trans_clean.groupby('Month')['Total_Paid'].sum().reset_index()
    
    fig, ax = plt.subplots()
    sns.lineplot(data=monthly_revenue, x='Month', y='Total_Paid', marker='o', color='firebrick', ax=ax)
    ax.set_ylabel("Doanh thu (VND)")
    ax.set_xticks(range(1, 13))
    # Format trục Y dạng tiền tệ
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    st.pyplot(fig)

    # 2.3 Khung giờ cao điểm
    st.subheader("Khung giờ Cao điểm (Peak Hours)")
    hourly_counts = df_trans_clean.groupby('Hour')['Transaction_ID'].count().reset_index()
    
    fig, ax = plt.subplots()
    sns.barplot(data=hourly_counts, x='Hour', y='Transaction_ID', palette='Blues_d', ax=ax)
    ax.set_title("Số lượng đơn hàng theo giờ")
    st.pyplot(fig)

# --------------------------------------------------------
# TAB 3: PROFIT & PRODUCT (PHẦN PHỨC TẠP NHẤT CỦA BẠN)
# --------------------------------------------------------
with tab3:
    st.header("3. Phân tích Lợi nhuận & Hành vi tiêu dùng")
    st.markdown("Phần này kết hợp dữ liệu từ cả 3 bảng: Transaction, Product và Customer.")

    # --- BƯỚC MERGE DỮ LIỆU (Quan trọng) ---
    with st.spinner("Đang thực hiện Merge dữ liệu bảng lớn..."):
        # Merge 1: Trans + Product (Để lấy COGS)
        df_merged = pd.merge(df_trans_clean, df_product, on='Product_ID', how='left')
        
        # Tính Gross Profit (Logic quan trọng trong code cũ của bạn)
        # Profit = (Price_Paid - COGS) * Quantity  <-- Lưu ý: Total_Paid trong data thường đã nhân Quantity hoặc chưa? 
        # Kiểm tra logic: Total_Paid là giá cuối cùng khách trả cho CẢ dòng đó.
        # COGS trong Product Master là giá vốn cho 1 đơn vị.
        df_merged['Total_COGS'] = df_merged['COGS'] * df_merged['Quantity']
        df_merged['Gross_Profit'] = df_merged['Total_Paid'] - df_merged['Total_COGS']
        
        # Merge 2: + Customer (Để lấy Occupation, Gender...)
        df_full = pd.merge(df_merged, df_cust_clean[['Customer_ID', 'Occupation', 'Income_Level', 'Gender']], on='Customer_ID', how='left')
    
    st.success("Đã merge xong dữ liệu!")

    # --- BIỂU ĐỒ 1: MARGIN THEO NGHỀ NGHIỆP ---
    st.subheader("3.1. Biên lợi nhuận gộp trung bình theo Nghề nghiệp")
    st.caption("Biểu đồ này giúp xác định nhóm khách hàng nào mang lại lợi nhuận thực tế cao nhất.")
    
    avg_profit_occ = df_full.groupby('Occupation')['Gross_Profit'].mean().reset_index().sort_values(by='Gross_Profit', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=avg_profit_occ, x='Occupation', y='Gross_Profit', palette='Greens_r', ax=ax)
    ax.set_ylabel("Lợi nhuận gộp TB / Đơn hàng (VND)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- BIỂU ĐỒ 2: TOP SẢN PHẨM (BEST SELLERS) ---
    st.subheader("3.2. Top 10 Sản phẩm bán chạy nhất (Theo số lượng)")
    top_products = df_full.groupby('Product_Name')['Quantity'].sum().nlargest(10).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products.plot(kind='barh', color='orange', ax=ax)
    ax.set_xlabel("Tổng số lượng bán ra")
    st.pyplot(fig)

    # --- BIỂU ĐỒ 3: HEATMAP (NGHỀ NGHIỆP vs DANH MỤC) ---
    st.subheader("3.3. Heatmap: Nhóm nghề nghiệp thích uống gì?")
    st.caption("Tỷ lệ phần trăm số lượng sản phẩm tiêu thụ theo từng nhóm ngành nghề.")

    # Tạo Pivot Table (Crosstab)
    # Normalize='index' để tính % theo hàng (Mỗi nghề nghiệp tổng là 100%)
    heatmap_data = pd.crosstab(
        df_full['Occupation'], 
        df_full['Category'], 
        values=df_full['Quantity'], 
        aggfunc='sum', 
        normalize='index'
    ) * 100 # Nhân 100 để ra %

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax)
    ax.set_title("Tỷ trọng tiêu dùng (%)")
    st.pyplot(fig)

# --------------------------------------------------------
# TAB 4: XEM DỮ LIỆU THÔ
# --------------------------------------------------------
with tab4:
    st.header("Dữ liệu mẫu")
    option = st.selectbox("Chọn bảng dữ liệu:", ["Transaction (Cleaned)", "Customer (Cleaned)", "Product Master"])
    
    if option == "Transaction (Cleaned)":
        st.dataframe(df_trans_clean.head(100))
    elif option == "Customer (Cleaned)":
        st.dataframe(df_cust_clean.head(100))
    else:
        st.dataframe(df_product)