import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ========================================================
# 0. CẤU HÌNH VÀ TẢI DỮ LIỆU CẦN THIẾT
# ========================================================

st.set_page_config(page_title="Price Optimization", page_icon="🎯", layout="wide")
st.title("🎯 Chiến lược Tối ưu hóa Giá Động")
st.caption("Sử dụng Mô hình Hồi quy Log-Log và Toán Tối ưu hóa để tìm ra Giá P*.")


# Dữ liệu giả định: Streamlit cần phải load các dataframes này từ các bước trước
# VÌ KHÔNG CÓ CODE TIỀN XỬ LÝ (Log-Transform), ta phải giả định hàm load data
@st.cache_data(show_spinner="Đang tải dữ liệu tiền xử lý (Log-Transformed, Segmented)...")
def load_optimization_data():
    """
    Hàm này phải tải 3 thành phần CẦN THIẾT từ các bước trước:
    1. df_pivot: Dữ liệu giao dịch đã được Log-transform và gán Segment.
    2. df_prod: Product Master (chứa COGS).
    3. df_txn_all: Transaction gốc (để tính baseline demand).
    
    LƯU Ý QUAN TRỌNG: Bạn phải tự tạo hàm này dựa trên code tiền xử lý (Log-Log)
    và kết quả Clustering của bạn.
    """
    try:
        # Load data gốc
        df_prod = pd.read_csv("data/product_master.csv")
        df_txn_all = pd.read_csv("data/transaction_data.csv")
        df_cust = pd.read_csv("data/customer_profile.csv")
        
        # --- BƯỚC GIẢ ĐỊNH DỮ LIỆU THIẾU ---
        # Giả định dữ liệu df_pivot đã có Log-transform và Segment (CẦN TỪ BƯỚC 3)
        # Vì không có code tiền xử lý Log-Log, ta chỉ tạo một DataFrame giả định cho đủ cột
        
        # Tạo df_pivot giả định (Cần có 3 Segment A, B, C và các cột Ln_...)
        products = df_prod['Product_ID'].tolist()
        num_rows = 500
        
        mock_data = {
            'Segment': np.random.choice(['A', 'B', 'C'], size=num_rows),
        }
        for p in products:
            # Giả định cột Log-Quantity
            mock_data[f"Ln_Quantity_{p}"] = np.random.lognormal(mean=2, sigma=1, size=num_rows)
            # Giả định cột Log-Price
            mock_data[f"Ln_Unit_Price_Listed_{p}"] = np.log(df_prod.loc[df_prod['Product_ID'] == p, 'Unit_Price_List'].iloc[0] * np.random.uniform(0.9, 1.1, size=num_rows))
            
        df_pivot = pd.DataFrame(mock_data)

        # Giả định cột Segment trên df_txn_all (từ Clustering)
        df_txn_all['Segment'] = np.random.choice(['A', 'B', 'C'], size=len(df_txn_all))
        df_txn_all['Date'] = pd.to_datetime(df_txn_all['Date_Time']).dt.date
        df_txn_all['Year'] = pd.to_datetime(df_txn_all['Date_Time']).dt.year
        
        return df_pivot, df_prod, df_txn_all
    
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu. Vui lòng kiểm tra lại: {e}")
        st.stop()

df_pivot, df_prod, df_txn_all = load_optimization_data()

# --------------------------------------------------------
# CÁC HẰNG SỐ ĐƯỢC ĐỊNH NGHĨA TRONG CODE GỐC CỦA BẠN
# --------------------------------------------------------
FALLBACK_ELASTICITY = {
    'A': -2.5,  # Segment A: Nhạy cảm giá (Sinh viên)
    'B': -1.3,  # Segment B: Trung bình (Văn phòng)
    'C': -0.5   # Segment C: Thấp (VIP/Trung thành)
}
product_groups = df_prod.groupby('Category')['Product_ID'].apply(list).to_dict()
product_list = df_prod['Product_ID'].tolist()


# ========================================================
# 1. TÍNH TOÁN HỆ SỐ CO GIÃN (ELASTICITY MODELING)
# ========================================================

@st.cache_data(show_spinner="1. Đang huấn luyện mô hình Elasticity (Hồi quy Log-Log) & Áp dụng Smart Fallback...")
def calculate_elasticity(df_pivot, df_prod, fallback_map):
    """
    Tính toán hệ số co giãn giá (Own & Cross) dựa trên mô hình Linear Regression
    và áp dụng Fallback/Clamping như logic gốc.
    """
    elasticity_results = []
    segments = df_pivot['Segment'].unique()
    product_groups = df_prod.groupby('Category')['Product_ID'].apply(list).to_dict()

    for seg in segments:
        df_seg_data = df_pivot[df_pivot['Segment'] == seg].copy()
        if len(df_seg_data) < 10: continue
        
        for category, prod_list in product_groups.items():
            valid_prods = [p for p in prod_list if f"Ln_Quantity_{p}" in df_seg_data.columns]
            if len(valid_prods) < 1: continue

            x_cols = [f"Ln_Unit_Price_Listed_{p}" for p in valid_prods]
            X = df_seg_data[x_cols]
            
            for target_prod in valid_prods:
                y_col = f"Ln_Quantity_{target_prod}"
                y = df_seg_data[y_col]
                
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    for i, driver_prod in enumerate(valid_prods):
                        coef = model.coef_[i]
                        
                        if driver_prod == target_prod:
                            el_type = "Own-Price"
                            note = "Machine Learning"
                            
                            # --- FIX LỖI TẠI ĐÂY (SMART GUARDRAILS TỪ CODE GỐC) ---
                            if coef > 0: 
                                coef = fallback_map.get(seg, -1.0)
                                note = "Fallback (Positive Coef)"
                            elif coef < -5:
                                coef = -4.0 # Clamping giá trị quá extreme
                                note = "Clamped (Extreme)"
                                
                        else:
                            el_type = "Cross-Price"
                            note = "ML"
                            
                        elasticity_results.append({
                            'Segment': seg,
                            'Category': category,
                            'Product_ID': target_prod,
                            'Driver_Product': driver_prod,
                            'Elasticity': coef,
                            'Type': el_type,
                            'Source': note
                        })
                except Exception:
                    # Bỏ qua nếu mô hình không hội tụ (hoặc lỗi dữ liệu)
                    pass
    
    return pd.DataFrame(elasticity_results)

df_elasticity = calculate_elasticity(df_pivot, df_prod, FALLBACK_ELASTICITY)

# --------------------------------------------------------
# 1.1 HIỂN THỊ KẾT QUẢ ELASTICITY
# --------------------------------------------------------
st.subheader("1.2. Tổng hợp Hệ số Co giãn Giá (Own-Price Elasticity)")

df_own_price = df_elasticity[df_elasticity['Type']=='Own-Price'].copy()
summary_table = df_own_price.groupby(['Category', 'Segment'])['Elasticity'].mean().unstack()

st.dataframe(
    summary_table.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), 
    use_container_width=True
)
st.caption("Màu xanh đậm: Ít nhạy cảm giá (cơ hội tăng giá). Màu đỏ đậm: Rất nhạy cảm giá.")

with st.expander("Xem chi tiết các cặp Co giãn chéo (Cross-Price)"):
    st.dataframe(df_elasticity.head(50), use_container_width=True)


# ========================================================
# 2. CẤU HÌNH RÀNG BUỘC & TỐI ƯU HÓA
# ========================================================
st.header("⚙️ Bước 2: Cấu hình và Chạy Mô hình Tối ưu hóa")

# --- 2.1 CẤU HÌNH TƯƠNG TÁC (TỪ CÁC HẰNG SỐ GỐC) ---
with st.sidebar:
    st.subheader("Tham số Tối ưu hóa (Reality Check)")
    MAX_PRICE_INCREASE = st.slider("Tăng giá TỐI ĐA (%)", 1.05, 1.50, 1.20, 0.01)
    MAX_PRICE_DECREASE = st.slider("Giảm giá TỐI ĐA (%)", 0.70, 0.99, 0.90, 0.01)
    MAX_DEMAND_GROWTH = st.slider("Sản lượng tăng TỐI ĐA (%)", 1.10, 1.50, 1.25, 0.01)
    ELASTICITY_DAMPING = st.slider("Hệ số Giảm chấn Elasticity (Damping)", 0.5, 1.0, 0.80, 0.05)
    
    # Ràng buộc sản lượng tổng thể
    MAX_VOLUME_DROP = st.slider("Sản lượng TỔNG thể SỤT GIẢM TỐI ĐA (%)", 0.0, 0.20, 0.05, 0.01)


@st.cache_data(show_spinner=False)
def calculate_baseline_demand(df_txn_all, df_prod):
    """Tính toán Baseline Demand theo Segment và Product ID"""
    df_2024_clean = df_txn_all[df_txn_all['Year'] == 2024].copy()
    n_days_2024 = df_2024_clean['Date'].nunique()
    
    # Tính tổng Quantity theo Segment và Product
    q_sum = df_2024_clean.groupby(['Segment', 'Product_ID'])['Quantity'].sum().reset_index()
    
    # Baseline là Quantiy TB / ngày
    base_demand_dict = q_sum.set_index(['Segment', 'Product_ID'])['Quantity'].apply(lambda x: x / max(1, n_days_2024)).to_dict()
    
    return base_demand_dict

base_demand_dict = calculate_baseline_demand(df_txn_all, df_prod)


# ========================================================
# 3. HÀM MỤC TIÊU & RÀNG BUỘC
# ========================================================

@st.cache_resource
def run_optimization(df_elasticity, df_prod, base_demand_dict, 
                     MAX_PRICE_INCREASE, MAX_PRICE_DECREASE, 
                     MAX_DEMAND_GROWTH, ELASTICITY_DAMPING, 
                     MAX_VOLUME_DROP):
    
    """Chạy mô hình tối ưu hóa chính"""
    st.info("Đang chạy mô hình Tối ưu hóa (SLSQP)...")
    
    product_list = df_prod['Product_ID'].tolist()
    n_products = len(product_list)
    prod_to_idx = {p: i for i, p in enumerate(product_list)}
    
    df_prod_optim = df_prod.set_index('Product_ID').reindex(product_list)
    p_base_arr = df_prod_optim['Unit_Price_List'].values
    cost_arr = df_prod_optim['COGS'].values
    segments = df_elasticity['Segment'].unique()
    
    # 3.1 Lookup Table Elasticity (Có áp dụng Damping)
    elasticity_lookup = {}
    for seg in segments:
        elasticity_lookup[seg] = {}
        df_e_seg = df_elasticity[df_elasticity['Segment'] == seg]
        for _, row in df_e_seg.iterrows():
            t_idx = prod_to_idx.get(row['Product_ID'])
            d_idx = prod_to_idx.get(row['Driver_Product'])
            if t_idx is not None and d_idx is not None:
                if t_idx not in elasticity_lookup[seg]: elasticity_lookup[seg][t_idx] = {}
                # Damping applied
                elasticity_lookup[seg][t_idx][d_idx] = row['Elasticity'] * ELASTICITY_DAMPING
    
    
    # 3.2 HÀM MỤC TIÊU (MAX PROFIT) - GIỮ NGUYÊN LOGIC CỦA BẠN
    def objective_function(p_new_arr):
        total_profit = 0
        price_ratios = p_new_arr / (p_base_arr + 1e-9)
        
        for seg in segments:
            for i in range(n_products):
                prod_id = product_list[i]
                q_base = base_demand_dict.get((seg, prod_id), 0)
                if q_base <= 0: continue
                
                multiplier = 1.0
                if i in elasticity_lookup[seg]:
                    for driver_idx, e_val in elasticity_lookup[seg][i].items():
                        ratio = price_ratios[driver_idx]
                        multiplier *= (ratio ** e_val)
                
                # Giới hạn trần sản lượng
                multiplier = min(multiplier, MAX_DEMAND_GROWTH)
                multiplier = max(multiplier, 0.2) # Giới hạn sàn 20%
                
                q_new = q_base * multiplier
                margin = p_new_arr[i] - cost_arr[i]
                total_profit += margin * q_new
                
        return -total_profit # Minimize negative profit (Maximize Profit)

    # 3.3 RÀNG BUỘC (CONSTRAINTS) - GIỮ NGUYÊN LOGIC CỦA BẠN
    constraints = []
    
    # Ràng buộc 1: Cấu trúc giá (M > S, L > M)
    product_families = {}
    for p in product_list:
        if '_' in p:
            root, size = p.rsplit('_', 1)
            if root not in product_families: product_families[root] = {}
            product_families[root][size] = prod_to_idx[p]
    
    for root, sizes in product_families.items():
        # M > S (Giá M lớn hơn giá S ít nhất 3000 VND)
        if 'M' in sizes and 'S' in sizes:
            idx_m, idx_s = sizes['M'], sizes['S']
            constraints.append({'type': 'ineq', 'fun': lambda x, m=idx_m, s=idx_s: x[m] - x[s] - 3000})
        # L > M
        if 'L' in sizes and 'M' in sizes:
            idx_l, idx_m = sizes['L'], sizes['M']
            constraints.append({'type': 'ineq', 'fun': lambda x, l=idx_l, m=idx_m: x[l] - x[m] - 3000})

    # Ràng buộc 2: Tổng sản lượng sụt giảm tối đa (Áp dụng tham số MAX_VOLUME_DROP)
    total_base_volume = sum(base_demand_dict.values())
    
    def volume_constraint(p_new_arr):
        total_new_volume = 0
        price_ratios = p_new_arr / (p_base_arr + 1e-9)
        
        for seg in segments:
            for i in range(n_products):
                prod_id = product_list[i]
                q_base = base_demand_dict.get((seg, prod_id), 0)
                if q_base <= 0: continue
                
                multiplier = 1.0
                if i in elasticity_lookup[seg]:
                    for driver_idx, e_val in elasticity_lookup[seg][i].items():
                        ratio = price_ratios[driver_idx]
                        multiplier *= (ratio ** e_val)
                
                multiplier = min(multiplier, MAX_DEMAND_GROWTH)
                multiplier = max(multiplier, 0.2)
                
                total_new_volume += q_base * multiplier
        
        # Ràng buộc: Total_New_Volume >= Total_Base_Volume * (1 - MAX_VOLUME_DROP)
        return total_new_volume - (total_base_volume * (1 - MAX_VOLUME_DROP))

    constraints.append({'type': 'ineq', 'fun': volume_constraint, 'name': 'Volume_Guardrail'})
    
    # 3.4 Giới hạn giá (Bounds) - Giữ nguyên logic của bạn
    bounds = []
    for i in range(n_products):
        # Sàn: Giá vốn + 15% HOẶC Giảm tối đa (MAX_PRICE_DECREASE)
        lower = max(cost_arr[i] * 1.15, p_base_arr[i] * MAX_PRICE_DECREASE)
        # Trần: Tăng tối đa (MAX_PRICE_INCREASE)
        upper = p_base_arr[i] * MAX_PRICE_INCREASE
        
        if lower > upper: lower = upper - 500 # Safety check
        bounds.append((lower, upper))
        
    # 3.5 CHẠY TỐI ƯU HÓA
    result = minimize(
        objective_function,
        p_base_arr,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-6, 'disp': False, 'maxiter': 1000} 
    )
    
    return result, objective_function, p_base_arr, cost_arr, product_list, df_prod_optim

# Chạy mô hình khi nút bấm được kích hoạt
if st.button("🚀 Chạy Mô hình Tối ưu hóa Giá", type="primary"):
    result, obj_func, p_base_arr, cost_arr, product_list, df_prod_optim = run_optimization(
        df_elasticity, df_prod, base_demand_dict, 
        MAX_PRICE_INCREASE, MAX_PRICE_DECREASE, 
        MAX_DEMAND_GROWTH, ELASTICITY_DAMPING, 
        MAX_VOLUME_DROP
    )
    
    st.markdown("---")
    st.header("📈 Kết quả Tối ưu hóa")

    # --------------------------------------------------------
    # 4. BÁO CÁO & VISUALIZE
    # --------------------------------------------------------
    if result.success:
        st.success(f"✅ Tối ưu hóa thành công! Solver dừng: {result.message}")
        optimal_prices = result.x
    else:
        st.warning(f"⚠️ Solver dừng: {result.message}. Kết quả có thể không tối ưu hoàn toàn.")
        optimal_prices = p_base_arr # Dùng giá cũ nếu tối ưu thất bại

    # Tính toán kết quả
    profit_old = -obj_func(p_base_arr)
    profit_new = -obj_func(optimal_prices)
    uplift_pct = ((profit_new - profit_old) / profit_old) * 100
    
    # --- HIỂN THỊ METRICS ---
    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
    col_kpi_1.metric("Lợi nhuận Gốc (Ngày)", f"{profit_old:,.0f} VND")
    col_kpi_2.metric("Lợi nhuận Tối ưu (Ngày)", f"{profit_new:,.0f} VND")
    col_kpi_3.metric("Uplift Lợi nhuận", f"+{uplift_pct:.2f}%", delta_color="normal")
    
    st.markdown("---")

    # --- TẠO BẢNG KẾT QUẢ ---
    df_result = pd.DataFrame({
        'Product_ID': product_list,
        'Category': df_prod_optim['Category'].values,
        'Old_Price': p_base_arr,
        'New_Price_Raw': optimal_prices,
        'Cost': cost_arr
    })
    
    # Làm tròn 500 đồng (Việt Nam style) - GIỮ NGUYÊN LOGIC CỦA BẠN
    def round_price(x): return round(x / 1000) * 1000
    df_result['New_Price'] = df_result['New_Price_Raw'].apply(round_price)
    df_result['Change %'] = (df_result['New_Price'] - df_result['Old_Price']) / df_result['Old_Price'] * 100
    
    st.subheader("📋 Chi tiết Giá đề xuất Tối ưu")
    st.dataframe(
        df_result[['Product_ID', 'Old_Price', 'New_Price', 'Cost', 'Change %']].sort_values('Change %', ascending=False).set_index('Product_ID'),
        use_container_width=True
    )

    # --- VISUALIZE ---
    st.subheader("📊 So sánh Giá cũ, Giá mới và Giá vốn")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(product_list))
    width = 0.4
    
    ax.bar(x - width/2, df_result['Old_Price'], width, label='Giá Cũ', color='#95a5a6', alpha=0.7)
    ax.bar(x + width/2, df_result['New_Price'], width, label='GIÁ TỐI ƯU', color='#27ae60')
    ax.plot(x, df_result['Cost'], color='#c0392b', marker='o', linestyle='--', linewidth=2, label='Giá Vốn (COGS)')
    
    ax.set_ylabel('Giá (VND)')
    ax.set_title(f'Tối ưu hóa Giá: Uplift +{uplift_pct:.1f}%', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_result['Product_ID'], rotation=90)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Nhãn phần trăm (Giữ nguyên logic của bạn)
    for i, v in enumerate(df_result['Change %']):
        if abs(v) > 0.1:
            color = 'darkgreen' if v > 0 else 'darkred'
            ax.text(i + width/2, df_result['New_Price'][i] + 1500, f"{v:+.1f}%", 
                    ha='center', va='bottom', fontsize=10, color=color, rotation=90, weight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)