import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle # Cần thiết để save/load model nếu không dùng @st.cache_resource

# Thư viện Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Thư viện Mô hình Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ========================================================
# 0. STREAMLIT CONFIGURATION
# ========================================================
st.set_page_config(page_title="Phân khúc Khách hàng", page_icon="👥", layout="wide")
st.title("👥 Mô hình Phân khúc Khách hàng (K-Means)")
st.caption("Dựa trên các đặc trưng RFM, Hành vi giao dịch và Hồ sơ cá nhân.")

# Cài đặt hiển thị (chỉ áp dụng cho code)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ========================================================
# 1. TẢI VÀ KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)
# ========================================================

@st.cache_data
def load_and_engineer_features(data_dir="data"):
    """
    Tải dữ liệu, xây dựng các đặc trưng RFM và hành vi.
    Sử dụng @st.cache_data để caching dữ liệu đầu ra (df_analysis).
    """
    st.info(f"Đang tải dữ liệu và xây dựng đặc trưng từ thư mục '{data_dir}'...")
    try:
        # Sử dụng đường dẫn tương đối
        df_trans = pd.read_csv(f'{data_dir}/transaction_data.csv')
        df_cust = pd.read_csv(f'{data_dir}/customer_profile.csv')
        
        # Chuyển đổi kiểu dữ liệu ngày tháng
        df_trans['Date_Time'] = pd.to_datetime(df_trans['Date_Time'])
        
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy tệp CSV. Vui lòng kiểm tra thư mục '{data_dir}' trên GitHub/Local.")
        st.stop()

    # 1.1. Xây dựng đặc trưng RFM
    snapshot_date = df_trans['Date_Time'].max() + pd.Timedelta(days=1)
    
    rfm_df = df_trans.groupby('Customer_ID').agg(
        Recency=('Date_Time', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Transaction_ID', 'nunique'),
        Monetary=('Total_Paid', 'sum')
    ).reset_index()
    
    # 1.2. Xây dựng thêm các đặc trưng hành vi (Giữ nguyên logic của bạn)
    df_trans['Is_Discount'] = (df_trans['Discount_Amount'] > 0).astype(int)
    df_trans['Is_Weekend'] = df_trans['Date_Time'].dt.dayofweek >= 5
    
    # Suy Category từ Product_ID (Giữ nguyên hàm của bạn)
    def map_category(pid):
        if isinstance(pid, str):
            if pid.startswith('CF'):
                return 'Coffee'
            elif pid.startswith('TE'):
                return 'Tea'
            elif pid.startswith('FR'):
                return 'Freeze'
        return 'Other'
    
    df_trans['Category'] = df_trans['Product_ID'].apply(map_category)
    
    cust_txn_features = df_trans.groupby('Customer_ID').agg(
        Discount_Usage=('Is_Discount', 'mean'), 
        Weekend_Visit_Rate=('Is_Weekend', 'mean'),
        Coffee_Share=('Category', lambda x: (x == 'Coffee').mean()),
        Tea_Share=('Category', lambda x: (x == 'Tea').mean()),
        Freeze_Share=('Category', lambda x: (x == 'Freeze').mean())
    ).reset_index()
    
    def preferred_category(row):
        shares = {'Coffee': row['Coffee_Share'], 'Tea': row['Tea_Share'], 'Freeze': row['Freeze_Share']}
        if max(shares.values()) == 0:
            return 'Unknown'
        return max(shares, key=shares.get)
    
    cust_txn_features['Preferred_Category'] = cust_txn_features.apply(preferred_category, axis=1)
    
    # 1.3. Kết hợp và Xây dựng đặc trưng Hồ sơ (Profile)
    df_analysis = pd.merge(df_cust, rfm_df, on='Customer_ID', how='left')
    df_analysis = pd.merge(df_analysis, cust_txn_features, on='Customer_ID', how='left')
    
    # Xử lý thiếu (điền 0 hoặc 999 cho khách hàng không có giao dịch)
    df_analysis['Recency'] = df_analysis['Recency'].fillna(999) 
    df_analysis[['Frequency', 'Monetary']] = df_analysis[['Frequency', 'Monetary']].fillna(0)
    behaviour_cols = ['Discount_Usage', 'Weekend_Visit_Rate', 'Coffee_Share', 'Tea_Share', 'Freeze_Share']
    df_analysis[behaviour_cols] = df_analysis[behaviour_cols].fillna(0)
    df_analysis['Preferred_Category'] = df_analysis['Preferred_Category'].fillna('Unknown')
    
    # Tạo đặc trưng 'Age'
    current_year = snapshot_date.year
    df_analysis['Age'] = current_year - df_analysis['YoB']
    
    # Xử lý ngoại lệ (Capping ở Phân vị 99%)
    for col in ['Recency','Frequency','Monetary']:
        cap_value = df_analysis[col].quantile(0.99)
        df_analysis[col] = df_analysis[col].clip(upper=cap_value)
    
    # 1.4. Chọn đặc trưng cuối cùng
    features_to_cluster = [
        'Age', 'Recency', 'Frequency', 'Monetary',
        'Income level', 'Membership_Tier',
        'Occupation', 'Gender'
    ]
    df_model_input = df_analysis[features_to_cluster].copy()
    
    st.success(f"Hoàn tất BƯỚC 1. Dữ liệu đầu vào có {df_model_input.shape[0]} khách hàng và {df_model_input.shape[1]} đặc trưng.")
    return df_analysis, df_model_input

df_analysis, df_model_input = load_and_engineer_features()


# ========================================================
# 2. TIỀN XỬ LÝ (PREPROCESSING PIPELINE)
# ========================================================

@st.cache_resource
def build_and_fit_preprocessor(df_input):
    """
    Xây dựng và huấn luyện Pipeline tiền xử lý.
    Sử dụng @st.cache_resource vì Preprocessor là một Model/Đối tượng nặng.
    """
    st.info("Đang xây dựng và huấn luyện Pipeline Mã hóa & Chuẩn hóa...")
    
    # Định nghĩa các nhóm cột (Giữ nguyên định nghĩa của bạn)
    numerical_features = ['Age', 'Recency', 'Frequency', 'Monetary']
    income_levels = ['< 2M', '2-5M', '5-10M', '10-20M', '20-50M', '> 50M']
    membership_tiers = ['Standard', 'Silver', 'Gold', 'Diamond']
    ordinal_features = ['Income level', 'Membership_Tier']
    nominal_features = ['Occupation', 'Gender']
    
    # Xây dựng các pipeline con (Giữ nguyên logic của bạn)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[income_levels, membership_tiers],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Kết hợp các pipeline con
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features)
        ],
        remainder='passthrough'
    )
    
    # Áp dụng preprocessor
    X_scaled = preprocessor.fit_transform(df_input)
    
    try:
        feature_names = preprocessor.get_feature_names_out()
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    except Exception:
        X_scaled_df = pd.DataFrame(X_scaled)

    st.success(f"Pipeline tiền xử lý hoàn tất. Dữ liệu đã chuẩn hóa có {X_scaled_df.shape[1]} đặc trưng.")
    return preprocessor, X_scaled_df

preprocessor, X_scaled_df = build_and_fit_preprocessor(df_model_input)
X_scaled = X_scaled_df.values # Lấy lại mảng numpy để tính toán

# ========================================================
# 3. XÁC ĐỊNH SỐ CỤM TỐI ƯU (K)
# ========================================================

st.header("🎯 Bước 3: Xác định Số cụm Tối ưu (K)")
st.caption("Dùng phương pháp Elbow và Silhouette Score để tìm K phù hợp.")

@st.cache_data
def calculate_optimal_k(X_data, k_range=range(2, 9)):
    """
    Tính toán WCSS (Inertia) và Silhouette Score cho các K khác nhau.
    """
    inertia_values = []
    silhouette_scores = []
    
    status_text = st.empty()
    for i, k in enumerate(k_range):
        status_text.text(f"Đang chạy K-Means với K={k} ({i+1}/{len(k_range)})...")
        kmeans_test = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans_test.fit(X_data)
        
        inertia_values.append(kmeans_test.inertia_)
        score = silhouette_score(X_data, kmeans_test.labels_)
        silhouette_scores.append(score)
    status_text.success("Hoàn tất tính toán WCSS (Inertia) và Silhouette.")
    
    return list(k_range), inertia_values, silhouette_scores

K_range, inertia_values, silhouette_scores = calculate_optimal_k(X_scaled)

# 3.1. Vẽ biểu đồ Elbow và Silhouette (Giữ nguyên logic vẽ của bạn)
col_elbow, col_silhouette = st.columns(2)

with col_elbow:
    st.subheader("3.1. Phương pháp Elbow")
    fig, ax = plt.subplots()
    ax.plot(K_range, inertia_values, 'bo-')
    ax.set_xlabel('Số cụm (K)')
    ax.set_ylabel('WCSS (Inertia)')
    ax.set_title('Phương pháp Elbow (Elbow Method)')
    ax.grid(True)
    st.pyplot(fig)

with col_silhouette:
    st.subheader("3.2. Chỉ số Silhouette")
    fig, ax = plt.subplots()
    ax.plot(K_range, silhouette_scores, 'rs-')
    ax.set_xlabel('Số cụm (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Chỉ số Silhouette')
    ax.grid(True)
    st.pyplot(fig)


# ========================================================
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH K-MEANS
# ========================================================

st.header("📊 Bước 4: Huấn luyện Mô hình Cuối cùng")

# Widget để người dùng chọn K (Hoặc giữ K=3)
N_CLUSTERS = st.slider("Chọn số cụm cuối cùng (N_CLUSTERS):", min_value=2, max_value=8, value=3)

@st.cache_resource
def train_final_model(X_data, n_clusters):
    """
    Huấn luyện mô hình K-Means cuối cùng.
    Sử dụng @st.cache_resource để caching model object.
    """
    model_name = "K-Means"
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    
    with st.spinner(f"Đang huấn luyện mô hình {model_name} với K={n_clusters}..."):
        labels = model.fit_predict(X_data)
    
    # Đánh giá mô hình
    silhouette = silhouette_score(X_data, labels)
    davies_bouldin = davies_bouldin_score(X_data, labels)
    
    results = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin
    }
    
    return model, labels, results

model, labels, results = train_final_model(X_scaled, N_CLUSTERS)

st.subheader("4.1. Kết quả Đánh giá Mô hình")
results_df = pd.DataFrame(results, index=["K-Means"]).T
st.dataframe(results_df)

st.success(f"Mô hình K-Means với K={N_CLUSTERS} đã được huấn luyện thành công!")


# ========================================================
# 5. GIẢM CHIỀU DỮ LIỆU BẰNG PCA & TRỰC QUAN HÓA
# ========================================================

st.header("✨ Bước 5: Trực quan hóa Cụm (PCA)")

@st.cache_data
def run_pca_and_format(X_data, labels, model_name):
    """
    Giảm chiều dữ liệu và tạo DataFrame cho biểu đồ scatter.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_data)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca[model_name] = labels
    return df_pca

df_pca = run_pca_and_format(X_scaled, labels, "K-Means")

# Vẽ biểu đồ cụm (Giữ nguyên logic vẽ của bạn)
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='K-Means', data=df_pca,
                palette='viridis', legend='full', alpha=0.7, ax=ax)
ax.set_title(f'K-Means Clustering - K={N_CLUSTERS} (PCA 2D)')
ax.grid(True)
st.pyplot(fig)


# ========================================================
# 6. PHÂN TÍCH HỒ SƠ KHÁCH HÀNG THEO CỤM
# ========================================================

st.header("📝 Bước 6: Phân tích Hồ sơ Cụm (Cluster Profiling)")

def mode_or_na(x):
    """Giữ nguyên hàm của bạn: Lấy mode hoặc trả về N/A."""
    m = x.mode()
    return m.iloc[0] if not m.empty else 'N/A'

@st.cache_data(show_spinner=False)
def generate_cluster_summary(df_original, labels, n_clusters):
    """
    Tạo bảng tóm tắt đặc điểm chi tiết của từng cụm.
    """
    df_analysis_labeled = df_original.copy()
    df_analysis_labeled['Cluster'] = labels
    
    # 6.1. Cluster size
    cluster_size = df_analysis_labeled.groupby('Cluster')['Customer_ID'].count().rename('Cluster_Size')
    
    # 6.2. Các đặc trưng số trung bình (Giữ nguyên định nghĩa của bạn)
    numeric_profile_cols = [
        'Age', 'Recency', 'Frequency', 'Monetary',
        'Discount_Usage', 'Weekend_Visit_Rate',
        'Coffee_Share', 'Tea_Share', 'Freeze_Share'
    ]
    cluster_numeric = df_analysis_labeled.groupby('Cluster')[numeric_profile_cols].mean()
    
    # 6.3. Các đặc trưng phân loại (Mode)
    cluster_categorical = df_analysis_labeled.groupby('Cluster').agg({
        'Income level': mode_or_na,
        'Membership_Tier': mode_or_na,
        'Occupation': mode_or_na,
        'Gender': mode_or_na,
        'Preferred_Category': mode_or_na
    })
    
    cluster_categorical = cluster_categorical.rename(columns={
        'Income level': 'Income_Level_Mode',
        'Membership_Tier':'Membership_Tier_Mode',
        'Occupation': 'Occupation_Mode',
        'Gender': 'Gender_Mode',
        'Preferred_Category':'Preferred_Category_Mode'
    })
    
    # 6.4. Gộp tất cả thành một bảng
    cluster_summary = pd.concat([cluster_size, cluster_numeric, cluster_categorical], axis=1).reset_index()
    
    # Chuyển các tỷ lệ sang % (Giữ nguyên logic của bạn)
    ratio_cols = ['Discount_Usage', 'Weekend_Visit_Rate', 'Coffee_Share', 'Tea_Share', 'Freeze_Share']
    cluster_summary[ratio_cols] = cluster_summary[ratio_cols] * 100
    
    return cluster_summary

cluster_summary = generate_cluster_summary(df_analysis, labels, N_CLUSTERS)

st.subheader(f"6.1. Bảng Tóm tắt Đặc điểm Cụm (K={N_CLUSTERS})")
st.dataframe(cluster_summary)

st.info("💡 Lưu ý: Cột **Cluster_Size** thể hiện số lượng khách hàng trong cụm. Các cột **Share** và **Usage** được tính theo %.")
st.success("Hoàn tất phân tích Clustering. Mô hình và kết quả tóm tắt đã sẵn sàng cho báo cáo.")

# ========================================================
# LƯU MODEL (CHO CÁC BƯỚC SAU)
# ========================================================
# *QUAN TRỌNG:* Lưu mô hình K-Means và Preprocessor vào thư mục 'models/'
# để các bước Price Optimization (Elasticity) có thể tái sử dụng mà không cần huấn luyện lại.
# Dùng `pickle.dump` để lưu file .pkl (như đã hướng dẫn trước đó).
# Bạn cần chạy đoạn này *một lần* ở local nếu muốn tái sử dụng mô hình trong các file Streamlit khác.
# st.write("Mô hình đã được lưu cache. Sẵn sàng cho Price Optimization.")