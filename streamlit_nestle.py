# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:19:35 2026

@author: i-belt
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

# ===== 1. 页面配置与字体处理 =====
st.set_page_config(page_title="UHT 生产线 PCA 异常分析", layout="wide")

# 设置绘图字体 (Streamlit 环境通常建议使用系统默认或通过配置加载)
plt.rcParams['axes.unicode_minus'] = False 

# ===== 2. 核心函数定义 =====

@st.cache_data
def get_data(host, port, dbname, user, password):
    """缓存数据库读取结果，避免每次操作都重连数据库"""
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        df = pd.read_sql("SELECT * FROM results", conn)
        conn.close()
        return df, "成功读取数据"
    except Exception as e:
        return None, f"数据库连接失败: {e}"

def plot_cov_ellipse(points, nstd=3, ax=None, **kwargs):
    if len(points) < 3: return None
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(np.maximum(vals, 0))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    if ax is None: ax = plt.gca()
    ax.add_artist(ellip)
    return ellip

def create_pca_plot(df_input, pcs=(1, 2), nstd=3, arrow_scale=1.0):
    """生成 PCA 双标图并返回 fig 对象"""
    idx1, idx2 = pcs[0] - 1, pcs[1] - 1
    
    # 数据清洗与标准化
    X = df_input.drop(columns=['DateTime', 'tag'])
    y_tags = df_input['tag'].values
    feature_names = X.columns
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    
    # PCA 计算
    n_components = max(pcs)
    pca = PCA(n_components=n_components)
    all_scores = pca.fit_transform(X_scaled)
    all_eigvals = pca.explained_variance_
    all_loadings = pca.components_.T * np.sqrt(all_eigvals)
    all_ratios = pca.explained_variance_ratio_
    
    # 提取选定轴数据
    scores = all_scores[:, [idx1, idx2]]
    loadings = all_loadings[:, [idx1, idx2]]
    ratios = [all_ratios[idx1], all_ratios[idx2]]
    eigvals = [all_eigvals[idx1], all_eigvals[idx2]]
    
    n_samples = X_scaled.shape[0]
    scores_scaled = scores / (np.sqrt(eigvals) * np.sqrt(n_samples - 1))
    ratio_factor = np.max(np.abs(loadings)) / np.max(np.abs(scores_scaled)) * arrow_scale
    final_scores = scores_scaled * ratio_factor

    # 绘图逻辑
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.axhline(0, color='#cccccc', lw=1)
    ax.axvline(0, color='#cccccc', lw=1)

    circle_radius = np.max(np.linalg.norm(loadings, axis=1))
    circle = plt.Circle((0, 0), circle_radius, color='gray', fill=False, linestyle='--', lw=1.5, alpha=0.5)
    ax.add_patch(circle)

    unique_tags = np.unique(y_tags)
    color_map = cm.get_cmap('Dark2', len(unique_tags))
    
    for i, tag in enumerate(unique_tags):
        mask = (y_tags == tag)
        tag_points = final_scores[mask]
        color = color_map(i)
        ax.scatter(tag_points[:, 0], tag_points[:, 1], color=color, label=tag, alpha=0.7, s=40, edgecolors='w', zorder=3)
        if len(tag_points) > 2:
            plot_cov_ellipse(tag_points, nstd=nstd, ax=ax, color=color, alpha=0.1, zorder=2)

    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.7, head_width=circle_radius*0.02, lw=1.5, zorder=5)
        ax.text(loadings[i, 0] * 1.05, loadings[i, 1] * 1.05, name, color='darkred', fontsize=9, weight='bold', zorder=6)

    ax.set_title(f"PCA : PC{pcs[0]} vs PC{pcs[1]}", fontsize=15)
    ax.set_xlabel(f"PC{pcs[0]} ({ratios[0]:.2%})", fontsize=12)
    ax.set_ylabel(f"PC{pcs[1]} ({ratios[1]:.2%})", fontsize=12)
    ax.axis('equal')
    limit = circle_radius * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    return fig

# ===== 3. Streamlit UI 侧边栏 =====
st.sidebar.header("📡 数据库配置")
db_host = st.sidebar.text_input("Host", "localhost")
db_name = st.sidebar.text_input("Database", "nestla")
db_user = st.sidebar.text_input("User", "postgres")
db_pwd = st.sidebar.text_input("Password", "admin", type="password")

st.sidebar.header("⚙️ PCA parameter Control")
pc_x = st.sidebar.selectbox("Select the X-axis", [1, 2, 3, 4], index=0)
pc_y = st.sidebar.selectbox("Select the y-axis", [1, 2, 3, 4], index=1)
nstd_val = st.sidebar.slider("confidence level of the ellipse (nstd)", 1.0, 5.0, 3.0)
arrow_scale = st.sidebar.slider("arrow length scaling", 0.5, 2.0, 1.0)

# ===== 4. 主程序逻辑 =====
st.title("🏭 UHT Production Line Process Anomaly PCA Dashboard")

# 获取数据
df_raw, msg = get_data(db_host, 5432, db_name, db_user, db_pwd)

if df_raw is not None:
    st.success(f"Data loaded successfully！Sampling count: {len(df_raw)}")
    
    # --- 数据处理逻辑 (保留你的业务逻辑) ---
    cols = [
        "UHTRoom_VTIS3_FT04.PV", "VTIS03_TI044_TE_Value.PV", "VTIS03_TI008_TE_Value.PV",
        "UHTRoom_VTIS3_V44.PV", "UHTRoom_VTIS3_PT313.PV", "UHTRoom_VTIS3_TT344.PV",
        "VTIS03_HC_IND_Time.PV", "UHTRoom_VTIS3_TT342.PV", "UHTRoom_VTIS3_V344.PV",
        "VTIS03_TTA_AO_V330_AN.PV", "UHTRoom_VTIS3_TT331.PV", "UHTRoom_VTIS3_P330.PV",
        "VTIS03_PT006_PT_Value.PV", "VTIS03_DPI61_PT_Value.PV", "VTIS03_DPI62_PT_Value.PV",
        "RSV", "DateTime"
    ]
    
    df_filtered = df_raw[cols].copy()
    df_filtered['DateTime'] = pd.to_datetime(df_filtered['DateTime'])
    
    # 标签段配置
    periods = [
        {"label": "normal1", "start": "2025-09-20 02:14", "end": "2025-09-20 03:14"},
        {"label": "CIP1_before", "start": "2025-09-21 05:00", "end": "2025-09-21 06:00"},
        {"label": "CIP1_after", "start": "2025-09-21 15:10", "end": "2025-09-21 16:10"},
        {"label": "err1", "start": "2025-09-21 22:00", "end": "2025-09-21 23:00"},
        {"label": "normal2", "start": "2025-09-26 08:00", "end": "2025-09-26 09:00"},
        {"label": "err2", "start": "2025-09-26 12:10", "end": "2025-09-26 13:10"},
    ]

    df_list = []
    for p in periods:
        mask = (df_filtered['DateTime'] >= p['start']) & (df_filtered['DateTime'] <= p['end'])
        temp_df = df_filtered[mask].copy()
        temp_df['tag'] = p['label']
        df_list.append(temp_df)

    result_df = pd.concat(df_list, ignore_index=True)

    # --- 显示 PCA 图表 ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("PCA Biplot (Score Plot + Loading Plot)")
        fig = create_pca_plot(result_df, pcs=(pc_x, pc_y), nstd=nstd_val, arrow_scale=arrow_scale)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Data view")
        st.dataframe(result_df[['DateTime', 'tag'] + list(result_df.columns[:3])].head(100))
        st.info("📌 **如何分析：**\n1. 观察不同标签（如 normal vs err）的椭圆是否有重叠。\n2. 红色箭头代表特征，箭头越长说明对该 PC 轴贡献越大。\n3. 远离正常簇的群组（err1, err2）说明其生产参数偏离了基准状态。")

else:
    st.error(msg)
    st.info("💡 请检查本地 PostgreSQL 是否已启动，且连接信息正确。")
