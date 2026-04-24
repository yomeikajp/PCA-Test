# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです。
"""

import streamlit as st
import matplotlib
matplotlib.use('Agg')  # ⭐ 防止后端问题

from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# ===== 椭圆函数 =====
def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):

    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)

    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)

    return ellip


# ===== 主绘图 =====
def show_ellipse(X_pca, y, pca, nstd=3, show_arrow=True,scale=1):

    colors = ['tab:blue', 'tab:orange', 'seagreen']
    category_label = ['Setosa', 'Versicolor', 'Virginica']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    xs = X_pca[:, 0]
    ys = X_pca[:, 1]

    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    xs = xs * scalex
    ys = ys * scaley

    data = np.concatenate((xs[:, None], ys[:, None]), 1)

    for i in range(max(y) + 1):
        ax.plot(
            data[:, 0][y == i],
            data[:, 1][y == i],
            '.',
            color=colors[i],
            label=category_label[i],
            markersize=8
        )

        plot_point_cov(data[y == i], nstd=nstd, ax=ax, alpha=0.25, color=colors[i])

    # PCA箭头
    if show_arrow:
        coeff = np.transpose(pca.components_[0:2, :])
        #scale = 0.7 # ⭐ 防止箭头太长
        for i in range(coeff.shape[0]):
            ax.arrow(
                0, 0,
                coeff[i, 0] * scale*0.5,
                coeff[i, 1] * scale*0.5,
                color='red',
                head_width=0.04,
                head_length=0.03
            )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xlabel(f'PC1 ({round(pca.explained_variance_ratio_[0] * 100, 2)} %)')
    ax.set_ylabel(f'PC2 ({round(pca.explained_variance_ratio_[1] * 100, 2)} %)')

    ax.legend()

    return fig

st.write("THIS IS MY FILE")
# ===== Streamlit UI =====
st.title("PCA Confidence Ellipse Dashboard")

# ⭐ 交互区
st.sidebar.header("参数控制")

nstd = st.sidebar.slider("椭圆置信倍数", 1, 5, 3)
scale=st.sidebar.slider("箭头长度",  1, 2,3)
show_arrow = st.sidebar.checkbox("显示PCA载荷箭头", True)

# ⭐ 数据准备
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 标准化
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# ⭐ 只取2维
pca = PCA(n_components=2)
x_new = pca.fit_transform(X)

# ⭐ 直接画图（不需要按钮）
fig = show_ellipse(x_new, y, pca, nstd=nstd, show_arrow=show_arrow,scale=scale)
st.pyplot(fig)
st.write("RUN OK")