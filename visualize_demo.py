"""3D点云可视化 - 三种常用方案演示"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ============================================================================
# 方案 1: Matplotlib (已有，简单但不支持交互旋转放大)
# ============================================================================
def viz_matplotlib(points, title="Point Cloud"):
    """用 matplotlib 的 3D 散点图展示点云 (静态图像)"""
    import matplotlib.pyplot as plt
    
    points = np.asarray(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = points[:, 2]  # 按 z 值着色
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 方案 2: Open3D (最推荐！效果最好，支持交互旋转放大)
# ============================================================================
def viz_open3d(points, title="Point Cloud"):
    """用 Open3D 展示点云 (专业点云库，支持旋转、放大、拖动)
    
    交互控制:
    - 鼠标左键拖动: 旋转
    - 鼠标滚轮: 放大/缩小
    - 鼠标中键: 平移
    - P: 截屏
    """
    import open3d as o3d
    
    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 着色: 按高度(z)着色
    z_values = points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
    colors = plt.cm.viridis(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name=title)


# ============================================================================
# 方案 3: Plotly (交互式，支持web，可保存为HTML)
# ============================================================================
def viz_plotly(points, title="Point Cloud", save_html=None):
    """用 Plotly 展示点云 (交互式web图表)
    
    Args:
        points: 形状 (N, 3) 的点坐标
        title: 图标题
        save_html: 如果提供文件名，会保存为 HTML 文件
    """
    import plotly.graph_objects as go
    
    points = np.asarray(points)
    z_values = points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=z_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Z value")
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        width=900,
        height=700
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"已保存为 {save_html}")
    
    fig.show()


# ============================================================================
# 方案 4: Vispy (快速、支持大规模点云)
# ============================================================================
def viz_vispy(points, title="Point Cloud"):
    """用 Vispy 展示点云 (GPU加速，支持百万级点云)
    
    快捷键:
    - 鼠标拖动: 旋转
    - 鼠标滚轮: 放大/缩小
    """
    from vispy import scene
    from vispy.color import get_colormap
    
    points = np.asarray(points)
    
    canvas = scene.SceneCanvas(title=title, keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    z_values = points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
    colors = get_colormap('viridis').map(z_norm)
    
    scatter = scene.visuals.Markers()
    scatter.set_data(points, face_color=colors, edge_width=0, size=2)
    view.add(scatter)
    
    view.camera = 'arcball'
    view.camera.scale_factor = max(points.max() - points.min()) * 1.2
    
    canvas.app.run()


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 方法A: 随机点云 (快速测试)
    print("生成随机点云...")
    points = np.random.randn(500, 3) * 0.5
    
    # 方法B: 从 ModelNet40 读取 (真实数据)
    print("\n从 ModelNet40 读取点云...")
    try:
        with h5py.File('modelnet40_ply_hdf5_2048/ply_data_train0.h5', 'r') as f:
            points = f['data'][0]  # 第一个样本
            print(f"点云形状: {points.shape}")
    except FileNotFoundError:
        print("未找到 ModelNet40 数据，使用随机点云")
    
    # 选择展示方法
    print("\n========== 点云可视化方案 ==========")
    print("1. Matplotlib - 简单静态图 (推荐入门)")
    print("2. Open3D - 专业交互 (推荐！效果最好)")
    print("3. Plotly - Web交互式 (推荐分享/保存)")
    print("4. Vispy - 大规模高速 (百万级点云)")
    print("=====================================\n")
    
    choice = input("选择方案 (1-4，默认2): ").strip() or "2"
    
    if choice == "1":
        print("使用 Matplotlib...")
        viz_matplotlib(points, "Point Cloud - Matplotlib")
    elif choice == "2":
        print("使用 Open3D...")
        viz_open3d(points, "Point Cloud - Open3D")
    elif choice == "3":
        print("使用 Plotly...")
        viz_plotly(points, "Point Cloud - Plotly", save_html="point_cloud.html")
    elif choice == "4":
        print("使用 Vispy...")
        viz_vispy(points, "Point Cloud - Vispy")
    else:
        print("无效选择，使用 Open3D")
        viz_open3d(points, "Point Cloud - Open3D")
