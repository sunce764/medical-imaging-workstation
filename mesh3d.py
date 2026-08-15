# =============================================================================
# 器官三维表面重建与渲染 纯计算模块
# 负责：从分割蒙版提取等值面网格（marching cubes）、计算形状特征、
#       用正交投影 + 法向着色把网格渲染为 2D 图像、导出 STL。
# 设计：无任何 Qt/UI/OpenGL 依赖，输入输出皆为 numpy 数组与普通元组，
#       故可脱离 MedicalViewer 独立单元测试（见 tests/test_gui.py::test_mesh3d）。
#
# 为何自己用 numpy 渲染而不上 OpenGL/VTK：
#   本项目 CPU-only、且不引入重依赖（VTK 逾百 MB）。正交投影 + Lambert 着色
#   用纯 numpy 即可，几十毫秒量级，还能脱离显示环境单测——渲染正确性可断言，
#   而 OpenGL 路径在离屏测试里几乎无法验证。代价是没有透视/阴影/交互旋转，
#   只做静态多角度预览，对教学与作品集展示足够。
#
# marching cubes 的副产物：表面积与球形度。这两个是器官形状特征，
# 单靠体素计数得不到（体素法的"表面积"随分辨率剧烈变化，而网格面积稳定得多）。
# =============================================================================

from __future__ import annotations

import numpy as np
from skimage import measure


def extract_surface(mask: np.ndarray, label: int, spacing: tuple[float, float, float],
                    step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """从标签图提取指定器官的三角网格。返回 (verts, faces)，顶点单位为 mm。

    mask:    3D 标签图 (Z, H, W)
    label:   要提取的器官标签号
    spacing: (行间距, 列间距, 层厚) mm；marching_cubes 的 spacing 顺序须与轴序一致，
             即 (z, y, x) = (层厚, 行间距, 列间距)
    step:    降采样步长。step=1 精度最高但顶点数与耗时约为 step=2 的 4 倍；
             实测 233×512×512 的器官：step=1 约 1.4 s、step=2 约 0.11 s，
             故默认 2——交互预览用，不是几何精算。

    该标签不存在或体素过少无法成面时返回两个空数组，由调用方判断。
    """
    ps0, ps1, st = spacing
    binary = (mask == label)
    if not binary.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            binary.astype(np.uint8), level=0.5, step_size=max(1, int(step)),
            spacing=(st, ps0, ps1))          # 轴序 (z, y, x)
    except (RuntimeError, ValueError):
        # 体素太少 / 全部贴边导致无法构面——不是错误，返回空网格即可
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    return verts.astype(np.float32), faces.astype(np.int32)


def mesh_shape_stats(verts: np.ndarray, faces: np.ndarray) -> dict:
    """由三角网格算形状特征。顶点单位 mm ⇒ 面积 mm²、体积 mm³。

    surface_area_mm2  三角面积之和
    volume_mm3        散度定理下的封闭网格体积，取绝对值以免朝向翻转导致负值
    sphericity        球形度 = π^(1/3)·(6V)^(2/3) / A，完美球=1，越不规则越小。
                      这是形状的无量纲描述，与器官大小无关，故可跨器官/跨病例比较。
    网格为空时各项返回 0.0（不返回 nan：调用方多半直接格式化显示）。

    【精度实测，勿当作精确值】以解析球体（R=20 体素、spacing=1mm）验算：
      体积   误差 0.00%（step=1）/ 0.37%（step=2）—— 可靠
      表面积 高估 约 9%，球形度因此约 0.91 而非 1.0
    表面积的高估是 marching cubes 在体素化曲面上的**系统性偏差**（阶梯效应），
    不是实现缺陷；它随分辨率提高而减小，但不会消失。故表面积与球形度适合做
    **同条件下的相对比较**（同一病例的不同器官、同一器官的随访变化），
    不宜当作绝对几何量引用。体积则可直接用。
    """
    if len(faces) == 0 or len(verts) == 0:
        return {'surface_area_mm2': 0.0, 'volume_mm3': 0.0, 'sphericity': 0.0,
                'n_vertices': 0, 'n_faces': 0}
    tri = verts[faces]                         # (F, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    # 散度定理：V = |Σ (a · (b × c)) / 6|
    vol = float(abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0))
    sph = 0.0
    if area > 0 and vol > 0:
        sph = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
    return {'surface_area_mm2': area, 'volume_mm3': vol, 'sphericity': sph,
            'n_vertices': int(len(verts)), 'n_faces': int(len(faces))}


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """绕 z 轴转 azimuth、再绕 x 轴转 elevation 的 3×3 旋转矩阵。"""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], np.float32)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], np.float32)
    return rx @ rz


def render_mesh(verts: np.ndarray, faces: np.ndarray, size: int = 400,
                azimuth: float = 30.0, elevation: float = 20.0,
                rgb: tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """把网格渲染为 (size, size, 4) 的 RGBA 图像。正交投影 + Lambert 着色 + 深度排序。

    以三角面为绘制单位，按面心深度从远到近排序后逐面填充（画家算法）。
    不做透视、阴影与抗锯齿；面数上万时耗时在几十毫秒量级。
    网格为空时返回全透明图像。
    """
    img = np.zeros((size, size, 4), np.uint8)
    if len(faces) == 0 or len(verts) == 0:
        return img
    # macOS 的 Accelerate BLAS 在 matmul 后会误置浮点异常标志：已最小复现——
    # 纯随机 float32 数组乘单位矩阵同样报 divide-by-zero/overflow，而结果完全正确
    # （无 nan/inf）。这是后端的标志误报，不是数值问题，故在本渲染函数内局部抑制，
    # 避免每帧向终端刷警告；不使用全局 seterr，以免掩盖别处的真实数值异常。
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return _render_mesh_impl(img, verts, faces, size, azimuth, elevation, rgb)


def _render_mesh_impl(img, verts, faces, size, azimuth, elevation, rgb):
    """render_mesh 的实现体（已在调用方包了 errstate）。"""
    p = verts @ _rotation(azimuth, elevation).T          # 旋转到视角坐标系
    xy, depth = p[:, [2, 1]], p[:, 0]                    # x 轴朝向观察者作深度
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = float(max(hi - lo)) or 1.0
    margin = size * 0.08
    scale = (size - 2 * margin) / span
    px = ((xy - lo) * scale + margin).astype(np.int32)
    px[:, 1] = size - 1 - px[:, 1]                       # 图像 y 轴向下
    px = np.clip(px, 0, size - 1)
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    # 用阈值而非 ==0 判退化面：marching cubes 会产出面积极小的三角形，其法向长度
    # 可低至 1e-30 量级，直接相除会溢出为 inf 并污染后续着色（实测触发 overflow 警告）。
    nrm = nrm / np.maximum(ln, 1e-12)
    light = _rotation(azimuth, elevation).T @ np.array([1.0, 0.35, 0.6], np.float32)
    light /= np.linalg.norm(light)
    # Lambert 漫反射 + 环境光；取绝对值使背面同样受光，避免朝向不一致的网格出现黑洞
    shade = 0.30 + 0.70 * np.abs(nrm @ light)
    order = np.argsort(depth[faces].mean(axis=1))        # 远→近
    base = np.array(rgb, np.float32)
    for fi in order:
        tri_px = px[faces[fi]]
        col = np.clip(base * shade[fi], 0, 255).astype(np.uint8)
        _fill_triangle(img, tri_px, col)
    return img


def _fill_triangle(img: np.ndarray, tri: np.ndarray, color: np.ndarray) -> None:
    """在 RGBA 图像上用重心坐标填充一个三角形（就地修改）。"""
    x0, x1 = int(tri[:, 0].min()), int(tri[:, 0].max())
    y0, y1 = int(tri[:, 1].min()), int(tri[:, 1].max())
    if x1 < x0 or y1 < y0:
        return
    ax, ay = float(tri[0, 0]), float(tri[0, 1])
    bx, by = float(tri[1, 0]), float(tri[1, 1])
    cx, cy = float(tri[2, 0]), float(tri[2, 1])
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-9:                                   # 退化三角形（共线）跳过
        return
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
    w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
    inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
    if not inside.any():
        return
    sub = img[y0:y1 + 1, x0:x1 + 1]
    sub[inside, 0], sub[inside, 1], sub[inside, 2] = color[0], color[1], color[2]
    sub[inside, 3] = 255


def to_stl_bytes(verts: np.ndarray, faces: np.ndarray, name: str = "organ") -> bytes:
    """把网格序列化为 ASCII STL（可直接写文件，供 3D 打印或外部软件打开）。

    选 ASCII 而非二进制 STL：体积大一些，但纯文本可读、便于核对，
    且不涉及字节序问题。顶点单位 mm，与多数 3D 打印切片软件的默认单位一致。
    """
    if len(faces) == 0:
        return f"solid {name}\nendsolid {name}\n".encode()
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    # 用阈值而非 ==0 判退化面：marching cubes 会产出面积极小的三角形，其法向长度
    # 可低至 1e-30 量级，直接相除会溢出为 inf 并污染后续着色（实测触发 overflow 警告）。
    nrm = nrm / np.maximum(ln, 1e-12)
    out = [f"solid {name}"]
    for i in range(len(faces)):
        n = nrm[i]
        out.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        out.append("    outer loop")
        for v in tri[i]:
            out.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        out.append("    endloop")
        out.append("  endfacet")
    out.append(f"endsolid {name}")
    return ("\n".join(out) + "\n").encode()
