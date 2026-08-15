# =============================================================================
# 厚层投影（slab projection）纯计算模块
# 负责：在指定平面上取一定厚度的层块，沿该平面法向做最大/最小/平均投影。
# 设计：无任何 Qt/UI 依赖，输入输出皆为 numpy 数组，
#       故可脱离 MedicalViewer 独立单元测试（见 tests/test_gui.py::test_projection）。
#       _render_clinical_plane 只是按当前投影设置调用本模块取切片的薄包装。
#
# 为何做厚层而非整卷投影：临床上用的是 slab MIP（典型 5–20mm），而不是把整个体积
# 压成一张图——整卷投影会把无关解剖结构一并叠上来，反而掩盖目标。厚度可调是关键。
#
# 三种模式的用途（CT 常规）：
#   max  最大密度投影 MIP  —— 高密度结构：肺结节、血管、骨。本项目的肺癌数据即典型场景。
#   min  最小密度投影 MinIP —— 低密度结构：气道、肺气肿区。
#   mean 平均密度投影 AIP  —— 降噪、看整体密度分布。
# =============================================================================

from __future__ import annotations

import numpy as np

from constants import AXIAL, CORONAL, SAGITTAL

MODES = ('max', 'min', 'mean')


def slab_bounds(index: int, thickness: int, axis_len: int) -> tuple[int, int]:
    """以 index 为中心、厚度 thickness 层的层块范围 [lo, hi)，并夹在体积边界内。

    厚度为偶数时中心略偏前（lo 多取一层），与多数 PACS 的行为一致。
    thickness<=1 或超出边界时自动退化为合法范围，不抛异常——调用方在渲染热路径上。
    """
    if axis_len <= 0:
        return 0, 0
    t = max(1, int(thickness))
    half = t // 2
    lo = max(0, index - half)
    hi = min(axis_len, lo + t)
    lo = max(0, hi - t)          # 靠近上边界时回推，保证厚度尽量足额
    return lo, hi


def project(volume: np.ndarray, plane: int, index: int,
            thickness: int = 1, mode: str = 'max') -> np.ndarray:
    """在 plane 平面上以 index 为中心取 thickness 层，沿法向投影为 2D 数组。

    volume:    3D 数组 (Z, H, W)
    plane:     constants.AXIAL / CORONAL / SAGITTAL
    index:     该平面的当前层号（Axial→z, Coronal→y, Sagittal→x）
    thickness: 层块厚度（层数）。=1 时等价于取单层，返回值与直接切片完全一致。
    mode:      'max' / 'min' / 'mean'

    返回的 2D 形状与该平面的单层切片一致，故可直接替换原切片进入既有渲染路径。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，得到 {mode!r}")
    if volume.ndim != 3:
        raise ValueError(f"volume 必须是 3D，得到 {volume.ndim}D")
    Z, H, W = volume.shape
    axis_len = {AXIAL: Z, CORONAL: H, SAGITTAL: W}[plane]
    idx = int(np.clip(index, 0, max(0, axis_len - 1)))
    lo, hi = slab_bounds(idx, thickness, axis_len)
    if plane == AXIAL:
        slab, ax = volume[lo:hi, :, :], 0
    elif plane == CORONAL:
        slab, ax = volume[:, lo:hi, :], 1
    else:                                    # SAGITTAL
        slab, ax = volume[:, :, lo:hi], 2
    if slab.shape[ax] == 1:                  # 单层：直接压掉该轴，避免多余的 reduce 开销
        return np.squeeze(slab, axis=ax)
    if mode == 'max':
        return slab.max(axis=ax)
    if mode == 'min':
        return slab.min(axis=ax)
    return slab.mean(axis=ax, dtype=np.float32)


def thickness_mm(thickness: int, plane: int, px_sp: float, slice_thick: float) -> float:
    """把层数厚度换算为毫米，供界面标注真实厚度。

    投影沿平面法向进行：Axial 的法向是 z（层厚），Coronal/Sagittal 的法向落在
    平面内（行/列间距）——两者物理尺度不同，故必须按平面分别换算，不能一律用层厚。
    """
    per = slice_thick if plane == AXIAL else px_sp
    return max(1, int(thickness)) * float(per)
