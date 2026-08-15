# =============================================================================
# MPR 坐标几何纯计算模块
# 负责：MPR 三平面与 3D 光标 [z,y,x] 之间的坐标换算，以及双序列解剖 z 配准。
# 设计：无任何 Qt/UI 依赖，纯整数/数组运算。把原先散落在 sync_crosshair /
#       _render_clinical_plane / _render_compare 里的同一套坐标约定收拢为单一
#       可信来源（历史上轴向易错，见 BUG J），并可独立单测。
#
# 坐标约定（三平面共用同一个 3D 光标 [z, y, x]）：
#   - Axial    视图 (px,py) → 3D 的 (x=px, y=py)，z 不变
#   - Coronal  视图 (px,py) → 3D 的 (x=px, z=py)，y 不变
#   - Sagittal 视图 (px,py) → 3D 的 (y=px, z=py)，x 不变
# =============================================================================

from __future__ import annotations

import numpy as np

from constants import AXIAL, CORONAL, SAGITTAL


def hover_to_voxel(plane: int, px: int, py: int,
                   cur: tuple[int, int, int],
                   shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """把某平面上的悬停像素 (px,py) 映射为完整 3D 体素 (z,y,x)，非该平面的轴沿用当前
    光标 cur=(z,y,x)，并按 shape=(Z,Y,X) 裁剪到体积范围内。返回 (z,y,x)。"""
    z, y, x = cur
    Z, Y, X = shape
    if plane == AXIAL:
        x, y = px, py
    elif plane == CORONAL:
        x, z = px, py
    elif plane == SAGITTAL:
        y, z = px, py
    x = max(0, min(x, X - 1))
    y = max(0, min(y, Y - 1))
    z = max(0, min(z, Z - 1))
    return z, y, x


def voxel_to_crosshair(plane: int, z: int, y: int, x: int) -> tuple[int, int]:
    """把 3D 光标 (z,y,x) 投影为某平面上十字准线的 2D 坐标 (cx,cy)。"""
    if plane == CORONAL:
        return x, z
    if plane == SAGITTAL:
        return y, z
    return x, y  # AXIAL（含缺省）


def nearest_slice(zpos, target_z: float) -> int:
    """在解剖 z 坐标数组 zpos 中，返回与 target_z 最接近的切片索引（双序列配准）。

    zpos 为空时返回 0 而非让 np.argmin 抛 ValueError：本函数在渲染热路径上被调用，
    崩在这里会让整个对比视图挂掉；退回首层是安全且可见的降级。
    （正常路径下 compare_lab 的守卫已保证 zpos 非空，此处是纵深防御。）
    """
    a = np.asarray(zpos)
    if a.size == 0:
        return 0
    return int(np.argmin(np.abs(a - target_z)))
