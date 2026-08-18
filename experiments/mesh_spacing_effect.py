# =============================================================================
# spacing 重采样对三维形状特征的传导影响
# ---------------------------------------------------------------------------
# 动机：ai_engine 现按 nnU-Net 契约把体积重采样到 1.5mm 再推理，蒙版边界因此按该
#       网格量化。这个误差不会止步于分割——三维表面重建直接吃这份蒙版，形状特征
#       （表面积 / 体积 / 球形度）随之带上一项系统偏差。
#
#       该数字此前只写在 mesh3d.py 的注释里，没有可复现的产物。而项目的一条底线是
#       「每个对外声称的数字都能在 experiments/results 找到出处」，故补上本脚本。
#
# 方法：用解析球体作真值（表面积与体积有闭式解），对比两条链路——
#         A. 直接在原生网格上提取表面
#         B. 先降到 1.5mm、再最近邻升回原网格（正是 ai_engine 现在做的事），再提取
#       两者都走产品同一套 marching cubes + Taubin 平滑，故差异只来自那次往返。
#
# 用法：python experiments/mesh_spacing_effect.py
# 产出：results/mesh_spacing_effect.csv
# =============================================================================

import csv
import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import mesh3d  # noqa: E402

RESULTS = os.path.join(_HERE, "results")
NATIVE = 0.712891    # RIDER 的 in-plane spacing
TARGET = 1.5         # organs.onnx 的训练 spacing
R = 20.0             # 解析球半径 mm


def main():
    n = 80
    g = (np.arange(n) - n / 2) * NATIVE
    Z, Y, X = np.meshgrid(g, g, g, indexing='ij')
    truth = ((Z ** 2 + Y ** 2 + X ** 2) < R ** 2).astype(np.uint8)

    f = NATIVE / TARGET
    coarse = ndimage.zoom(truth, f, order=0, prefilter=False)
    back = ndimage.zoom(coarse, [o / s for o, s in zip(truth.shape, coarse.shape, strict=True)],
                        order=0, prefilter=False)
    back = back[tuple(slice(0, min(a, b)) for a, b in zip(back.shape, truth.shape, strict=True))]

    exact_a, exact_v = 4 * np.pi * R ** 2, 4 / 3 * np.pi * R ** 3
    rows = []
    for tag, m in (("native", truth), ("via_1.5mm_roundtrip", back)):
        verts, faces = mesh3d.extract_surface(m, 1, (NATIVE,) * 3, step=1)
        verts = mesh3d.smooth_taubin(verts, faces, iterations=10)
        s = mesh3d.mesh_shape_stats(verts, faces)
        rows.append(dict(path=tag,
                         area_mm2=round(s['surface_area_mm2'], 1),
                         area_err_pct=round(100 * (s['surface_area_mm2'] / exact_a - 1), 2),
                         volume_mm3=round(s['volume_mm3'], 1),
                         volume_err_pct=round(100 * (s['volume_mm3'] / exact_v - 1), 2),
                         sphericity=round(s['sphericity'], 4)))
        print(f"  {tag:<22} 表面积 {rows[-1]['area_mm2']:>8.1f} ({rows[-1]['area_err_pct']:+.2f}%)"
              f"  体积 {rows[-1]['volume_mm3']:>9.1f} ({rows[-1]['volume_err_pct']:+.2f}%)"
              f"  球形度 {rows[-1]['sphericity']:.4f}")
    with open(os.path.join(RESULTS, "mesh_spacing_effect.csv"), 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = rows[1]['area_err_pct'] - rows[0]['area_err_pct']
    print(f"\n  解析真值：表面积 {exact_a:.0f} mm²  体积 {exact_v:.0f} mm³  球形度 1.0")
    print(f"  重采样往返使表面积偏差多出 {d:+.2f} 个百分点")
    print("    → results/mesh_spacing_effect.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
