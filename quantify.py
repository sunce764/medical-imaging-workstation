# =============================================================================
# 器官定量纯计算模块
# 负责：从分割蒙版 + HU 体积 + 体素尺寸算各器官的体积(mL)与平均 HU。
# 设计：无任何 Qt/UI 依赖，输入输出皆为普通 numpy 数组与 dict/list，
#       故可脱离 MedicalViewer 独立单元测试（见 tests/test_gui.py::test_quantify）。
#       AnnotationMixin._compute_organ_stats 只是读取 self 状态后调用本函数的薄包装。
# =============================================================================

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage


def compute_organ_stats(volume_hu: np.ndarray, volume_mask: np.ndarray,
                        spacing: tuple[float, float, float],
                        organ_names: dict[int, tuple[str, str]]) -> list[dict]:
    """统计 volume_mask 中各标签的体积(mL)与平均 HU，按体积降序返回。

    volume_hu:    3D HU 值体素数组，shape=(Z,H,W)
    volume_mask:  同形状的 uint8 标签图（0=背景，1-255=器官/手动层）
    spacing:      (行间距 ps0, 列间距 ps1, 层厚 st)，单位 mm
    organ_names:  {标签号: (中文名, 英文名)}；缺失标签回退为 "类{id}"/"cls{id}"
    返回:         [{'id','name_zh','name_en','voxels','volume_ml','mean_hu'}, ...]，
                  按 volume_ml 降序；无前景标签时返回 []。
    """
    ps0, ps1, st = spacing
    vox_ml = ps0 * ps1 * st / 1000.0  # 单体素体积，mm³ → mL
    counts = np.bincount(volume_mask.ravel(), minlength=256)
    present = [i for i in range(1, 256) if counts[i] > 0]
    if not present:
        return []
    # ndimage.mean 一次性算出所有标签区域的平均 HU，避免逐类布尔索引
    means = np.atleast_1d(ndimage.mean(volume_hu, labels=volume_mask, index=present))
    rows = []
    for i, lid in enumerate(present):
        zh, en = organ_names.get(lid, (f"类{lid}", f"cls{lid}"))
        rows.append({'id': lid, 'name_zh': zh, 'name_en': en, 'voxels': int(counts[lid]),
                     'volume_ml': counts[lid] * vox_ml, 'mean_hu': float(means[i])})
    rows.sort(key=lambda r: -r['volume_ml'])
    return rows
