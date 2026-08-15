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
    """统计 volume_mask 中各标签的体积(mL)与 HU 一阶统计量，按体积降序返回。

    volume_hu:    3D HU 值体素数组，shape=(Z,H,W)
    volume_mask:  同形状的 uint8 标签图（0=背景，1-255=器官/手动层）
    spacing:      (行间距 ps0, 列间距 ps1, 层厚 st)，单位 mm
    organ_names:  {标签号: (中文名, 英文名)}；缺失标签回退为 "类{id}"/"cls{id}"
    返回:         [{'id','name_zh','name_en','voxels','volume_ml',
                    'mean_hu','sd_hu','median_hu','p5_hu','p95_hu','min_hu','max_hu'}, ...]，
                  按 volume_ml 降序；无前景标签时返回 []。

    为何不止 mean：只报均值无法反映区域内的密度离散程度，而离散度正是判断分割是否
    误纳入邻近组织、以及做任何统计比较的前提（椭圆 ROI 一直给的是 mean±SD，
    器官定量此前只给 mean，两者口径不一致）。p5/p95 比 min/max 抗单体素噪声，
    故一并给出，min/max 仍保留供查看极端值。
    """
    ps0, ps1, st = spacing
    vox_ml = ps0 * ps1 * st / 1000.0  # 单体素体积，mm³ → mL
    counts = np.bincount(volume_mask.ravel(), minlength=256)
    present = [i for i in range(1, 256) if counts[i] > 0]
    if not present:
        return []
    # ndimage 一次性算出所有标签区域的统计量，避免逐类布尔索引。
    # errstate：scipy 的 _sum_centered 内部对不连续的 label 索引会建出空 bin 并做
    # counts=0 的除法，抛 invalid-value 警告——传进去的 present 全都有体素，是 scipy
    # 的实现细节而非本函数的问题（结果已由单测对手算值验证）。局部抑制，避免每次
    # 器官定量都刷警告；不用全局 seterr，以免掩盖别处的真实数值异常。
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.atleast_1d(ndimage.mean(volume_hu, labels=volume_mask, index=present))
        sds = np.atleast_1d(ndimage.standard_deviation(volume_hu, labels=volume_mask, index=present))
        mins = np.atleast_1d(ndimage.minimum(volume_hu, labels=volume_mask, index=present))
        maxs = np.atleast_1d(ndimage.maximum(volume_hu, labels=volume_mask, index=present))
    rows = []
    for i, lid in enumerate(present):
        zh, en = organ_names.get(lid, (f"类{lid}", f"cls{lid}"))
        # 百分位需按标签取值，ndimage 无对应聚合函数；仅对本标签体素取一次
        vals = volume_hu[volume_mask == lid]
        p5, med, p95 = np.percentile(vals, (5, 50, 95))
        rows.append({'id': lid, 'name_zh': zh, 'name_en': en, 'voxels': int(counts[lid]),
                     'volume_ml': counts[lid] * vox_ml,
                     'mean_hu': float(means[i]), 'sd_hu': float(sds[i]),
                     'median_hu': float(med), 'p5_hu': float(p5), 'p95_hu': float(p95),
                     'min_hu': float(mins[i]), 'max_hu': float(maxs[i])})
    rows.sort(key=lambda r: -r['volume_ml'])
    return rows
