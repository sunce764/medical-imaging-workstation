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
                        organ_names: dict[int, tuple[str, str]],
                        volume_conf: np.ndarray | None = None) -> list[dict]:
    """统计 volume_mask 中各标签的体积(mL)与 HU 一阶统计量，按体积降序返回。

    volume_hu:    3D HU 值体素数组，shape=(Z,H,W)
    volume_mask:  同形状的 uint8 标签图（0=背景，1-255=器官/手动层）
    spacing:      (行间距 ps0, 列间距 ps1, 层厚 st)，单位 mm
    organ_names:  {标签号: (中文名, 英文名)}；缺失标签回退为 "类{id}"/"cls{id}"
    volume_conf:  可选，同形状 uint8 置信度（255=1.0），来自 softmax 最大类概率。
                  给了才输出 mean_conf/p5_conf，没给则该键缺席——数学降级路径没有
                  概率输出，此时宁可不报，也不填一个看起来像置信度的数。
    返回:         [{'id','name_zh','name_en','voxels','volume_ml',
                    'mean_hu','sd_hu','median_hu','p5_hu','p95_hu','min_hu','max_hu'}, ...]，
                  给了 volume_conf 时另含 'mean_conf','p5_conf'（0-1）。
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
    # 标签含 254/255 时必须先加宽类型：scipy 的 _select 内部做 np.zeros(labels.max() + 2)，
    # labels 为 uint8 时 255+2 溢出回绕成 1，于是分配出长度 1 的数组，再按标签值索引就
    # IndexError。本项目的 MANUAL_TRACK_LABEL 正是 255（3D 追踪工具的专属标签），
    # 追踪完立刻调本函数 → 必崩。实测：mask 只要出现 255，无论是否同时含其他标签都会崩。
    # 仅在真的出现高位标签时才拷贝，正常 AI 分割（标签 1–24）零额外开销。
    lbl = volume_mask.astype(np.int32) if counts[254:].any() else volume_mask
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.atleast_1d(ndimage.mean(volume_hu, labels=lbl, index=present))
        sds = np.atleast_1d(ndimage.standard_deviation(volume_hu, labels=lbl, index=present))
    rows = []
    for i, lid in enumerate(present):
        zh, en = organ_names.get(lid, (f"类{lid}", f"cls{lid}"))
        # 百分位需按标签取值，ndimage 无对应聚合函数；仅对本标签体素取一次。
        # min/max 也从这份 vals 直接取：原先另调 ndimage.minimum/maximum，
        # 每个都要再扫一遍完整体积（233×512² ≈ 6100 万体素），而所需数据此处已在手；
        # 顺带消掉「统计量走 lbl、百分位走 volume_mask」的双索引口径。
        sel = volume_mask == lid
        vals = volume_hu[sel]
        p5, med, p95 = np.percentile(vals, (5, 50, 95))
        row = {'id': lid, 'name_zh': zh, 'name_en': en, 'voxels': int(counts[lid]),
               'volume_ml': counts[lid] * vox_ml,
               'mean_hu': float(means[i]), 'sd_hu': float(sds[i]),
               'median_hu': float(med), 'p5_hu': float(p5), 'p95_hu': float(p95),
               'min_hu': float(vals.min()), 'max_hu': float(vals.max())}
        if volume_conf is not None and volume_conf.shape == volume_mask.shape:
            # conf==0 是哨兵，表示该体素没有模型置信度（手动 3D 追踪写入的、或被画笔
            # 改过的）。必须排除：它们的原值是模型对**改动前那个器官**的置信度，
            # 拿来当这个标签的置信度纯属张冠李戴。若整个标签都无模型体素
            # （手动追踪层就是这种），干脆不报——宁可没有，也不给一个假的。
            cv = volume_conf[sel]
            cv = cv[cv > 0].astype(np.float32) / 255.0
            if cv.size:
                # p5 一并给出：平均置信度会被大片确信的内部体素拉高，掩盖边界处的低置信，
                # 而分割出错恰恰多发生在边界——低分位比均值更能暴露问题
                row['mean_conf'] = float(cv.mean())
                row['p5_conf'] = float(np.percentile(cv, 5))
                # 模型判定过的体素占比：远小于 1 说明这个器官已被大量手工改动
                row['conf_cover'] = float(cv.size / max(1, int(counts[lid])))
        rows.append(row)
    rows.sort(key=lambda r: -r['volume_ml'])
    return rows
