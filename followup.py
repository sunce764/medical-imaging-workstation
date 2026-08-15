# =============================================================================
# 随访对比定量纯计算模块
# 负责：对两个已按解剖 z 坐标配准的切片，计算 HU 差值图与差异统计量。
# 设计：无任何 Qt/UI 依赖，输入输出皆为 numpy 数组与 dict，
#       故可脱离 MedicalViewer 独立单元测试（见 tests/test_gui.py::test_followup）。
#       CompareMixin 只是读取 self 状态后调用本模块的薄包装。
#
# 【重要前提，勿夸大本模块的能力】
#   本软件的双序列对比只做了 **z 轴层面配准**（按 ImagePositionPatient 找最近解剖层），
#   xy 平面内 **未做任何刚性或形变配准**。因此这里的差值反映的是
#   「同一解剖层面上的密度差异 + 未校正的体位/呼吸相位差异」之和，
#   只能作定性参考，不能当作临床意义上的病灶变化量。
#   真正的随访定量需要 rigid/deformable registration，本项目未实现。
# =============================================================================

from __future__ import annotations

import numpy as np


def can_compare(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """判断两个切片是否具备可比性。返回 (是否可比, 不可比的原因)。

    形状不同 → 逐像素相减没有意义（不同重建矩阵/视野），直接拒绝而非强行缩放：
    强行 resize 会引入插值误差并制造「看起来能比」的假象。
    """
    if a is None or b is None:
        return False, "缺少切片数据"
    if a.shape != b.shape:
        return False, f"矩阵尺寸不同（{a.shape} vs {b.shape}），逐像素比较无意义"
    if a.size == 0:
        return False, "切片为空"
    return True, ""


def _finite(a: np.ndarray) -> np.ndarray:
    """中和 NaN / ±Inf，返回有限的 float32 副本。

    畸形 DICOM（异常 RescaleSlope、损坏像素）可能产出非有限 HU；若不中和，
    差值统计会整片变成 nan/inf 并直接显示到界面上（"Δnan 绝对差 nan"），
    差值图转 uint8 时更是未定义行为。与 recon._finite_clip 同一防御思路。
    ±Inf 映射到 HU 的合理极值而非 0：置 0 会把"极端密度"伪装成"无差异"。
    """
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=3071.0, neginf=-1024.0)


def diff_map(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """当前切片 − 既往切片的 HU 差值图（float32，正值=密度升高）。非有限值先中和。"""
    return _finite(cur) - _finite(prev)


def compare_slices(cur: np.ndarray, prev: np.ndarray, hu_range: float = 2000.0) -> dict:
    """计算两个已配准切片的差异统计量。

    cur / prev: 同形状的 2D HU 数组（当前序列 / 既往序列）
    hu_range:   计算归一化相似度时的 HU 动态范围，默认 2000（约 -1000~1000 常用窗）

    返回 dict：
      mean_diff  有符号平均差（正=当前更致密）        sd_diff  差值标准差
      mae        平均绝对差                          rmse     均方根差
      p5 / p95   差值的第 5 / 95 百分位（抗单点噪声）  max_abs  最大绝对差
      corr       两切片的 Pearson 相关系数（结构一致性，1=完全线性相关）
      nrmse      rmse / hu_range，便于跨病例横向比较
    任一切片方差为 0（全同值）时 corr 无定义，返回 float('nan')。
    """
    ok, why = can_compare(cur, prev)
    if not ok:
        raise ValueError(why)
    d = diff_map(cur, prev)                       # 内部已中和非有限值
    flat_c = _finite(cur).astype(np.float64).ravel()
    flat_p = _finite(prev).astype(np.float64).ravel()
    # 任一方无变化时相关系数无定义（分母为 0），显式返回 nan 而非让 numpy 报警
    if flat_c.std() == 0 or flat_p.std() == 0:
        corr = float('nan')
    else:
        corr = float(np.corrcoef(flat_c, flat_p)[0, 1])
    rmse = float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))
    p5, p95 = np.percentile(d, (5, 95))
    return {
        'mean_diff': float(d.mean()), 'sd_diff': float(d.std()),
        'mae': float(np.abs(d).mean()), 'rmse': rmse,
        'p5': float(p5), 'p95': float(p95), 'max_abs': float(np.abs(d).max()),
        'corr': corr, 'nrmse': rmse / hu_range if hu_range else float('nan'),
    }


def diff_to_rgba(d: np.ndarray, clip_hu: float = 200.0) -> np.ndarray:
    """把差值图渲染为发散配色的 RGBA（H,W,4，uint8），供叠加显示。

    正差值（密度升高）→ 暖色；负差值（密度降低）→ 冷色；接近 0 → 全透明。
    clip_hu: 色标饱和阈值，|差值| ≥ clip_hu 即取满色。取 200 HU 是软组织变化的
             常见量级；固定阈值使不同切片之间的颜色可直接比较（自适应拉伸不可比）。
    """
    if clip_hu <= 0:
        raise ValueError("clip_hu 必须为正")
    # 先中和再 clip：np.clip 对 NaN 无效（NaN 会穿透），而 NaN → uint8 是未定义行为，
    # 实测会产生任意像素值，即把"无法计算"渲染成看似真实的颜色。
    t = np.clip(_finite(d) / clip_hu, -1.0, 1.0)
    out = np.zeros(d.shape + (4,), dtype=np.uint8)
    mag = np.abs(t)
    pos, neg = t > 0, t < 0
    # 暖色（升高）：偏红橙；冷色（降低）：偏蓝青。与器官 LUT 的色系区分开，避免混淆。
    out[..., 0] = np.where(pos, 255, np.where(neg, 40, 0))
    out[..., 1] = np.where(pos, (140 * (1 - mag)).astype(np.uint8), (170 * mag).astype(np.uint8))
    out[..., 2] = np.where(pos, (60 * (1 - mag)).astype(np.uint8), 255 * neg)
    out[..., 3] = (mag * 200).astype(np.uint8)   # 差异越大越不透明；0 差异全透明
    return out
