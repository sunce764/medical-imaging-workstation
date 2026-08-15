# =============================================================================
# 二维刚性配准 纯计算模块
# 负责：估计两张同尺寸切片之间的平面内平移与旋转，并施加校正。
# 设计：无任何 Qt/UI 依赖，输入输出皆为 numpy 数组与普通元组，
#       故可脱离 MedicalViewer 独立单元测试（见 tests/test_gui.py::test_registration）。
#
# 【符号约定，务必看清——本文件的作者在实测中两次栽在方向上】
#   estimate_translation(ref, moving) 返回的是「moving 相对 ref 已经发生的位移」，
#   即若 moving = shift(ref, t) 则返回 t。
#   要把 moving 对齐回 ref，必须施加 **负** 的位移——这一步封装在 apply_rigid 里，
#   调用方不应自己拿返回值去 ndi.shift，否则会朝同方向再移一次，把图对得更歪。
#   实测教训：符号写反时位移数值完全正确、图像看着"也动了"，但 MAE 从 262 涨到 355，
#   属于典型的静默失真——故本模块所有对外函数都同时返回配准前后的 NCC 供核对。
#
# 为何是刚性而非形变配准：随访 CT 的主要差异是床位与体位造成的平移、以及轻微旋转；
# 形变配准（B-spline/Demons）能吸收呼吸导致的器官形变，但也会把**真实的病灶变化**
# 一并"配没了"——对以观察变化为目的的随访比较是有害的。刚性配准只校正体位，
# 不改变解剖内部的相对关系，这是随访定量的常规选择。
# =============================================================================

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """零均值归一化互相关（NCC）。1=完全一致，0=无关。任一方无变化时返回 0（无定义）。"""
    x = a.astype(np.float64).ravel(); y = b.astype(np.float64).ravel()
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den < 1e-12:
        return 0.0
    return float((x * y).sum() / den)


def estimate_translation(ref: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    """用相位相关估计 moving 相对 ref 的整像素平移 (dy, dx)。见文件头的符号约定。

    相位相关只保留互功率谱的相位、丢弃幅度，故对两次扫描间的亮度/对比度差异不敏感，
    也对噪声稳健（实测：真实 CT 切片加 σ=100 HU 高斯噪声后估计值仍精确）。
    结果为整像素；亚像素精度对本用途（定性对比）没有必要。
    """
    a = np.nan_to_num(moving.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(ref.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    cross = A * np.conj(B)
    cross /= np.maximum(np.abs(cross), 1e-12)      # 只留相位
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    # 峰值超过半幅时代表负方向位移（FFT 的周期性折叠）
    return tuple(int(p if p < s // 2 else p - s) for p, s in zip(peak, corr.shape, strict=True))


def apply_rigid(img: np.ndarray, angle_deg: float, shift_yx: tuple[int, int]) -> np.ndarray:
    """把 estimate_* 得到的 (角度, 位移) 作为**校正量**施加到 img 上，返回对齐后的图。

    内部对两者都取负——因为估计出的是"已经发生的变换"，校正要反着做。
    先反旋转再反平移，与 estimate_rigid 的搜索顺序严格一致；顺序颠倒会因旋转
    绕图像中心而引入额外位移。
    """
    out = img
    if angle_deg:
        out = ndimage.rotate(out, -angle_deg, reshape=False, order=1, mode='nearest')
    if shift_yx != (0, 0):
        out = ndimage.shift(out, [-s for s in shift_yx], order=1, mode='nearest')
    return out.astype(np.float32)


def register_rigid(ref: np.ndarray, moving: np.ndarray, max_angle: float = 6.0,
                   angle_step: float = 0.5) -> dict:
    """估计 moving → ref 的平面内刚性变换。返回含变换量与配准质量的 dict。

    max_angle=0 时只搜平移（约 10 ms）；默认搜 ±6°、步长 0.5°（约 0.5 s，实测
    512² 切片）。±6° 覆盖了随访扫描常见的体位差异，再大通常意味着摆位本身有问题。

    返回：
      angle_deg / shift_yx  估计出的变换量（含义见文件头约定）
      ncc_before/ncc_after  配准前后的 NCC
      improved              配准后 NCC 是否确有提升
      applied               是否建议采用（improved 为假时为 False，见下）

    **安全阀**：若配准后 NCC 不升反降（解剖变化过大、层面不对应等），
    applied 置 False 并把变换量归零——宁可不配准，也不能把图对得更歪。
    这条不是理论顾虑：实测中一个符号错误就会造成"数值看着对、图却更歪"的静默失真。
    """
    if ref.shape != moving.shape:
        raise ValueError(f"配准要求同尺寸，得到 {ref.shape} vs {moving.shape}")
    ncc0 = normalized_cross_correlation(ref, moving)
    best_angle, best_shift, best_ncc = 0.0, (0, 0), -2.0
    if max_angle > 0 and angle_step > 0:
        angles = np.arange(-max_angle, max_angle + angle_step / 2, angle_step)
    else:
        angles = np.array([0.0])
    for ang in angles:
        rot = (ndimage.rotate(moving, -ang, reshape=False, order=1, mode='nearest')
               if ang else moving)
        sh = estimate_translation(ref, rot)
        aligned = ndimage.shift(rot, [-s for s in sh], order=1, mode='nearest')
        score = normalized_cross_correlation(ref, aligned)
        if score > best_ncc:
            best_ncc, best_angle, best_shift = score, float(ang), sh
    improved = best_ncc > ncc0
    if not improved:
        return {'angle_deg': 0.0, 'shift_yx': (0, 0), 'ncc_before': ncc0,
                'ncc_after': ncc0, 'improved': False, 'applied': False}
    return {'angle_deg': best_angle, 'shift_yx': best_shift, 'ncc_before': ncc0,
            'ncc_after': best_ncc, 'improved': True, 'applied': True}
