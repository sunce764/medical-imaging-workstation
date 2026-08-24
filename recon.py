# =============================================================================
# 重建算法纯计算模块
# 负责：所有 CT 重建相关的纯数值计算，不依赖任何 Qt/UI 代码
# 调用方：MedicalViewer 的方法作为薄包装层，负责读取 UI 状态并展示结果
# =============================================================================

from __future__ import annotations

import hashlib
import inspect
import multiprocessing as _mp
import os
import time
from collections.abc import Callable

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from skimage.transform import iradon, radon

# -------------------------------------------------------------------------
# 模块级缓存（顶部集中声明，便于阅读时一眼看清全局可变状态）
# -------------------------------------------------------------------------

# 系统矩阵磁盘缓存目录：放在本模块所在目录下的 .matrix_cache/，
# 避免与项目根混淆；首次访问时按需创建。
_MATRIX_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matrix_cache")

# 圆形掩码缓存：{n: mask}。同 n 在多次重建中反复使用，避免重复 ogrid + 比较运算。
# key 空间极小（≤ 4 种 n），无需淘汰策略。
_CIRCLE_MASK_CACHE = {}

# DFR Delaunay 三角剖分缓存：key=(num_detectors, len(theta), 角度表指纹)。
# scipy.griddata 内部每次都会重做 Delaunay 三角剖分（O(N log N) 但常数极大），
# 同参数复算时复用 Delaunay 对象可省去 60%-80% 的 DFR 总耗时；
# 数学上完全等价，三角剖分由几何点集唯一确定。
#
# 【内存：有意的取舍，不是疏忽】原注释说「key 空间小（≈16 entry），无需淘汰策略」，
# 那只数了条目数、没数单条体积。DFR 走的是全分辨率源图（512² 切片直接做 Radon，
# 未降采样），点数 = 探测器数 × 角度数，最贵一档 512×1440 = 737,280 点，单个
# Delaunay 对象 88.3MB。实测把 UI 可达的 12 种配置依次点一遍：缓存常驻由 48MB
# （3 种）涨到 443MB（12 种），进程 maxRSS 375MB → 1603MB。
# 仍不加淘汰：最贵那一档首次 3.03s、命中 0.05s，相差 60 倍，而重建实验室是交互式
# 使用，淘汰会把「换个角度再看看」变成每次等三秒。443MB 相对本机 AI 推理 ~8.8GB 的
# 峰值是可接受的代价。若日后目标机器内存吃紧，这里是第一个该动的地方。
_DFR_TRI_CACHE = {}


# -------------------------------------------------------------------------
# 系统矩阵并行 worker（必须是模块顶层函数，multiprocessing 才能 pickle）
# -------------------------------------------------------------------------

def _matrix_worker(args: tuple) -> tuple[int, int, np.ndarray]:
    """计算系统矩阵 A 中 [start_j, end_j) 列对应像素的 Radon 贡献。
    子进程独立导入 skimage，避免父进程 Qt 状态被复制到子进程。
    """
    start_j, end_j, n, theta = args
    import numpy as _np
    from skimage.transform import radon as _radon

    # 先跑一次空图确定探测器数量
    n_rays = _radon(_np.zeros((n, n), dtype=_np.float32), theta=theta, circle=True).size
    cols = _np.zeros((n_rays, end_j - start_j), dtype=_np.float32)
    img = _np.zeros((n, n), dtype=_np.float32)
    for k, j in enumerate(range(start_j, end_j)):
        r, c = j // n, j % n
        img[r, c] = 1.0
        cols[:, k] = _radon(img, theta=theta, circle=True).ravel()
        img[r, c] = 0.0
    return start_j, end_j, cols


# _matrix_worker 源码的 SHA1 前 8 位——嵌入缓存文件名中，
# worker 代码一改哈希就变，旧缓存自动失效，永远不需要手动清理 .matrix_cache/。
_WORKER_HASH = hashlib.sha1(
    inspect.getsource(_matrix_worker).encode('utf-8')
).hexdigest()[:8]


def _purge_stale_matrix_cache() -> None:
    """启动时清理 .matrix_cache/ 中哈希与当前 _WORKER_HASH 不匹配的过期文件。

    文件名规范：A_n{n}_na{na}_t{ts}_{te}_{hash8}.npy，其中 hash8 必须是 8 位十六进制；
    任何不严格匹配此模式的文件一律跳过，杜绝误删风险。
    """
    if not os.path.isdir(_MATRIX_CACHE_DIR):
        return
    hex_chars = set("0123456789abcdef")
    for fn in os.listdir(_MATRIX_CACHE_DIR):
        if not fn.startswith("A_n") or not fn.endswith(".npy"):
            continue
        stem = fn[:-4]  # 去掉 .npy
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        hash_part = parts[1]
        # 严格校验：hash 段必须正好 8 位且全部是小写十六进制字符
        if len(hash_part) != 8 or not all(c in hex_chars for c in hash_part):
            continue
        if hash_part == _WORKER_HASH:
            continue
        try:
            os.remove(os.path.join(_MATRIX_CACHE_DIR, fn))
        except OSError as e:
            print(f"Warning: failed to remove stale cache {fn}: {e}")


_purge_stale_matrix_cache()


# -------------------------------------------------------------------------
# 正向投影
# -------------------------------------------------------------------------

def compute_sinogram(img_norm: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """对归一化图像执行 Radon 变换，返回弦图。

    img_norm: 归一化到 [0,1] 的 2D 浮点数组
    theta:    投影角度数组（度），如 np.linspace(0, 180, 180, endpoint=False)
    返回:     sinogram，shape=(探测器数, 角度数)
    """
    # 非有限值防护：与 compute_fbp / compute_dfr 保持一致（此前只有那两个做了，是个缺口）。
    # 一个 NaN 像素经 Radon 的线积分会污染所有穿过它的射线——实测局部 4×4 的 NaN
    # 就让整幅弦图 100% 非有限，而下游 BP/FBP/DFR 照常「跑通」出图、界面无任何提示。
    # 源头已在 MedicalViewer._dcm_float 兜住（NaN 的 RescaleSlope/PixelSpacing 退回默认值），
    # 此处为纵深防御，也保护直接调用本模块的实验脚本。
    img_norm = np.nan_to_num(img_norm, nan=0.0, posinf=1.0, neginf=0.0)
    # circle=True：假定图像圆外为零（不是替你置零），与 iradon 的输出支撑一致。
    # 输入已由 prepare_small_image/shepp_logan 掩过圆，故满足该前提。
    return radon(img_norm, theta=theta, circle=True)


def make_theta(angle_range: float, n_proj: int | None = None) -> np.ndarray:
    """在 [0, angle_range) 度范围内均匀生成 n_proj 个投影角度。

    angle_range: 角度覆盖范围（度）。180 为 CT 完整半圈（对径投影冗余，覆盖完整）。
    n_proj:      投影数量。None 时取 angle_range（每度一个投影，向后兼容旧行为）。
                 n_proj > angle_range 即"过采样"——角度间隔更细（如 180°取360个=0.5°间隔），
                 弦图信息更充分，反投影/迭代重建质量更高（代价是耗时随投影数线性增加）。
    返回:        np.ndarray，长度 = n_proj，范围 [0, angle_range)
    """
    if n_proj is None:
        n_proj = angle_range
    return np.linspace(0., float(angle_range), int(n_proj), endpoint=False)


# -------------------------------------------------------------------------
# BP / FBP
# -------------------------------------------------------------------------

def compute_bp(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """纯反投影（不滤波），返回重建图像。

    缺陷：低频分量被过度叠加，边缘极度模糊（星形伪影）。
    对比目的：展示滤波（FBP）对图像质量的改善效果。
    """
    # compute_fbp 的注释写着「此前只有矩阵/迭代法设了这道防线，解析法没有——防御
    # 不一致，此处补齐」，但补齐时漏了同一节里的 compute_bp。实测 sino 里一个 NaN：
    # 本函数输出非有限占比 0.7830，而 compute_fbp 是 0.0000。产品路径上传进来的是
    # 有防护的 compute_sinogram 输出，故不可达；但 recon.py 是对外的纯计算模块，
    # experiments/ 与直接调用者都能命中。
    # 不转 dtype：np.asarray(..., dtype=float) 会把 float32 提升成 float64，
    # 使本函数与 compute_fbp 内部未滤波路径的结果不再逐位相等——回归套件当场抓到。
    sinogram = np.nan_to_num(sinogram, nan=0.0, posinf=0.0, neginf=0.0)
    # filter_name=None 表示纯反投影，不做任何频域滤波
    return iradon(sinogram, theta=theta, filter_name=None, circle=True)


def compute_fbp(sinogram: np.ndarray, theta: np.ndarray, filter_name: str) -> tuple[np.ndarray, np.ndarray]:
    """滤波反投影（FBP），返回 (recon_bp_unfiltered, recon_fbp)。

    同时返回未滤波结果供对比展示，避免调用方重复计算。
    filter_name: 'ramp' / 'shepp-logan' / 'cosine' / 'hamming' / 'hann'
                 注意：UI 显示 'Ram-Lak'，调用前需映射为 'ramp'
    """
    # skimage.transform.iradon 内部滤波器名为 'ramp'，调用前需做名称映射
    if filter_name.lower() == "ram-lak":
        filter_name = "ramp"
    # 与 DMR/ART/SIRT 的 _finite_clip 对齐：弦图若混入 NaN/±Inf（损坏输入、上游异常），
    # iradon 会把它扩散到整幅重建图，而 NaN 在后续 clip/显示中会静默变黑并污染 RMSE。
    # 此前只有矩阵/迭代法设了这道防线，解析法没有——防御不一致，此处补齐。
    s = np.nan_to_num(sinogram, nan=0.0, posinf=0.0, neginf=0.0)
    recon_bp = iradon(s, theta=theta, filter_name=None, circle=True)
    recon_fbp = iradon(s, theta=theta, filter_name=filter_name, circle=True)
    return recon_bp, recon_fbp


# -------------------------------------------------------------------------
# DFR（直接傅里叶重建）
# -------------------------------------------------------------------------

def _theta_fingerprint(theta) -> str:
    """把整张角度表压成 8 位十六进制指纹，用于缓存键。

    【为什么不能只用 (长度, 首, 尾)】那三项相同的角度表可以完全不同：
    [0,60,120,150,170] 与 [0,5,10,15,170] 长度、首、尾全同。实测撞键的后果——
    DFR 三角剖分缓存复用错剖分，整幅重建图相对峰值差 196%；system matrix 直接原样
    返回上一张 A，两者逐元素最大差 1.06，而且它还【落盘】，跨进程跨会话持久生效。
    当前所有调用方都经 make_theta 产出均匀 linspace，对均匀网格 (首,尾,点数) 是单射，
    所以此前没出事；但任何一次「换个角度表」的实验设计变更都会静默复用错矩阵，
    而错的是结果本身，不会报错。
    """
    a = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:8]


def compute_dfr(sinogram: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """直接傅里叶重建法（DFR），基于傅里叶中心切片定理。

    返回: (freq_domain_2d, fft_1d_display, recon_dfr)
      freq_domain_2d:   插值后的二维复数频域矩阵，供"二维频域分布"视图展示
      fft_1d_display:   log1p 压缩后的一维频谱幅度图，供"一维FFT谱"视图展示
      recon_dfr:        2D 逆 FFT 重建结果（复数，取 abs 后显示）。朝向已在本函数内
                        校正为与输入同方位（含偶数 n 的 1 像素修正），调用方直接
                        np.abs(recon_dfr) 即可，无需再自行 rot90。见 test_recon_numerics
                        的 DFR 断言（实测：治本后与真值相关 n=64→0.906、偏心脉冲峰值零偏移）。

    算法步骤：
      1. 对弦图每列（沿探测器方向）做 1D FFT → 极坐标频域样本
      2. 将极坐标 (r, θ) 样本插值到直角坐标网格（griddata，method='linear'）
      3. 对插值后的 2D 频域做 2D 逆 FFT → 重建图像
    """
    num_detectors, num_angles = sinogram.shape

    # 步骤1：对弦图沿探测器方向（axis=0）做 1D FFT
    # ifftshift 将数据中心移到 FFT 起点（左端），fft 计算，再 fftshift 将零频移回中心
    # 这样 proj_fft[num_detectors//2, :] 对应零频（直流分量）
    # 入口先中和非有限值：FFT 遇 NaN/±Inf 会把污染扩散到全部频率分量，
    # 后面第 227 行的 nan_to_num 已来不及救（那时整幅频域都成了 NaN）。
    sino = np.nan_to_num(sinogram, nan=0.0, posinf=0.0, neginf=0.0)
    proj_fft = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(sino, axes=0), axis=0),
        axes=0
    )

    # 提取供展示的 1D 频谱（对数压缩后的幅度谱，log1p 避免 log(0) 的问题）
    fft_1d_display = np.log1p(np.abs(proj_fft))

    # 步骤2：构建极坐标网格（r 为频率半径，theta 为投影角度）
    r = np.arange(num_detectors) - num_detectors // 2
    r_grid, theta_grid = np.meshgrid(r, np.deg2rad(theta), indexing='ij')

    # 极坐标 → 直角坐标转换：(r, θ) → (r·cosθ, r·sinθ) = (kx, ky)
    x_polar = r_grid * np.cos(theta_grid)
    y_polar = r_grid * np.sin(theta_grid)
    points = np.column_stack((x_polar.flatten(), y_polar.flatten()))
    values = proj_fft.flatten()  # 对应每个极坐标点的复数频域值

    # 目标直角网格（与频域图像一一对应的均匀网格）
    grid_x, grid_y = np.meshgrid(r, r, indexing='ij')

    # 散点插值：将不均匀的极坐标样本插值到均匀的直角坐标网格
    # 性能优化：Delaunay 三角剖分对相同 (num_detectors, theta) 是确定的，
    # 缓存复用避免每次 griddata 重做剖分（DFR 主要瓶颈）。
    # 数学完全等价：LinearNDInterpolator(tri, values) 与
    # griddata(points, values, ..., method='linear') 用相同三角剖分 + 重心插值算法。
    tri_key = (num_detectors, len(theta), _theta_fingerprint(theta))
    tri = _DFR_TRI_CACHE.get(tri_key)
    if tri is None:
        tri = Delaunay(points)
        _DFR_TRI_CACHE[tri_key] = tri
    # fill_value=0：超出极坐标覆盖范围的格点填零（高频端无测量数据）
    interp = LinearNDInterpolator(tri, values, fill_value=0)
    freq_domain_2d = interp(grid_x, grid_y)

    # 步骤3：2D 逆 FFT 还原图像
    # nan_to_num：griddata 在极端条件下可能产生 NaN/Inf，需在 ifft2 前清零，
    # 否则 NaN 会通过 FFT 线性运算扩散到整个重建图像
    freq_domain_2d = np.nan_to_num(freq_domain_2d, nan=0.0, posinf=0.0, neginf=0.0)
    # ifftshift 先将零频移回左上角（FFT 约定原点位置），ifft2 计算，再 fftshift 将图像中心化
    recon_dfr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(freq_domain_2d)))

    # 朝向校正：极坐标→直角映射使重建相对输入转了 90°，此处转回输入同方位，使调用方
    # 直接 np.abs(recon_dfr) 即可（消除"须自行 rot90"的隐式契约）。np.rot90 绕几何中心
    # (n-1)/2 旋转，而本函数的 FFT 原点在 n//2：偶数 n 两者差半格，仅 rot90 会使重建整体
    # 错位 1 像素（奇数 n 恰好重合、无错位）。故偶数 n 补 roll(+1) 把原点移回 n//2。
    # 已由脉冲响应实测：n=32/64/65 峰值均精确对齐输入（偏移 0），根因见此三行之上。
    recon_dfr = np.rot90(recon_dfr)
    if num_detectors % 2 == 0:
        recon_dfr = np.roll(recon_dfr, 1, axis=0)

    return freq_domain_2d, fft_1d_display, recon_dfr


# -------------------------------------------------------------------------
# 辅助：小图准备 / 上采样
# -------------------------------------------------------------------------

def _finite_clip(arr: np.ndarray, n: int) -> np.ndarray:
    """把重建结果整理为可显示的有限 [0,1] 图：先中和 NaN/±Inf，再 clip 到 [0,1]，reshape 成 n×n。
    对齐 DFR 已有做法（compute_dfr 里对频域先 nan_to_num）。DMR/ART/SIRT 若遇病态/损坏输入
    （如弦图混入非有限值）会产出 NaN，而 np.clip 对 NaN 无效——NaN 会静默变黑图并污染 RMSE。
    此处保证任何情况下返回的都是确定、有限、可显示的图像。"""
    a = np.nan_to_num(arr.reshape(n, n), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _circle_mask(n: int) -> np.ndarray:
    """返回 n×n 的圆形掩码（内切圆内为 1，圆外为 0），float32。同 n 直接复用缓存。"""
    m = _CIRCLE_MASK_CACHE.get(n)
    if m is None:
        cy, cx = n // 2, n // 2
        Y, X = np.ogrid[:n, :n]
        m = ((Y - cy) ** 2 + (X - cx) ** 2 <= (n // 2) ** 2).astype(np.float32)
        _CIRCLE_MASK_CACHE[n] = m
    return m


# Shepp-Logan 头部模体的十椭圆参数——**Toft 修订版**（Toft 1996），而非 1974 原版。
# 每行 (a, b, x0, y0, phi_deg, gray)：a/b 为半轴长，(x0,y0) 为中心，phi 为逆时针旋转角，
# gray 为该椭圆叠加的衰减增量。坐标系为 [-1,1] 归一化，y 轴向上。
#
# 【为何取修订版】原版病灶与脑实质的对比度仅 0.01/0.02，肉眼近乎不可见——它是为
# 检验低对比分辨力设计的，拿来做教学演示则什么都看不出来。修订版把这些灰度整体
# 放大（脑实质 0.2、病灶 +0.1），是 skimage、MATLAB phantom() 等的默认。
# 【更要紧的是口径统一】研究一（experiments/recon_study.py）全程用 skimage 的
# shepp_logan_phantom，即此修订版；实验室与实验若各用一版，两边的 RMSE/对比度
# 数字就不可比。已实测两者结构对齐（NCC 见 tests 的模体用例）。
_SHEPP_LOGAN = (
    (0.6900, 0.9200, 0.0, 0.0000, 0.0, 1.0),    # 颅骨外缘
    (0.6624, 0.8740, 0.0, -0.0184, 0.0, -0.8),  # 脑实质（挖去颅骨内部，净值 0.2）
    (0.1100, 0.3100, 0.22, 0.0, -18.0, -0.2),   # 右侧脑室（净值 0，与背景同）
    (0.1600, 0.4100, -0.22, 0.0, 18.0, -0.2),   # 左侧脑室
    (0.2100, 0.2500, 0.0, 0.35, 0.0, 0.1),      # 以下六个为不同大小的"病灶"，净值 0.3
    (0.0460, 0.0460, 0.0, 0.10, 0.0, 0.1),
    (0.0460, 0.0460, 0.0, -0.10, 0.0, 0.1),
    (0.0460, 0.0230, -0.08, -0.605, 0.0, 0.1),
    (0.0230, 0.0230, 0.0, -0.606, 0.0, 0.1),
    (0.0230, 0.0460, 0.06, -0.605, 0.0, 0.1),
)


def shepp_logan(n: int = 256) -> np.ndarray:
    """生成 n×n 的 Shepp-Logan 头部模体，归一化到 [0,1] 并施加圆形掩码。

    自实现解析定义而非取自图像库：本模块服务的是教学用重建实验室，模体由十个
    解析椭圆叠加而成这件事本身就是教学内容；解析生成还能任意分辨率无插值失真，
    而位图模体缩放到 64² 做 DMR/ART 时会先糊掉一轮。

    圆形掩码与 compute_sinogram（skimage radon 的 circle=True）口径一致：弦图只
    编码内切圆内的信息（对解析反投影 iradon 成立，它显式把圆外置零；矩阵法 DMR/ART/SIRT
    并不置零），不掩掉的话误差图会在四角显示虚假的大误差。

    返回值域 [0,1]，与 generate_sinogram 对真实切片所做的归一化同口径，
    因此模体与真实数据走的是同一条重建链路，无需任何分支。
    """
    if n < 2:
        raise ValueError(f"模体尺寸至少 2，收到 {n}")
    # 像素中心落在 [-1,1] 上：用 n 个等分区间的中点，避免 n 为偶数时错开半个像素
    g = (np.arange(n, dtype=np.float32) + 0.5) * (2.0 / n) - 1.0
    X, Y = np.meshgrid(g, -g)     # y 轴向上，与模体参数表的坐标约定一致
    img = np.zeros((n, n), dtype=np.float32)
    for a, b, x0, y0, phi, gray in _SHEPP_LOGAN:
        t = np.deg2rad(phi)
        xr = (X - x0) * np.cos(t) + (Y - y0) * np.sin(t)
        yr = -(X - x0) * np.sin(t) + (Y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += gray
    # 标准参数下值域为 [0,1]，clip 只为防浮点边界溢出
    return (np.clip(img, 0.0, 1.0) * _circle_mask(n)).astype(np.float32)


def prepare_small_image(img_norm: np.ndarray, n: int, angle_range: float,
                        n_proj: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将归一化图像缩小为 n×n，施加圆形掩码后执行 Radon 变换。

    img_norm:    归一化到 [0,1] 的 2D 浮点数组（原始切片）
    n:           目标边长（16 / 32 / 64）
    angle_range: 角度覆盖范围（度，60 / 120 / 180 / 360）
    n_proj:      投影数量；None 时等于 angle_range。见 make_theta 的过采样说明。

    返回: (img_small, sinogram, theta)
      img_small: 缩小后加了圆形掩码的图像
      sinogram:  对 img_small 的 Radon 变换结果
      theta:     对应角度数组

    圆形掩码说明：
      radon(circle=True) 并不把圆外置零——它只在圆外有非零值时发一条 warning
    （skimage/transform/radon_transform.py 的 `if np.any(image[outside...]): warn(...)`），
    之后照常旋转整幅方图求和。掩码是为了让输入落进它的无警告前提、并与 iradon 的
    输出支撑对齐（iradon 才是显式置零的，见其 `reconstructed[out_...] = 0.0`）。
    实测：圆外单位冲激在 60 个角度里有 56 个投影质量为 0——旋转出方形即被静默丢弃，
    所以圆外像素的前向模型不自洽，掩掉它们是让输入合法，不是让 radon「只算圆内」。
      若不施加此掩码，
      原图角落有值而重建角落为0，误差图会在角落出现虚假大误差，
      迷惑用户误判算法质量。
    """
    # 与 compute_sinogram 同一道防线：那里的注释写着「一个 NaN 像素经 Radon 的线积分
    # 会污染所有穿过它的射线」，而本函数承担同一职责却没有它，且先过 ndimage.zoom
    # （三次样条）——局部 NaN 会被扩散成【整幅】。实测 128² 输入里 4×4 个 NaN：
    # 经本函数后小图与弦图的非有限占比达 1.0000 / 0.9997，而同一张图走
    # compute_sinogram 是 0.0000。下游 DMR/ART/SIRT 的 _finite_clip 会把 NaN 归零，
    # 于是界面拿到一张全黑图、没有任何提示——而同一份数据走 BP/FBP/DFR 完全正常。
    img_norm = np.nan_to_num(img_norm, nan=0.0, posinf=0.0, neginf=0.0)
    h0, w0 = img_norm.shape
    # ndimage.zoom 使用样条插值缩放，clip 防止插值超出 [0,1] 范围（Gibbs 现象）
    img_small = np.clip(
        ndimage.zoom(img_norm, (n / h0, n / w0)), 0.0, 1.0
    ).astype(np.float32)

    # 圆形掩码：与 radon(circle=True) 的处理区域完全对齐（缓存复用，同 n 不重复计算）
    img_small = img_small * _circle_mask(n)

    theta = make_theta(angle_range, n_proj)
    sinogram = radon(img_small, theta=theta, circle=True)
    return img_small, sinogram, theta


def upscale_recon(arr: np.ndarray, n: int) -> np.ndarray:
    """用 np.kron 做最近邻整数倍上采样，将小图放大到至少 256×256 显示。

    选择 kron 而非 zoom/resize 的原因：
      kron 执行严格的像素复制（每个原像素变成 scale×scale 的色块），
      保留像素块感——教学中展示重建分辨率差异的重要视觉线索。
      双线性插值会"美化"粗糙的 16×16 重建结果，失去教学价值。
    """
    scale = max(1, 256 // n)
    if scale == 1:
        return arr
    # np.kron(A, B)：将 A 的每个元素替换为 A[i,j]*B，等价于像素块复制
    return np.kron(arr, np.ones((scale, scale), dtype=np.float32))


# -------------------------------------------------------------------------
# 系统矩阵构建
# -------------------------------------------------------------------------

def build_system_matrix(n: int, theta: np.ndarray, cached_A: np.ndarray | None,
                        cached_A_key: tuple | None,
                        progress_cb: Callable[[int, int], None] | None = None
                        ) -> tuple[np.ndarray, tuple]:
    """逐像素构建系统矩阵 A，用于 DMR 和 ART/SIRT。

    n:          图像边长（A 的列数 = n²）
    theta:      投影角度数组
    cached_A:   上次缓存的矩阵（None 表示无缓存）
    cached_A_key: 上次缓存的 key
    progress_cb: 可选进度回调 progress_cb(j, n_pixels)，每步调用

    返回: (A, key)
      A:   系统矩阵，shape=(n_rays, n²)，float32
      key: 本次计算的缓存键

    缓存键 = (n, 角度数, 整张角度表的 sha1 指纹)；见 _theta_fingerprint 说明为何不能只取首尾。
    64×64 × 180角的 A 矩阵约需数分钟构建，缓存节省大量等待时间。
    """
    key = (n, len(theta), _theta_fingerprint(theta))
    if cached_A is not None and cached_A_key == key:
        return cached_A, key

    # 磁盘缓存命中：A 矩阵在 (n, n_angles, 角度表指纹) 完全相同时是确定值，
    # 第一次算完后写盘，之后所有进程启动都可秒级 np.load 复用，无任何精度损失。
    # 文件名嵌入 _WORKER_HASH：worker 代码改动后哈希变，旧缓存被自然忽略（无需手动清理）。
    cache_file = os.path.join(
        _MATRIX_CACHE_DIR,
        f"A_n{key[0]}_na{key[1]}_th{key[2]}_{_WORKER_HASH}.npy"
    )
    if os.path.exists(cache_file):
        try:
            A = np.load(cache_file)
            return A, key
        except Exception as e:
            print(f"Warning: matrix cache {cache_file} corrupted, rebuilding: {e}")

    n_pixels = n * n
    n_rays = radon(np.zeros((n, n), dtype=np.float32), theta=theta, circle=True).size
    A = np.zeros((n_rays, n_pixels), dtype=np.float32)

    # 每个批次处理的像素数：让每个 worker 大约承担 1/4 的工作量，
    # 多批次（batch 数 > worker 数）使负载均衡更好
    # os.cpu_count() 在无法探测时返回 None（回退 4）；不用 _mp.cpu_count()——后者探测不到会抛
    # NotImplementedError。与项目其余处（ai_engine/graphics_view/_read_dicom_dir）保持同一写法。
    n_workers = min(os.cpu_count() or 4, 8)
    batch = max(32, n_pixels // (n_workers * 4))
    jobs = [(i, min(i + batch, n_pixels), n, theta)
            for i in range(0, n_pixels, batch)]

    completed = 0
    # 'spawn' 避免 fork 复制 Qt 父进程状态到子进程导致崩溃
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        # imap_unordered：哪个 batch 先算完先回来，不等最慢的那个
        for start_j, end_j, cols in pool.imap_unordered(_matrix_worker, jobs):
            A[:, start_j:end_j] = cols
            completed += end_j - start_j
            if progress_cb is not None:
                progress_cb(completed, n_pixels)

    # 写入磁盘缓存，下次同参数直接 np.load 跳过整个并行构建过程
    try:
        os.makedirs(_MATRIX_CACHE_DIR, exist_ok=True)
        np.save(cache_file, A)
    except Exception as e:
        print(f"Warning: failed to write matrix cache {cache_file}: {e}")

    return A, key


# -------------------------------------------------------------------------
# DMR（直接矩阵重建）
# -------------------------------------------------------------------------

def compute_dmr(A: np.ndarray, p_vec: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """用最小二乘法求解 A·x = p，返回 (img_recon, error_time_ms)。

    A:     系统矩阵，shape=(n_rays, n²)
    p_vec: 弦图展平向量，shape=(n_rays,)
    n:     图像边长

    返回: (img_recon, elapsed_ms)
      img_recon: 重建图像，shape=(n, n)，值已 clip 到 [0, 1]
      elapsed_ms: lstsq 求解耗时（毫秒）

    数学原理：
      x* = argmin ||A·x - p||₂²  等价于伪逆 x* = (AᵀA)⁻¹Aᵀ·p
      rcond=None：使用机器精度作为截断阈值，处理近奇异矩阵（欠定系统）
    """
    start_t = time.perf_counter()
    x_recon, _, _, _ = np.linalg.lstsq(A, p_vec, rcond=None)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    # clip 将可能出现的负值（最小二乘的数学解不保证非负）截断到 [0, 1]；
    # _finite_clip 同时中和病态输入下的 NaN/Inf，保证输出可显示。
    img_recon = _finite_clip(x_recon, n)
    return img_recon, elapsed_ms


# -------------------------------------------------------------------------
# ART / SIRT 迭代重建
# -------------------------------------------------------------------------

def compute_art(A: np.ndarray, p_vec: np.ndarray, n: int, n_iter: int,
                cancel_check: Callable[[], bool] | None = None,
                progress_cb: Callable[[int], None] | None = None) -> tuple[np.ndarray, float]:
    """ART（Kaczmarz 迭代）重建，返回 (img_recon, elapsed_ms)。

    A:           系统矩阵，shape=(n_rays, n²)
    p_vec:       弦图展平向量
    n:           图像边长
    n_iter:      迭代次数
    cancel_check: 可选，cancel_check() 返回 True 时提前停止
    progress_cb: 可选，progress_cb(it) 每迭代一轮调用

    Kaczmarz 迭代公式（逐射线更新）：
      x ← x + (p_i - A_i·x) / ||A_i||² · A_i
    """
    x = np.zeros(n * n, dtype=np.float32)
    # einsum 'ij,ij->i'：逐行计算行向量的 L2 范数平方 ||A_i||²
    # 预计算避免内层循环重复计算，是 ART 的关键性能优化
    row_norms_sq = np.einsum('ij,ij->i', A, A)

    # 性能优化（保 bit-exact）：循环外预筛"有效射线索引"，
    # 跳过 row_norms_sq[i] <= 1e-10 的全零行；浮点除法保留原样，结果与原版本逐 bit 一致。
    valid_idx = np.flatnonzero(row_norms_sq > 1e-10)

    start_t = time.perf_counter()
    for it in range(n_iter):
        if cancel_check is not None and cancel_check():
            break
        for i in valid_idx:
            # 保留原 ART 更新公式：x += (p_i - A_i·x) / ||A_i||² · A_i
            # `+=` 已是 np.add(..., out=x) 原地累加，仅 `scale * A[i]` 创建 1 个临时数组
            x += ((p_vec[i] - A[i] @ x) / row_norms_sq[i]) * A[i]
        np.clip(x, 0.0, None, out=x)  # 非负约束，原地避免新数组分配
        if progress_cb is not None:
            progress_cb(it)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    img_recon = _finite_clip(x, n)
    return img_recon, elapsed_ms


def compute_sirt(A: np.ndarray, p_vec: np.ndarray, n: int, n_iter: int,
                 cancel_check: Callable[[], bool] | None = None,
                 progress_cb: Callable[[int], None] | None = None) -> tuple[np.ndarray, float]:
    """SIRT（同步迭代重建）重建，返回 (img_recon, elapsed_ms)。

    SIRT 更新公式（批量全射线更新）：
      x ← x + C · Aᵀ · (R · (p - A·x))
    其中 C = diag(1/列和)，R = diag(1/行和) 是归一化矩阵（用向量形式存储）。

    相比 ART：每次迭代计算量更大（矩阵乘法），但噪声鲁棒性更好，收敛更平滑。
    """
    x = np.zeros(n * n, dtype=np.float32)
    col_sums = A.sum(axis=0)  # 每列之和 = 每个像素被所有射线覆盖的总权重
    row_sums = A.sum(axis=1)  # 每行之和 = 每条射线穿过所有像素的总路径长度
    # 避免除以 0（完全不被射线覆盖的像素列/行）。
    # 注意不能写成 np.where(cond, 1.0/x, 0.0)：np.where **不短路**，两个分支都会被完整
    # 求值，故 1.0/x 仍会对 x=0 的位置做除法并抛 divide-by-zero 警告（结果虽被选对，
    # 但每次调用都会污染终端）。改用 np.divide 的 where= 参数，只在有效位置计算。
    C = np.zeros_like(col_sums, dtype=np.float64)
    R = np.zeros_like(row_sums, dtype=np.float64)
    np.divide(1.0, col_sums, out=C, where=col_sums > 1e-10)
    np.divide(1.0, row_sums, out=R, where=row_sums > 1e-10)
    C, R = C.astype(np.float32), R.astype(np.float32)

    start_t = time.perf_counter()
    # errstate：macOS 的 Accelerate BLAS 在 matmul 后会误置浮点异常标志——已最小复现，
    # 纯随机 float32 数组相乘同样报 divide-by-zero/overflow 而结果完全正确（无 nan/inf）。
    # 本循环每迭代含两次 matmul，20 轮会向终端刷上百条无意义警告，故局部抑制。
    # 不用全局 seterr：真正的数值异常仍由末尾的 _finite_clip 兜住并可见。
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for it in range(n_iter):
            if cancel_check is not None and cancel_check():
                break
            # x ← x + C·Aᵀ·(R·(p - A·x))
            # A @ x：正向投影（用当前估计值模拟弦图）
            # R * residual：对每条射线按其路径长度归一化
            # A.T @ ...：反投影（将射线残差分配回各像素）
            # C * ...：对每个像素按其被覆盖总权重归一化
            x = x + C * (A.T @ (R * (p_vec - A @ x)))
            x = np.clip(x, 0.0, None)  # 非负约束
            if progress_cb is not None:
                progress_cb(it)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    img_recon = _finite_clip(x, n)
    return img_recon, elapsed_ms


def _tv_grad(f: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """各向同性 TV 的梯度 ∂‖f‖_TV/∂f（前向差分 + eps 平滑）。

    ‖f‖_TV = Σ_{s,t} sqrt((f[s,t]-f[s-1,t])² + (f[s,t]-f[s,t-1])² )

    每个像素同时出现在三个相邻项的分母里（它自己、下方、右方），故梯度是三项之和。
    eps 只为让 sqrt 在平坦区可导，取 1e-8 时对结果的影响远小于噪声。
    边界用 mode='edge' 复制而非补零：补零会在图像边框上造出一圈虚假的强边缘，
    TV 会去"抹平"这圈不存在的结构。
    """
    fp = np.pad(f, 1, mode='edge')
    c = fp[1:-1, 1:-1]
    up, left = fp[0:-2, 1:-1], fp[1:-1, 0:-2]
    dn, right = fp[2:, 1:-1], fp[1:-1, 2:]
    dnl, upr = fp[2:, 0:-2], fp[0:-2, 2:]

    d_c = np.sqrt((c - up) ** 2 + (c - left) ** 2 + eps)          # 分母 @ (s,t)
    d_dn = np.sqrt((dn - c) ** 2 + (dn - dnl) ** 2 + eps)         # 分母 @ (s+1,t)
    d_r = np.sqrt((right - upr) ** 2 + (right - c) ** 2 + eps)    # 分母 @ (s,t+1)
    return (2 * c - up - left) / d_c - (dn - c) / d_dn - (right - c) / d_r


def compute_asdpocs(A: np.ndarray, p_vec: np.ndarray, n: int, n_iter: int = 300,
                    a: float = 0.2, n_grad: int = 20,
                    beta: float = 1.0, beta_red: float = 0.995,
                    a_red: float = 0.95, r_max: float = 0.95,
                    eps_data: float = 0.0,
                    cancel_check: Callable[[], bool] | None = None,
                    progress_cb: Callable[[int], None] | None = None
                    ) -> tuple[np.ndarray, float]:
    """ASD-POCS（TV 正则化的压缩感知重建），返回 (img_recon, elapsed_ms)。

    实现严格照 Sidky & Pan, Phys Med Biol 53(17):4777-4807 (2008) §2.4.2 的伪码
    第 1–24 行。每一轮外循环 = 一次带松弛的 ART 扫掠（POCS 步，投影到数据一致集与
    非负集）+ n_grad 次 TV 最速下降。

    参数（默认值取自该文，勿凭习惯改动）：
      n_iter=300  外循环轮数。**这是最容易踩的坑**：按本仓库 ART=5 / SIRT=100 的
                  习惯随手取 20，ASD-POCS 会比 FBP 还差（60 视角实测 0.106 vs
                  0.088）。实测最优点落在 145–300+，30 视角在 300 处仍未触底。
      a=0.2       TV 步长相对 POCS 步长的初始比例（文中 α）。
      n_grad=20   每轮 TV 最速下降的内层次数（文中 N_grad）。
      beta=1.0 / beta_red=0.995   ART 松弛因子及其逐轮衰减（伪码 L1、L22）。
      a_red=0.95 / r_max=0.95     TV 步长的自适应缩减规则（伪码 L21）：
                  若 TV 步的位移 d_g > r_max·d_p 且数据残差 d_d > eps_data，
                  说明 TV 走得比 POCS 还远、开始压过数据项，则 dtvg *= a_red。
      eps_data=0.0  数据残差容限（伪码 L21 的第二个条件）。

    三个易错点（都已按文中写法处理，改动前请回读伪码）：
      1. **dtvg 只在首轮初始化**（L13 `if {first iteration}`），此后只经 L21 单调
         缩小，绝不每轮重新按 a·d_p 赋值——否则步长永远不退火，算法性质全变。
      2. **TV 步长要对梯度做 L2 归一化**（L17-18 `f -= dtvg·df/‖df‖`）。这一步让
         dtvg 成为图像空间里的 L2 距离，与 d_p 同量纲，α 才是无量纲的——也正因
         如此，α 能跨几何/离散化移植（实测扫 100× 量程，RMSE 只从 0.0116 动到
         0.0126）。省掉归一化，α 就绑死在特定的 A 与 p 尺度上。
      3. **返回 POCS 步后、TV 步前的 f_res**（L9 捕获、L24 返回），不是 TV 之后的图。

    不复用 compute_art 作为 POCS 步：后者把 beta 硬编码为 1 且不暴露松弛参数，
    而本算法要求 beta 逐轮 ×0.995。
    """
    x = np.zeros(n * n, dtype=np.float32)
    row_norms_sq = np.einsum('ij,ij->i', A, A)
    valid_idx = np.flatnonzero(row_norms_sq > 1e-10)
    dtvg = None
    f_res = x

    start_t = time.perf_counter()
    # errstate 的理由同 compute_sirt：macOS Accelerate BLAS 在 matmul 后会误置浮点
    # 异常标志，本循环每轮含一次 A @ x，300 轮会向终端刷上百条无意义警告。
    # 真正的数值异常仍由末尾的 _finite_clip 兜住并可见。
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
      for it in range(n_iter):
          if cancel_check is not None and cancel_check():
              break
          f0 = x.copy()                                        # L6
          for i in valid_idx:                                  # L7 带松弛的 ART 扫掠
              x += (beta * (p_vec[i] - A[i] @ x) / row_norms_sq[i]) * A[i]
          np.clip(x, 0.0, None, out=x)                         # L8 非负约束
          f_res = x.copy()                                     # L9 ← 最终返回的就是它
          d_d = float(np.linalg.norm(A @ x - p_vec))           # L11 数据残差
          d_p = float(np.linalg.norm(x - f0))                  # L12 POCS 步位移
          if dtvg is None:                                     # L13 仅首轮初始化
              dtvg = a * d_p
          f0 = x.copy()                                        # L14
          img = x.reshape(n, n)
          for _ in range(n_grad):                              # L15-19 TV 最速下降
              df = _tv_grad(img)
              nrm = float(np.linalg.norm(df))
              if nrm > 1e-12:
                  img -= (dtvg / nrm) * df                     # L17-18 步长按 L2 归一化
          x = img.reshape(-1)
          d_g = float(np.linalg.norm(x - f0))                  # L20 TV 步位移
          if d_g > r_max * d_p and d_d > eps_data:             # L21 自适应缩步
              dtvg *= a_red
          beta *= beta_red                                     # L22
          if progress_cb is not None:
              progress_cb(it)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    img_recon = _finite_clip(f_res, n)
    return img_recon, elapsed_ms


# -------------------------------------------------------------------------
# 学习式后处理重建（研究三产物，见 experiments/recon_dl.py）
# -------------------------------------------------------------------------

_DL_SESSION = None          # onnxruntime 会话缓存：每次重建都重建会话会白等数百毫秒
_DL_SESSION_PATH = None


def dl_available(model_path: str) -> bool:
    """模型图、外部权重、onnxruntime 三者是否都就位。缺任一则调用方应保持功能禁用——
    功能可以缺，但不能假装能用。

    【为什么必须查 .data】torch.onnx 对超过阈值的权重采用外部数据格式：`.onnx` 只有
    20KB 图，真正的 7.7MB 权重在同名 `.onnx.data` 里，且 ONNX 按【相对 .onnx 的路径】
    解析它。仓库只提交图、不提交权重（与 organs.onnx / organs.onnx.data 同一套约定），
    所以只查 `.onnx` 存在会让按钮在权重缺失时假装可用，点下去才报错。
    """
    if not model_path or not os.path.exists(model_path):
        return False
    ext_data = model_path + ".data"
    if os.path.exists(ext_data) is False and os.path.getsize(model_path) < 1_000_000:
        # 图很小又没有伴生权重文件 → 权重必然缺失
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def compute_dl_recon(fbp_img: np.ndarray, model_path: str) -> tuple[np.ndarray, float]:
    """用学习式后处理网络去除稀疏角 FBP 的条纹伪影，返回 (重建图, 耗时ms)。

    输入必须是 **ramp 滤波** 的 FBP：模型即以此为输入训练。喂 hann 的结果会偏——
    hann 已在滤波阶段把高频连同细节滤掉，网络无从恢复那些信息。

    归一化：模型在 [0,1] 值域的模体上训练，故按输入自身的极值线性映射进 [0,1]，
    推理后再映射回原值域。不这样做的话，HU 量级的输入会完全落在训练分布之外。

    非有限值防护：与 compute_fbp/compute_dfr 一致——NaN 经卷积会扩散到整幅输出。
    """
    import onnxruntime as ort
    global _DL_SESSION, _DL_SESSION_PATH
    start_t = time.perf_counter()
    x = np.nan_to_num(np.asarray(fbp_img, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo if hi > lo else 1.0
    xn = ((x - lo) / span).astype(np.float32)[None, None]
    # 网络有 3 次 2× 下采样，边长须为 8 的倍数；不足则右下补零，推理后裁回
    h, w = xn.shape[2], xn.shape[3]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        xn = np.pad(xn, ((0, 0), (0, 0), (0, ph), (0, pw)), mode='edge')
    if _DL_SESSION is None or _DL_SESSION_PATH != model_path:
        _DL_SESSION = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        _DL_SESSION_PATH = model_path
    y = _DL_SESSION.run(None, {_DL_SESSION.get_inputs()[0].name: xn})[0]
    y = y[0, 0, :h, :w]
    out = y * span + lo                      # 映射回输入自身的值域
    # 刻意不用 _finite_clip：它为 DMR/ART 而写，会把结果硬 clip 到 [0,1] 且只接受方阵。
    # 本函数的输出必须留在输入的值域里（HU 输入就该给回 HU），clip 到 [0,1] 会把整幅
    # 图压平——而这种破坏在显示时因为要重新归一化，肉眼完全看不出来。
    out = np.nan_to_num(out, nan=0.0, posinf=hi, neginf=lo)
    return out.astype(np.float32), (time.perf_counter() - start_t) * 1000
