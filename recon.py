# =============================================================================
# 重建算法纯计算模块
# 负责：所有 CT 重建相关的纯数值计算，不依赖任何 Qt/UI 代码
# 调用方：MedicalViewer 的方法作为薄包装层，负责读取 UI 状态并展示结果
# =============================================================================

import os
import time
import hashlib
import inspect
import multiprocessing as _mp
import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from skimage.transform import radon, iradon

# -------------------------------------------------------------------------
# 模块级缓存（顶部集中声明，便于阅读时一眼看清全局可变状态）
# -------------------------------------------------------------------------

# 系统矩阵磁盘缓存目录：放在本模块所在目录下的 .matrix_cache/，
# 避免与项目根混淆；首次访问时按需创建。
_MATRIX_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matrix_cache")

# 圆形掩码缓存：{n: mask}。同 n 在多次重建中反复使用，避免重复 ogrid + 比较运算。
# key 空间极小（≤ 4 种 n），无需淘汰策略。
_CIRCLE_MASK_CACHE = {}

# DFR Delaunay 三角剖分缓存：key=(num_detectors, len(theta), θ_start, θ_end)。
# scipy.griddata 内部每次都会重做 Delaunay 三角剖分（O(N log N) 但常数极大），
# 同参数复算时复用 Delaunay 对象可省去 60%-80% 的 DFR 总耗时；
# 数学上完全等价，三角剖分由几何点集唯一确定。
# key 空间小（≤ 4 探测器数 × 4 角度配置 ≈ 16 entry），无需淘汰策略。
_DFR_TRI_CACHE = {}


# -------------------------------------------------------------------------
# 系统矩阵并行 worker（必须是模块顶层函数，multiprocessing 才能 pickle）
# -------------------------------------------------------------------------

def _matrix_worker(args):
    """计算系统矩阵 A 中 [start_j, end_j) 列对应像素的 Radon 贡献。
    子进程独立导入 skimage，避免父进程 Qt 状态被复制到子进程。
    """
    start_j, end_j, n, theta = args
    from skimage.transform import radon as _radon
    import numpy as _np

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


def _purge_stale_matrix_cache():
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

def compute_sinogram(img_norm, theta):
    """对归一化图像执行 Radon 变换，返回弦图。

    img_norm: 归一化到 [0,1] 的 2D 浮点数组
    theta:    投影角度数组（度），如 np.linspace(0, 180, 180, endpoint=False)
    返回:     sinogram，shape=(探测器数, 角度数)
    """
    # circle=True：只处理图像内切圆区域，与 iradon 默认行为一致，避免角落 padding 引入伪影
    return radon(img_norm, theta=theta, circle=True)


def make_theta(n_angles):
    """根据角度总数生成均匀分布的投影角度数组。

    n_angles: 60 / 120 / 180 / 360 中的一个
    返回:     np.ndarray，长度 = n_angles，范围 [0, n_angles)
    """
    return np.linspace(0., float(n_angles), n_angles, endpoint=False)


# -------------------------------------------------------------------------
# BP / FBP
# -------------------------------------------------------------------------

def compute_bp(sinogram, theta):
    """纯反投影（不滤波），返回重建图像。

    缺陷：低频分量被过度叠加，边缘极度模糊（星形伪影）。
    对比目的：展示滤波（FBP）对图像质量的改善效果。
    """
    # filter_name=None 表示纯反投影，不做任何频域滤波
    return iradon(sinogram, theta=theta, filter_name=None, circle=True)


def compute_fbp(sinogram, theta, filter_name):
    """滤波反投影（FBP），返回 (recon_bp_unfiltered, recon_fbp)。

    同时返回未滤波结果供对比展示，避免调用方重复计算。
    filter_name: 'ramp' / 'shepp-logan' / 'cosine' / 'hamming' / 'hann'
                 注意：UI 显示 'Ram-Lak'，调用前需映射为 'ramp'
    """
    # skimage.transform.iradon 内部滤波器名为 'ramp'，调用前需做名称映射
    if filter_name.lower() == "ram-lak":
        filter_name = "ramp"
    recon_bp = iradon(sinogram, theta=theta, filter_name=None, circle=True)
    recon_fbp = iradon(sinogram, theta=theta, filter_name=filter_name, circle=True)
    return recon_bp, recon_fbp


# -------------------------------------------------------------------------
# DFR（直接傅里叶重建）
# -------------------------------------------------------------------------

def compute_dfr(sinogram, theta):
    """直接傅里叶重建法（DFR），基于傅里叶中心切片定理。

    返回: (freq_domain_2d, fft_1d_display, recon_dfr)
      freq_domain_2d:   插值后的二维复数频域矩阵，供"二维频域分布"视图展示
      fft_1d_display:   log1p 压缩后的一维频谱幅度图，供"一维FFT谱"视图展示
      recon_dfr:        2D 逆 FFT 重建结果（复数，取 abs 后显示）

    算法步骤：
      1. 对弦图每列（沿探测器方向）做 1D FFT → 极坐标频域样本
      2. 将极坐标 (r, θ) 样本插值到直角坐标网格（griddata，method='linear'）
      3. 对插值后的 2D 频域做 2D 逆 FFT → 重建图像
    """
    num_detectors, num_angles = sinogram.shape

    # 步骤1：对弦图沿探测器方向（axis=0）做 1D FFT
    # ifftshift 将数据中心移到 FFT 起点（左端），fft 计算，再 fftshift 将零频移回中心
    # 这样 proj_fft[num_detectors//2, :] 对应零频（直流分量）
    proj_fft = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(sinogram, axes=0), axis=0),
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
    tri_key = (num_detectors, len(theta),
               round(float(theta[0]), 4), round(float(theta[-1]), 4))
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

    return freq_domain_2d, fft_1d_display, recon_dfr


# -------------------------------------------------------------------------
# 辅助：小图准备 / 上采样
# -------------------------------------------------------------------------

def _circle_mask(n):
    """返回 n×n 的圆形掩码（内切圆内为 1，圆外为 0），float32。同 n 直接复用缓存。"""
    m = _CIRCLE_MASK_CACHE.get(n)
    if m is None:
        cy, cx = n // 2, n // 2
        Y, X = np.ogrid[:n, :n]
        m = ((Y - cy) ** 2 + (X - cx) ** 2 <= (n // 2) ** 2).astype(np.float32)
        _CIRCLE_MASK_CACHE[n] = m
    return m


def prepare_small_image(img_norm, n, n_angles):
    """将归一化图像缩小为 n×n，施加圆形掩码后执行 Radon 变换。

    img_norm:  归一化到 [0,1] 的 2D 浮点数组（原始切片）
    n:         目标边长（16 / 32 / 64）
    n_angles:  投影角度数（60 / 120 / 180 / 360）

    返回: (img_small, sinogram, theta)
      img_small: 缩小后加了圆形掩码的图像
      sinogram:  对 img_small 的 Radon 变换结果
      theta:     对应角度数组

    圆形掩码说明：
      radon(circle=True) 只处理内切圆区域；若不施加此掩码，
      原图角落有值而重建角落为0，误差图会在角落出现虚假大误差，
      迷惑用户误判算法质量。
    """
    h0, w0 = img_norm.shape
    # ndimage.zoom 使用样条插值缩放，clip 防止插值超出 [0,1] 范围（Gibbs 现象）
    img_small = np.clip(
        ndimage.zoom(img_norm, (n / h0, n / w0)), 0.0, 1.0
    ).astype(np.float32)

    # 圆形掩码：与 radon(circle=True) 的处理区域完全对齐（缓存复用，同 n 不重复计算）
    img_small = img_small * _circle_mask(n)

    theta = make_theta(n_angles)
    sinogram = radon(img_small, theta=theta, circle=True)
    return img_small, sinogram, theta


def upscale_recon(arr, n):
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

def build_system_matrix(n, theta, cached_A, cached_A_key, progress_cb=None):
    """逐像素构建系统矩阵 A，用于 DMR 和 ART/SIRT。

    n:          图像边长（A 的列数 = n²）
    theta:      投影角度数组
    cached_A:   上次缓存的矩阵（None 表示无缓存）
    cached_A_key: 上次缓存的 key
    progress_cb: 可选进度回调 progress_cb(j, n_pixels)，每步调用

    返回: (A, key)
      A:   系统矩阵，shape=(n_rays, n²)，float32
      key: 本次计算的缓存键

    缓存键 = (n, 角度数, 起始角, 终止角)；图像尺寸和角度配置不变时直接复用。
    64×64 × 180角的 A 矩阵约需数分钟构建，缓存节省大量等待时间。
    """
    key = (n, len(theta), round(float(theta[0]), 4), round(float(theta[-1]), 4))
    if cached_A is not None and cached_A_key == key:
        return cached_A, key

    # 磁盘缓存命中：A 矩阵在 (n, n_angles, θ_start, θ_end) 完全相同时是确定值，
    # 第一次算完后写盘，之后所有进程启动都可秒级 np.load 复用，无任何精度损失。
    # 文件名嵌入 _WORKER_HASH：worker 代码改动后哈希变，旧缓存被自然忽略（无需手动清理）。
    cache_file = os.path.join(
        _MATRIX_CACHE_DIR,
        f"A_n{key[0]}_na{key[1]}_t{key[2]:.4f}_{key[3]:.4f}_{_WORKER_HASH}.npy"
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
    n_workers = min(_mp.cpu_count(), 8)
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

def compute_dmr(A, p_vec, n):
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

    # clip 将可能出现的负值（最小二乘的数学解不保证非负）截断到 [0, 1]
    img_recon = np.clip(x_recon.reshape(n, n), 0.0, 1.0).astype(np.float32)
    return img_recon, elapsed_ms


# -------------------------------------------------------------------------
# ART / SIRT 迭代重建
# -------------------------------------------------------------------------

def compute_art(A, p_vec, n, n_iter, cancel_check=None, progress_cb=None):
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

    img_recon = np.clip(x.reshape(n, n), 0.0, 1.0).astype(np.float32)
    return img_recon, elapsed_ms


def compute_sirt(A, p_vec, n, n_iter, cancel_check=None, progress_cb=None):
    """SIRT（同步迭代重建）重建，返回 (img_recon, elapsed_ms)。

    SIRT 更新公式（批量全射线更新）：
      x ← x + C · Aᵀ · (R · (p - A·x))
    其中 C = diag(1/列和)，R = diag(1/行和) 是归一化矩阵（用向量形式存储）。

    相比 ART：每次迭代计算量更大（矩阵乘法），但噪声鲁棒性更好，收敛更平滑。
    """
    x = np.zeros(n * n, dtype=np.float32)
    col_sums = A.sum(axis=0)  # 每列之和 = 每个像素被所有射线覆盖的总权重
    row_sums = A.sum(axis=1)  # 每行之和 = 每条射线穿过所有像素的总路径长度
    # where 保护：避免除以 0（完全不被射线覆盖的像素列/行）
    C = np.where(col_sums > 1e-10, 1.0 / col_sums, 0.0).astype(np.float32)
    R = np.where(row_sums > 1e-10, 1.0 / row_sums, 0.0).astype(np.float32)

    start_t = time.perf_counter()
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

    img_recon = np.clip(x.reshape(n, n), 0.0, 1.0).astype(np.float32)
    return img_recon, elapsed_ms
