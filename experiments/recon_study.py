# =============================================================================
# 重建量化研究：低剂量 CT 的剂量-质量权衡
# ---------------------------------------------------------------------------
# 目的：把 recon.py 里已实现的重建算法，从"我实现了"升级为"我量化测量了"。
# 被重建对象：Shepp-Logan 标准体模（CT 重建文献通用基准，可复现、无患者数据）。
# 被测代码：全部直接调用 recon.py 中 GUI 所用的同一批函数，测的即产品代码本身。
#
# 三组实验：
#   A  剂量-质量曲线：FBP(ramp) 下 RMSE/SSIM 随投影角度数（=剂量代理）的衰减
#   B  滤波器对比：5 种 FBP 滤波器在稀疏角度下的表现排序
#   C  解析 vs 迭代：低剂量下 FBP / DMR / ART / SIRT 的对比
#
# 用法：
#   python experiments/recon_study.py           # 跑 A + B（快，仅 FBP，无系统矩阵）
#   python experiments/recon_study.py c          # 单跑 C（需构建系统矩阵，较慢，有磁盘缓存）
#   python experiments/recon_study.py a b c      # 全跑
#
# 产出：experiments/results/ 下的 PNG 图与 CSV 表（可直接放进技术报告）。
# =============================================================================

import os
import sys
import csv
import warnings

import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use("Agg")  # 无显示环境，仅出图片文件

# 图表文字统一用英文：一来 matplotlib 默认字体（DejaVu Sans）无中文字形，
# 二来港校申请材料本身是英文，英文图可直接进技术报告。
# 静默 skimage 载入体模时的良性 RGB 转换除零警告（不影响体模数值）。
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# 让脚本能从 experiments/ 子目录导入项目根的 recon.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import recon  # noqa: E402  被测模块（产品代码）

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


# -------------------------------------------------------------------------
# 体模与度量
# -------------------------------------------------------------------------

def get_phantom(n):
    """Shepp-Logan 体模，缩放到 n×n，归一化 [0,1]，施加与 radon(circle=True) 对齐的圆形掩码。
    掩码后的图即"真值"(ground truth)：正弦图只编码内切圆内信息，圆外重建恒为 0，
    故只在圆内比较才公平。"""
    p = shepp_logan_phantom().astype(np.float32)      # 400×400, [0,1]
    p = ndimage.zoom(p, (n / p.shape[0], n / p.shape[1]))
    p = np.clip(p, 0.0, 1.0).astype(np.float32)
    p *= recon._circle_mask(n)                        # 复用产品代码的同一掩码
    return p


def roi_metrics(gt, rec):
    """在圆形 ROI 内比较重建 rec 与真值 gt，返回 (rmse, nrmse, ssim, psnr)。
    rec 可能因 FBP 频域振铃略超 [0,1]，度量按原值计算不裁剪（如实反映误差）。"""
    n = gt.shape[0]
    mask = recon._circle_mask(n) > 0
    diff = (rec.astype(np.float32) - gt)[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rng = float(gt[mask].max() - gt[mask].min()) or 1.0
    nrmse = rmse / rng
    # SSIM 需二维窗口，取整图计算（圆外两者皆 ~0，几乎不影响）；data_range 按真值 [0,1]
    ssim = float(structural_similarity(gt, rec.astype(np.float32), data_range=1.0))
    psnr = float(peak_signal_noise_ratio(gt[mask], np.clip(rec, 0, 1).astype(np.float32)[mask],
                                         data_range=1.0))
    return rmse, nrmse, ssim, psnr


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


# -------------------------------------------------------------------------
# 实验 A：剂量-质量曲线
# -------------------------------------------------------------------------

def experiment_a(n=256, angle_range=180.0):
    """FBP(ramp) 下，RMSE 与 SSIM 随投影角度数的变化。投影数越少≈剂量越低。"""
    print("[A] 剂量-质量曲线 (FBP ramp, n=%d, 角度范围 %g°)" % (n, angle_range))
    gt = get_phantom(n)
    n_projs = [15, 20, 30, 45, 60, 90, 120, 180, 240, 360]
    rows = []
    for npj in n_projs:
        theta = recon.make_theta(angle_range, npj)
        sino = recon.compute_sinogram(gt, theta)
        _, fbp = recon.compute_fbp(sino, theta, "ramp")
        rmse, nrmse, ssim, psnr = roi_metrics(gt, fbp)
        rows.append([npj, round(rmse, 5), round(nrmse, 5), round(ssim, 4), round(psnr, 2)])
        print(f"    proj={npj:4d}  RMSE={rmse:.4f}  SSIM={ssim:.3f}  PSNR={psnr:.1f}dB")

    _write_csv(os.path.join(RESULTS, "exp_a_dose_quality.csv"),
               ["n_projections", "rmse", "nrmse", "ssim", "psnr_db"], rows)

    xs = [r[0] for r in rows]
    rmse_ys = [r[1] for r in rows]
    ssim_ys = [r[3] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(xs, rmse_ys, "o-", color="#2a78d6", lw=2)
    ax1.set_xscale("log"); ax1.set_xlabel("Number of projections (dose proxy)"); ax1.set_ylabel("RMSE (in-circle)")
    ax1.set_title("Reconstruction error vs dose"); ax1.grid(True, alpha=0.3)
    ax2.plot(xs, ssim_ys, "s-", color="#1baf7a", lw=2)
    ax2.set_xscale("log"); ax2.set_xlabel("Number of projections (dose proxy)"); ax2.set_ylabel("SSIM (higher is better)")
    ax2.set_title("Structural similarity vs dose"); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
    fig.suptitle(f"Shepp-Logan phantom {n}x{n}  |  FBP (Ram-Lak filter)  |  angular range {angle_range:g} deg")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp_a_dose_quality.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")
    return rows


# -------------------------------------------------------------------------
# 实验 B：滤波器对比
# -------------------------------------------------------------------------

def experiment_b(n=256, angle_range=180.0):
    """5 种 FBP 滤波器在不同投影数（尤其稀疏角度）下的 RMSE 排序。"""
    print("[B] 滤波器对比 (n=%d, 角度范围 %g°)" % (n, angle_range))
    gt = get_phantom(n)
    filters = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
    n_projs = [20, 30, 45, 60, 90, 180]
    # 表：行=投影数，列=各滤波器 RMSE
    table = {flt: [] for flt in filters}
    for npj in n_projs:
        theta = recon.make_theta(angle_range, npj)
        sino = recon.compute_sinogram(gt, theta)
        line = [f"proj={npj:4d}"]
        for flt in filters:
            _, fbp = recon.compute_fbp(sino, theta, flt)
            rmse, *_ = roi_metrics(gt, fbp)
            table[flt].append(rmse)
            line.append(f"{flt}={rmse:.4f}")
        print("    " + "  ".join(line))

    rows = [[npj] + [round(table[flt][i], 5) for flt in filters]
            for i, npj in enumerate(n_projs)]
    _write_csv(os.path.join(RESULTS, "exp_b_filters.csv"),
               ["n_projections"] + filters, rows)

    colors = {"ramp": "#2a78d6", "shepp-logan": "#1baf7a", "cosine": "#eda100",
              "hamming": "#e34948", "hann": "#4a3aa7"}
    markers = {"ramp": "o", "shepp-logan": "s", "cosine": "^", "hamming": "D", "hann": "v"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for flt in filters:
        ax.plot(n_projs, table[flt], marker=markers[flt], color=colors[flt], lw=1.8,
                label=flt if flt != "ramp" else "ramp (Ram-Lak)")
    ax.set_xscale("log"); ax.set_xlabel("Number of projections"); ax.set_ylabel("RMSE (in-circle)")
    ax.set_title(f"FBP filter comparison  |  Shepp-Logan {n}x{n}  |  {angle_range:g} deg")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp_b_filters.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")
    return rows


# -------------------------------------------------------------------------
# 光子噪声模型（低剂量 CT 的物理本质）
# -------------------------------------------------------------------------

def add_poisson_noise(sino, i0, rng, mu_max=2.0):
    """按 Beer-Lambert + Poisson 光子统计给正弦图加噪，模拟低剂量 CT。
    I0 为入射光子数（剂量代理，越小越低剂量、噪声越大）：
      透射光子 N ~ Poisson(I0·e^{-p})，含噪投影 p' = -ln(N/I0)。
    把投影线性映射到最大衰减 mu_max 使 exp() 尺度合理，加噪后再映射回原尺度，
    保证 FBP/迭代重建接到的正弦图仍在正确量纲。"""
    smax = float(sino.max()) or 1.0
    p = sino / smax * mu_max                          # 投影 → 衰减 [0, mu_max]
    counts = rng.poisson(np.maximum(i0 * np.exp(-p), 1e-8))
    counts = np.maximum(counts, 1.0)                  # 防 log(0)
    p_noisy = -np.log(counts / i0)                    # 反解回衰减
    return (p_noisy / mu_max * smax).astype(np.float32)  # 衰减 → 投影，恢复尺度


# -------------------------------------------------------------------------
# 实验 C：解析 vs 迭代（低剂量 + 光子噪声）
# -------------------------------------------------------------------------

def experiment_c(n=64, angle_range=180.0, n_projs=(30, 60, 90),
                 art_iter=5, sirt_iter=100, photon_i0=3e4, seed=0):
    """低剂量（稀疏角度 + Poisson 光子噪声）下 FBP / DMR / ART / SIRT 对比。
    加噪是关键：无噪声时超定的 DMR(最小二乘)会几乎精确还原（RMSE≈0），掩盖其病态性；
    真实低剂量 CT 的本质是光子少→噪声大，此时朴素 lstsq 放大噪声，带非负约束的迭代法更鲁棒。
    需系统矩阵（n=64 首次构建较慢，之后走磁盘缓存）。"""
    print("[C] 解析 vs 迭代 (n=%d, 角度范围 %g°, ART %d 轮, SIRT %d 轮, 光子 I0=%.0e)"
          % (n, angle_range, art_iter, sirt_iter, photon_i0))
    gt = get_phantom(n)
    rng = np.random.default_rng(seed)    # 固定种子，噪声可复现
    methods = ["FBP", "DMR", "ART", "SIRT"]
    results = {m: [] for m in methods}   # 每方法在各 n_proj 下的 RMSE
    recon_gallery = {}                   # (method, npj) -> 重建图，供拼图
    cached_A, cached_key = None, None

    for npj in n_projs:
        theta = recon.make_theta(angle_range, npj)
        sino = add_poisson_noise(recon.compute_sinogram(gt, theta), photon_i0, rng)
        p_vec = sino.ravel()

        _, fbp = recon.compute_fbp(sino, theta, "ramp")

        cached_A, cached_key = recon.build_system_matrix(n, theta, cached_A, cached_key)
        A = cached_A
        dmr, _ = recon.compute_dmr(A, p_vec, n)
        art, _ = recon.compute_art(A, p_vec, n, art_iter)
        sirt, _ = recon.compute_sirt(A, p_vec, n, sirt_iter)

        for m, img in zip(methods, [fbp, dmr, art, sirt]):
            rmse, *_ = roi_metrics(gt, img)
            results[m].append(rmse)
            recon_gallery[(m, npj)] = np.clip(img, 0, 1)
        print("    proj=%3d  " % npj +
              "  ".join(f"{m}={results[m][-1]:.4f}" for m in methods))

    rows = [[npj] + [round(results[m][i], 5) for m in methods]
            for i, npj in enumerate(n_projs)]
    _write_csv(os.path.join(RESULTS, "exp_c_analytic_vs_iterative.csv"),
               ["n_projections"] + methods, rows)

    # 分组柱状图：x=投影数，每组 4 根柱=4 方法
    colors = {"FBP": "#2a78d6", "DMR": "#eda100", "ART": "#1baf7a", "SIRT": "#4a3aa7"}
    x = np.arange(len(n_projs)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, m in enumerate(methods):
        ax.bar(x + (k - 1.5) * w, results[m], w, label=m, color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels([str(p) for p in n_projs])
    ax.set_xlabel("Number of projections (dose)"); ax.set_ylabel("RMSE (in-circle)")
    ax.set_title(f"Analytic vs iterative under photon noise  |  Shepp-Logan {n}x{n}  |  "
                 f"{angle_range:g} deg  |  I0={photon_i0:.0e}")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp_c_analytic_vs_iterative.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")

    # 视觉拼图：最稀疏角度下 真值 + 4 方法重建
    npj0 = n_projs[0]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Ground truth")
    for ax, m in zip(axes[1:], methods):
        img = recon_gallery[(m, npj0)]
        rmse = results[m][n_projs.index(npj0)]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{m}\nRMSE={rmse:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Sparse-dose visual comparison under photon noise  |  {npj0} projections  |  "
                 f"{angle_range:g} deg  |  I0={photon_i0:.0e}")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp_c_gallery.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")
    return rows


# -------------------------------------------------------------------------
# 入口
# -------------------------------------------------------------------------

def main():
    args = [a.lower() for a in sys.argv[1:]] or ["a", "b"]
    if "a" in args:
        experiment_a()
    if "b" in args:
        experiment_b()
    if "c" in args:
        experiment_c()
    print("完成。图表与 CSV 见 experiments/results/")


if __name__ == "__main__":
    main()
