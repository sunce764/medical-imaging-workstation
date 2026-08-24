"""研究一补测：TV 正则化基线（ASD-POCS），以及它的优势在什么条件下消失。

动机：experiments/recon_dl.py 的文件头写着「本研究目前缺 TV 基线」——而稀疏角
重建的标准对照就是全变差（total variation）正则化迭代法。缺了它，研究一「FBP /
ART / SIRT 三选一」的比较是在一个不完整的方法集内做的，研究三拿学习式后处理去
比「最好的线性滤波」也少了一个更强的非学习对手。

本脚本补上 ASD-POCS（Sidky & Pan, Phys Med Biol 53(17):4777-4807, 2008），
实现在 recon.compute_asdpocs，参数默认值取自该文。

**结论不是「TV 最好」，而是「TV 的优势随信噪比单调、并在低剂量端反转」**：
在本研究的工作点（η≈0.9%）ASD-POCS 把 in-circle RMSE 压到 SIRT 最优点的一半，
但把剂量降到 η≈9% 时优势归零甚至转负，且此时 TV 的 SSIM 反而更差（阶梯效应）。
只报 RMSE 会把后半段说反，故本脚本同时输出 SSIM。

两个对照，各自防一种「赢在起跑线」的质疑：
  1. **等口径**：ART / SIRT / ASD-POCS 一律 oracle-stopped（取各自扫描区间内的
     最优轮），与 recon_stopping.py 同一约定。固定轮数的比较已被那个脚本推翻过一次。
  2. **模体不偏袒 TV**：Shepp-Logan 是分片常数的，天然合 TV 先验。故另跑一个
     「TV 敌意模体」= Shepp-Logan + 平滑随机场，破坏分片常数性，看优势是否还在。

【inverse crime 在这里更重，不是更轻】ASD-POCS 与 ART/SIRT/DMR 同属矩阵法，
反演的是与正演完全相同的算子（tests/test_gui.py 锁定 max|A·x − compute_sinogram(x)|
≈ 9.5e-07），而 FBP 不用 A。故本表对矩阵法有系统性偏袒，ASD-POCS 与 FBP 的差值
不可当作真实系统上的可得增益。

确定性（种子固定，与 experiment_c / recon_stopping 同一噪声序列）。
用法：python experiments/recon_tv.py
输出：experiments/results/exp_c_asdpocs.csv
"""
import csv
import os
import sys

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recon_study as rs  # noqa: E402

import recon  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ART_ITERS = (1, 2, 3, 5, 8, 12, 20, 35, 60, 100)
SIRT_ITERS = (20, 50, 100, 200, 400, 800, 1600, 3200)
# ASD-POCS 的轮数网格。**别按 ART=5 / SIRT=100 的习惯取 20**——实测 20 轮时它比
# FBP 还差（60 视角 0.106 vs 0.088）。最优点落在 145 以后，30 视角在 300 仍未触底，
# 故 300 是网格边缘而非收敛点，输出里显式标出 at_grid_edge。
ASD_ITERS = (20, 50, 100, 145, 200, 250, 300)
PHOTON_I0 = (3e4, 3e3, 3e2)          # 对应实测 η ≈ 0.9% / 2.9% / 9.1%


def tv_adversarial_phantom(n, seed=7):
    """TV 敌意模体：Shepp-Logan 叠一层平滑随机场，破坏分片常数性。

    目的是回答「ASD-POCS 赢是不是因为 Shepp-Logan 本来就是 TV 先验的理想对象」。
    随机场经高斯平滑（σ=3）后幅度归一，按 0.18 的权重叠加并 clip 回 [0,1]，
    再乘圆内掩膜——保持与 rs.get_phantom 相同的支撑域，否则圆外的值会污染
    圆内 RMSE 的口径之外还改变正演质量。
    """
    base = rs.get_phantom(n)
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal((n, n)).astype(np.float32), 3.0)
    field = field / (np.abs(field).max() + 1e-9)
    out = np.clip(base + 0.18 * field, 0.0, 1.0).astype(np.float32)
    return out * recon._circle_mask(n)


def _best(fn, iters, A, p, n, gt, cm):
    """在给定轮数网格上取 in-circle RMSE 最优的 (轮数, rmse, 图)。"""
    best = (None, float("inf"), None)
    for k in iters:
        img, _ = fn(A, p, n, k)
        r = float(np.sqrt(np.mean((img[cm] - gt[cm]) ** 2)))
        if r < best[1]:
            best = (k, r, img)
    return best


def sweep(n=64, angle_range=180.0, n_projs=(30, 60, 90), seed=0):
    cm = recon._circle_mask(n) > 0
    phantoms = (("shepp_logan", rs.get_phantom(n)),
                ("tv_adversarial", tv_adversarial_phantom(n)))
    cA, cK = None, None
    rows = []
    for pname, gt in phantoms:
        for i0 in PHOTON_I0:
            # 每个 (模体, 剂量) 重开种子并按 n_projs 顺序消耗，与 recon_stopping.sweep
            # 的消耗顺序一致——这样 (shepp_logan, 3e4) 一档能逐位复现 exp_c_stopping.csv
            # 的 ART*/SIRT*，从而证明本表与已提交产物在同一个噪声实现上。
            rng = np.random.default_rng(seed)
            for npj in n_projs:
                theta = recon.make_theta(angle_range, npj)
                clean = recon.compute_sinogram(gt, theta)
                sino = rs.add_poisson_noise(clean, i0, rng)
                eta = float(np.linalg.norm(sino - clean) / np.linalg.norm(clean))
                p = sino.ravel().astype(np.float32)
                cA, cK = recon.build_system_matrix(n, theta, cA, cK)
                A = cA

                _, fbp = recon.compute_fbp(sino, theta, "ramp")
                r_fbp = float(np.sqrt(np.mean((fbp[cm] - gt[cm]) ** 2)))
                k_art, r_art, _ = _best(recon.compute_art, ART_ITERS, A, p, n, gt, cm)
                k_sirt, r_sirt, im_sirt = _best(recon.compute_sirt, SIRT_ITERS,
                                                A, p, n, gt, cm)
                k_asd, r_asd, im_asd = _best(recon.compute_asdpocs, ASD_ITERS,
                                             A, p, n, gt, cm)

                s_sirt = float(ssim_fn(gt, im_sirt.astype(np.float32), data_range=1.0))
                s_asd = float(ssim_fn(gt, im_asd.astype(np.float32), data_range=1.0))
                rows.append(dict(
                    phantom=pname, photon_i0=f"{i0:.0e}", views=npj,
                    eta=round(eta, 5),
                    rmse_fbp=round(r_fbp, 6),
                    rmse_art_best=round(r_art, 6), art_iters=k_art,
                    rmse_sirt_best=round(r_sirt, 6), sirt_iters=k_sirt,
                    rmse_asdpocs_best=round(r_asd, 6), asdpocs_iters=k_asd,
                    gain_vs_sirt=round((r_sirt - r_asd) / r_sirt, 4),
                    ssim_sirt_best=round(s_sirt, 4),
                    ssim_asdpocs_best=round(s_asd, 4),
                    at_grid_edge=int(k_asd == ASD_ITERS[-1]),
                ))
                print(f"  {pname:<15}{i0:8.0e}{npj:5d}v  η={eta:6.2%} | "
                      f"FBP {r_fbp:.4f}  ART* {r_art:.4f}@{k_art}  "
                      f"SIRT* {r_sirt:.4f}@{k_sirt}  ASD* {r_asd:.4f}@{k_asd} | "
                      f"vs SIRT* {(r_sirt - r_asd) / r_sirt:+6.1%} | "
                      f"SSIM {s_sirt:.3f}→{s_asd:.3f}", flush=True)
    return rows


def main():
    print("ASD-POCS（TV 正则化）基线。oracle-stopped，与 recon_stopping.py 同口径。\n")
    rows = sweep()
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "exp_c_asdpocs.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {dest}（新文件，未覆盖任何既有产物）")
    edge = [r for r in rows if r["at_grid_edge"]]
    if edge:
        print(f"注意：{len(edge)} 行的 ASD-POCS 最优点落在轮数网格边缘 "
              f"({ASD_ITERS[-1]})，即未被 bracket——与 README 对 SIRT@3200 的同类声明一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
