"""研究一补测：ART 与 SIRT 的比较结论取决于**停止准则**，而非算法性质。

动机：实验 C 固定 ART 5 轮、SIRT 100 轮，据此得出「ART 在所有剂量下最鲁棒——其
非负约束与行作用更新起到隐式正则」。这两个轮数是硬编码的，无收敛检查、无停止准则。

等算力口径下，ART 的一个 sweep 与 SIRT 的一次迭代各含约一次正投影 + 一次反投影，
故「ART 5 轮 vs SIRT 100 轮」实际是 1:20 的算力差——给 SIRT 二十倍预算它还输，
这个比较不成立。本脚本扫两者的迭代数，给出各自的最优点。

半收敛（semi-convergence，Hansen & Saxild-Hansen, AIR Tools）指误差先降后升的拐点。
本研究的噪声水平实测仅 η≈0.9%，拐点被推到很远，故在常用轮数区间内看不到——
这是噪声水平的后果，不是算法没有该性质。脚本同时跑一档更低剂量以显示拐点。

确定性（种子固定）。用法：python experiments/recon_stopping.py
输出：experiments/results/exp_c_stopping.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recon_study as rs  # noqa: E402

import recon  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ART_ITERS = (1, 2, 3, 5, 8, 12, 20, 35, 60, 100)
SIRT_ITERS = (20, 50, 100, 200, 400, 800, 1600, 3200)


def _noise_level(sino_clean, sino_noisy):
    """相对噪声水平 η = ‖p_noisy − p_clean‖ / ‖p_clean‖。"""
    return float(np.linalg.norm(sino_noisy - sino_clean) / np.linalg.norm(sino_clean))


def sweep(n=64, angle_range=180.0, n_projs=(30, 60, 90), photon_i0=3e4, seed=0):
    gt = rs.get_phantom(n)
    cm = recon._circle_mask(n) > 0
    rng = np.random.default_rng(seed)     # 与 experiment_c 同一序列
    cA, cK = None, None
    rows = []
    for npj in n_projs:
        theta = recon.make_theta(angle_range, npj)
        clean = recon.compute_sinogram(gt, theta)
        sino = rs.add_poisson_noise(clean, photon_i0, rng)
        eta = _noise_level(clean, sino)
        p = sino.ravel().astype(np.float32)
        cA, cK = recon.build_system_matrix(n, theta, cA, cK)
        A = cA
        print(f"\n[{npj} 视角]  实测噪声水平 η = {eta:.3%}")
        for meth, iters, fn in (("ART", ART_ITERS, recon.compute_art),
                                ("SIRT", SIRT_ITERS, recon.compute_sirt)):
            best = (None, float("inf"))
            for k in iters:
                img, _ = fn(A, p, n, k)
                r = float(np.sqrt(np.mean((img[cm] - gt[cm]) ** 2)))
                rows.append(dict(views=npj, eta=round(eta, 5), method=meth,
                                 iters=k, rmse_in_circle=round(r, 6)))
                if r < best[1]:
                    best = (k, r)
                print(f"  {meth:<5}{k:>6} 轮  RMSE={r:.5f}", flush=True)
            print(f"  → {meth} 在所扫区间内最优：{best[1]:.5f} @ {best[0]} 轮")
    return rows


def main():
    rows = sweep()
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "exp_c_stopping.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {dest}（新文件，未覆盖 exp_c_analytic_vs_iterative.csv）")


if __name__ == "__main__":
    main()
