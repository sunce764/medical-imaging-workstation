"""研究一补测：直接测量系统矩阵 A 的条件数，检验 DMR 在 60 视角处的 RMSE 尖峰。

动机：preprint 与技术报告一直用「近方阵处条件数最差」解释 60 视角的 DMR 失稳，
但这条解释此前**从未被测量**——只是引自教科书的合理推断。本脚本直接算 σ_max/σ_min，
让该解释要么变成实测结论，要么被推翻。

与 recon_study.experiment_c 严格同源：调用同一个 recon.make_theta 与
recon.build_system_matrix，故 A 与研究一 §4.3 用的是同一个矩阵（走磁盘缓存）。

用法（须在 dicom_gui 环境内）：python experiments/recon_cond.py
输出：experiments/results/exp_c_conditioning.csv
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def measure(n=64, angle_range=180.0, n_projs=(30, 60, 90)):
    rows = []
    cached_A, cached_key = None, None
    for npj in n_projs:
        theta = recon.make_theta(angle_range, npj)
        cached_A, cached_key = recon.build_system_matrix(n, theta, cached_A, cached_key)
        A = np.asarray(cached_A, dtype=np.float64)   # SVD 用双精度，避免 float32 把小奇异值淹掉
        m_rows, n_cols = A.shape
        t0 = time.time()
        sv = np.linalg.svd(A, compute_uv=False)
        dt = time.time() - t0
        smax, smin = float(sv[0]), float(sv[-1])

        # —— 保留的秩：直接问 compute_dmr 用的那个函数，不自己复刻截断规则 ——
        # 早先这里手写了「阈 = max(m,n)·eps_float32·σ_max，因为 A 是 float32」。
        # 那是错的：numpy 的 lstsq 在 _commonType 里【无条件】把输入升到 double
        # （numpy/linalg/_linalg.py 的 _commonType 注释 "always double or cdouble"），
        # rcond = finfo(t).eps * max(n, m) 里的 t 恒为 float64。手写规则一旦与被测
        # 函数的真实行为不符，就会算出一组看似合理、实则错误的数——本文件曾如此。
        # 现在改为直接取 lstsq 返回的 rank，让它与 compute_dmr 不可能漂移。
        probe = np.zeros(m_rows, dtype=cached_A.dtype)
        _, _, lapack_rank, _ = np.linalg.lstsq(cached_A, probe, rcond=None)
        k = int(lapack_rank)
        sigma_k = float(sv[k - 1]) if k > 0 else float("nan")

        # cond₂ 只在满秩时才有意义。30/60 视角的 σ_min 低于 eps·σ_max·max(m,n)，
        # 即数值上就是零，此时 σ_max/σ_min 是舍入噪声而非矩阵性质——实测行置换
        # 就能让 30 与 60 的大小关系反转，故这两档记为 inf 而不报一个假精度的数。
        tol = max(m_rows, n_cols) * np.finfo(np.float64).eps * smax
        rank = int((sv > tol).sum())
        cond = smax / smin if smin > tol else float("inf")
        rows.append({
            "views": npj,
            "n_equations": m_rows,
            "n_unknowns": n_cols,
            "eq_over_unk": round(m_rows / n_cols, 4),
            "sigma_max": smax,
            "sigma_min": smin,
            "cond_2norm": cond,
            "cond_meaningful": bool(smin > tol),
            "numerical_rank": rank,
            "rank_deficient": rank < n_cols,
            "k_retained_by_lstsq": k,
            "sigma_k_retained": sigma_k,
            "noise_gain_pinv": 1.0 / sigma_k,
            "discarded_dims": n_cols - k,
            "svd_seconds": round(dt, 1),
        })
        print(f"[cond] views={npj:3d}  A={m_rows}x{n_cols}  "
              f"cond={cond:.4g} rank={rank}/{n_cols}  |  "
              f"lstsq 保留 {k} 向  σ_k={sigma_k:.4g}  噪声增益 1/σ_k={1.0/sigma_k:.4g}  "
              f"丢弃 {n_cols - k} 维  ({dt:.1f}s)")
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = measure()
    out = os.path.join(RESULTS, "exp_c_conditioning.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[cond] 已写出 {out}")


if __name__ == "__main__":
    main()
