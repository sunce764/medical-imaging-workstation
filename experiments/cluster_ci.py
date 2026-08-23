"""按病例聚类重算总体 Dice 的 95% 置信区间（研究四的诚实性修正）。

问题：seg3d_teacher.py / seg3d_eval.py 的 `overall_ci` 把所有 (病例, 肺叶) 行摊平成
一个列表后做 i.i.d. 自助重采样。但同一病例的 5 个肺叶不是独立观测——它们共享扫描
质量、spacing、病理与标注者。有效样本量接近病例数而非叶次数，故 i.i.d. 口径给出的
区间**系统性偏窄**。

正确做法是**聚类（按病例）自助法**：以放回方式抽 N 个病例，中签病例带上它全部的
肺叶行，再求均值。理论依据见 Field & Welsh, J. R. Statist. Soc. B 69(3):369-390
(2007)——cluster bootstrap 在 transformation 与 random-effect 两种模型下都相合，
而 residual（此处即 i.i.d.）只在前者下相合。

本脚本**只读已提交的 CSV，不做任何推理**，并把结果写到**新文件**
`results/cluster_ci.json`，不覆盖任何既有产物。两种口径并列输出，便于对账。

用法（须在 dicom_gui 环境内）：python experiments/cluster_ci.py
"""
import csv
import json
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# 四个带 (case, label, dice) 结构的已提交产物
TARGETS = ["seg3d_teacher_dice.csv", "seg3d_student_ch8.csv",
           "seg3d_student_ch8d3_33600s_sliding.csv",
           "seg3d_student_ch8d3_33600s_zslab.csv"]


def _read(path):
    """读 (case, dice) 对；跳过非有限值。CSV 带 BOM，故用 utf-8-sig。"""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                d = float(r["dice"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(d):
                rows.append((r["case"], d))
    return rows


def ci_pooled(rows, n_boot=2000, seed=0):
    """现行口径：把所有叶次摊平后 i.i.d. 重采样（统计上不成立，仅供对照）。"""
    v = np.array([d for _, d in rows], float)
    rng = np.random.RandomState(seed)
    means = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ci_clustered(rows, n_boot=2000, seed=0):
    """正确口径：以病例为单位放回抽样，中签病例带上其全部肺叶行。"""
    by_case = {}
    for c, d in rows:
        by_case.setdefault(c, []).append(d)
    cases = sorted(by_case)
    arrs = [np.array(by_case[c], float) for c in cases]
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(arrs), len(arrs))
        means.append(np.concatenate([arrs[i] for i in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    out = {}
    print(f"{'产物':<44}{'n例':>5}{'n叶次':>7}{'均值':>9}"
          f"{'i.i.d. 区间':>22}{'聚类区间':>22}{'加宽':>7}")
    for name in TARGETS:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            print(f"  跳过（不存在）：{name}")
            continue
        rows = _read(path)
        n_case = len(set(c for c, _ in rows))
        mean = float(np.mean([d for _, d in rows]))
        plo, phi = ci_pooled(rows)
        clo, chi = ci_clustered(rows)
        ratio = (chi - clo) / (phi - plo) if phi > plo else float("nan")
        # 簇数太少时聚类自助法自身退化：只有 k 个簇，重采样至多给出 k^k 种组合，
        # 区间会人为收窄而非加宽（本仓库的 seg3d_student_ch8 只有 2 例，比值 0.27×
        # 就是这个假象）。标注出来，避免把它读成「聚类口径反而更窄」。
        reliable = n_case >= 10
        out[name] = dict(n_cases=n_case, n_lobe_instances=len(rows),
                         mean=round(mean, 4),
                         ci_pooled_iid=[round(plo, 4), round(phi, 4)],
                         ci_case_clustered=[round(clo, 4), round(chi, 4)],
                         width_ratio=round(ratio, 3),
                         clustered_ci_reliable=reliable,
                         note="" if reliable else
                              f"仅 {n_case} 个簇，聚类自助法退化，该区间与比值不可用")
        print(f"{name:<44}{n_case:>5}{len(rows):>7}{mean:>9.4f}"
              f"{f'[{plo:.4f}, {phi:.4f}]':>22}{f'[{clo:.4f}, {chi:.4f}]':>22}"
              f"{ratio:>7.2f}×" + ("" if reliable else"  ← 簇数过少，不可用"))
    dest = os.path.join(RESULTS, "cluster_ci.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写出 {dest}（新文件，未覆盖任何既有产物）")


if __name__ == "__main__":
    main()
