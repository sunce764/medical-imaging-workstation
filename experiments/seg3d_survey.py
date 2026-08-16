# =============================================================================
# 研究四（数据勘察）：统计真值里各器官的覆盖率与体积，据此决定训练哪些类
# ---------------------------------------------------------------------------
# 为什么必须先做这一步：TotalSegmentator-CT-Lite 的每一例覆盖的解剖区域不同——
# 抽查 s0000 是腹盆扫描，肺叶（10–14）一个都不在场。若不先统计就按「训 5 个肺叶」
# 开工，会发现大半病例根本没有正样本，训出来的东西毫无意义。
#
# 标签体系：真值用 TotalSegmentator v2 "total"（117 类）编号，其 **1–21 与官方
# class_map_part_organs 完全一致**（见 experiments/seg_validate.py 的映射表与
# results/seg_mapping.md 的实测确证），故本研究的候选目标限定在 1–21。
# 22 以上是椎骨/肋骨等结构，不在本项目 organs.onnx 的 25 类范围内。
#
# 用法：python experiments/seg3d_survey.py
# 产出：results/seg3d_survey.csv（每类的覆盖率、平均体素占比、体积）
# =============================================================================

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
CACHE = os.path.join(HERE, ".seg3d_cache")
RESULTS = os.path.join(HERE, "results")
MANIFEST = os.path.join(RESULTS, "seg3d_manifest.json")

# 真值 1–21 的器官名（与 class_map_part_organs 一致，已由 seg_validate 实测确证）
ORGANS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
    16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
    20: "colon", 21: "urinary_bladder",
}


def survey():
    import nibabel as nib
    man = json.load(open(MANIFEST))
    cases = sorted(r['case'] for r in man['cases'])
    print(f"  勘察 {len(cases)} 例（{man['repo']}@{man['revision'][:8]}）")
    present = {k: 0 for k in ORGANS}          # 出现的病例数
    vox = {k: [] for k in ORGANS}             # 出现时的体素数
    shapes, spacings = [], []
    for i, cid in enumerate(cases, 1):
        pm = os.path.join(CACHE, f"{cid}_msk.nii.gz")
        if not os.path.exists(pm):
            continue
        nm = nib.load(pm)
        m = np.asarray(nm.dataobj).astype(np.int32)
        shapes.append(m.shape)
        spacings.append(tuple(round(float(x), 3) for x in nm.header.get_zooms()[:3]))
        ids, cnts = np.unique(m[m > 0], return_counts=True)
        # strict=True：np.unique 保证两者等长，长度不等意味着上游出了问题，应当立刻暴露
        for lid, c in zip(ids.tolist(), cnts.tolist(), strict=True):
            if lid in present:
                present[lid] += 1
                vox[lid].append(c)
        if i % 20 == 0:
            print(f"    {i}/{len(cases)}"); sys.stdout.flush()

    n = len(shapes)
    print(f"\n  体素间距: {len(set(spacings))} 种 → {sorted(set(spacings))[:3]}")
    hs = np.array(shapes)
    print(f"  形状范围: {hs.min(axis=0).tolist()} ~ {hs.max(axis=0).tolist()}（各例不同，需重采样/裁剪）")
    print(f"\n  {'标签':>4} {'器官':<24}{'覆盖率':>9}{'平均体素':>11}{'中位体素':>11}")
    rows = []
    for lid, name in ORGANS.items():
        cov = present[lid] / n * 100 if n else 0
        mean_v = float(np.mean(vox[lid])) if vox[lid] else 0.0
        med_v = float(np.median(vox[lid])) if vox[lid] else 0.0
        rows.append(dict(label=lid, organ=name, coverage_pct=round(cov, 1),
                         n_cases=present[lid], mean_voxels=round(mean_v),
                         median_voxels=round(med_v)))
        print(f"  {lid:>4} {name:<24}{cov:>8.1f}%{mean_v:>11,.0f}{med_v:>11,.0f}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "seg3d_survey.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("    → results/seg3d_survey.csv")

    # 选类建议：覆盖率高且体积够大的才适合做训练目标。
    # 覆盖率低 = 大半病例没有正样本；体积过小 = 单类 Dice 方差极大，n=20 的测试集
    # 根本区分不出模型差异（这正是 MVI 项目里 n=23 吃过的亏）。
    good = [r for r in rows if r['coverage_pct'] >= 60 and r['median_voxels'] >= 3000]
    print(f"\n  建议目标（覆盖率≥60% 且 中位体素≥3000）：{len(good)} 类")
    for r in sorted(good, key=lambda x: -x['coverage_pct']):
        print(f"    {r['label']:>3} {r['organ']:<24} 覆盖 {r['coverage_pct']:>5.1f}%  "
              f"中位 {r['median_voxels']:>8,} 体素")
    weak = [r for r in rows if 0 < r['coverage_pct'] < 60]
    if weak:
        # 不在 f-string 里嵌套引号：Py3.12 才允许，本项目锁 3.10
        txt = ", ".join("{}:{}({}%)".format(r['label'], r['organ'], r['coverage_pct'])
                        for r in weak[:6])
        print(f"  覆盖不足而排除：{txt}")
    return rows


if __name__ == "__main__":
    survey()
