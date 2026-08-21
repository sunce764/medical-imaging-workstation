# =============================================================================
# 研究四（汇总）：把教师与各容量学生的结果并成一张权衡表与一张曲线
# ---------------------------------------------------------------------------
# 回答的问题：把 31.2M 参数的 organs.onnx 压小，Dice 掉多少？这个交换值不值？
#
# 【读这张表时必须记住的三件事，已写进图注】
#   1. 教师是 25 类通用模型，学生只做 5 个肺叶——**任务不同**。学生参数少，一部分
#      来自模型小，一部分来自任务窄。表里同时给出这一点，不把它算成压缩的功劳。
#   2. 教师在 organs.onnx 的训练数据上训练（TotalSegmentator 全量），学生只用了
#      207 例。数据量差距同样会体现在 Dice 上。
#   3. 时间与内存两边口径已逐项对齐（同 CPU、同 z 分块、同测法），只有这一栏是
#      纯粹的模型差异。
#
# 用法：python experiments/seg3d_report.py
# 产出：results/seg3d_tradeoff.csv + seg3d_tradeoff.png
# =============================================================================

import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TEACHER_PARAMS = 31.2e6          # organs.onnx 权重元素数（seg3d_bench 实测）


def collect():
    tp = os.path.join(RESULTS, "seg3d_teacher_summary.json")
    if not os.path.exists(tp):
        print("  缺教师基线，请先跑 seg3d_teacher.py"); return None, []
    teacher = json.load(open(tp))
    students, skipped, unverified = [], [], []
    n_ref = teacher.get('n_cases', 0)
    for p in sorted(glob.glob(os.path.join(RESULTS, "seg3d_student_ch*.json"))):
        d = json.load(open(p))
        # 冒烟/中断产物必须挡在报告之外：目录里躺过一份 n_cases=2、best_ep=1、
        # Dice=0.031 的一轮冒烟结果，glob 会把它与教师的 57 例并排画进权衡曲线，
        # 看图的人无从分辨。判据取「测试集是否与教师同规模」——不同测试集本就
        # 不可比，逐例配对检验更要求同一批病例。跳过的必须打印出来，
        # 静默丢弃只是换一种不诚实。
        if d.get('n_cases', 0) < n_ref:
            skipped.append((os.path.basename(p), f"{d.get('n_cases', 0)} 例 < 教师 {n_ref}"))
            continue
        # 规模相同仍可能是**不同的病例**——配对比较要求同一批。两边都落了清单才验得了；
        # 旧产物没有 cases 键，此时只能退回按规模把关，并明示这一点，不假装验过。
        tc, sc = teacher.get('cases'), d.get('cases')
        if tc and sc:
            if set(tc) != set(sc):
                only_s, only_t = set(sc) - set(tc), set(tc) - set(sc)
                skipped.append((os.path.basename(p),
                                f"病例集合不同（学生独有 {len(only_s)}，教师独有 {len(only_t)}）"))
                continue
        else:
            unverified.append(os.path.basename(p))
        # 【口径必须一致，否则权衡曲线在比推理策略而不是比模型】
        # 教师基线由 seg3d_teacher.run_onnx 产出，走整幅 xy 沿 z 分块（zslab）。
        # 学生若用训练同尺寸滑窗（sliding）评，同一份权重能高出 0.25 Dice——
        # 把两者并排画进「参数量 vs 精度」，读者会以为那是两个不同的模型。
        # 判据优先取产物自带的 infer 字段；早于该字段的产物从文件名推断；
        # 都取不到时视为 zslab——因为在 --infer 开关引入之前，这条路径是唯一的。
        name = os.path.basename(p)
        mode = d.get('infer')
        if mode is None:
            mode = ('sliding' if '_sliding' in name else
                    'zslab' if '_zslab' in name else 'zslab（推断：早于 --infer 开关）')
        if not str(mode).startswith('zslab'):
            skipped.append((name, f"推理口径为 {mode}，与教师的 zslab 不可比"))
            continue
        students.append(d)
    students.sort(key=lambda d: d['params'])
    return teacher, students, skipped, unverified


def _read_dice_csv(path):
    """读逐例逐器官 Dice，返回 {(case, label): dice}。"""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                out[(row['case'], int(row['label']))] = float(row['dice'])
            except (KeyError, ValueError):
                continue
    return out


def paired_test(tag, n_boot=5000, seed=0):
    """学生 vs 教师的**逐例配对**检验。

    【为什么必须配对，以及初版错在哪】
    初版的判据是「两条 bootstrap CI 是否重叠」，两处都错：
      1. **CI 重叠推不出「无显著差异」**——这是统计学的经典误解。两个 95% CI 重叠
         时，配对差值的 CI 仍可能完全不含 0。用它下「未发现显著差异」的结论，对
         研究四的头号结论是**假阴性**。
      2. 教师与学生跑的是**同一批测试病例**（患者级划分、seed 固定），属配对设计。
         把它们当成两个独立样本比较，白白丢掉配对带来的方差削减——病例难易度的
         个体差异本可以被差分消掉。
    正确做法：按 (病例, 器官) 配对求差值，对**差值**做 bootstrap CI 与 Wilcoxon
    signed-rank（后者不假设正态，适合 Dice 这种有界且偏斜的量）。
    差值 CI 不含 0 才谈得上差异显著。
    """
    import numpy as np
    ta = _read_dice_csv(os.path.join(RESULTS, "seg3d_teacher_dice.csv"))
    st = _read_dice_csv(os.path.join(RESULTS, f"seg3d_student_{tag}.csv"))
    if not ta or not st:
        return None
    keys = sorted(set(ta) & set(st))
    d = np.array([st[k] - ta[k] for k in keys], float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return None
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p = None
    if len(d) >= 10 and np.any(d != 0):
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d, zero_method='wilcox').pvalue)
        except Exception:
            p = None
    if lo > 0:
        verdict = "差值 CI 完全在 0 以上 → 学生显著优于教师"
    elif hi < 0:
        verdict = "差值 CI 完全在 0 以下 → 学生显著劣于教师"
    else:
        verdict = ("差值 CI 跨 0 → 未能证明存在差异（注意：这是「证据不足」，"
                   "不是「已证明相同」）")
    # n<10 时 Wilcoxon 不可靠，如实说明而不是给个假装有效的 p
    if len(d) < 10:
        verdict += f"；n={len(d)} 对偏少，检验功效不足"
    return dict(n=len(d), mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi,
                wilcoxon_p=p, verdict=verdict)


def main():
    teacher, students, skipped, unverified = collect()
    if teacher is None:
        return 1
    for name, why in skipped:
        print(f"  ⊘ 已排除 {name}：{why} —— 测试集不同则无从比较")
    # 未能核验「同一批病例」的，必须说出来：把关退化成只比规模时，读者有权知道
    for name in unverified:
        print(f"  ⚠ {name}：产物未记录病例清单，仅按测试集规模把关，"
              f"未能核验是否与教师同一批（重跑 seg3d_teacher/seg3d_eval 即可补上）")
    if not students:
        print("  尚无学生模型结果，请先跑 seg3d_train.py + seg3d_eval.py"); return 1

    rows = [dict(model="organs.onnx (teacher)", classes=25, params_m=round(TEACHER_PARAMS / 1e6, 2),
                 dice=round(teacher['overall_mean'], 4),
                 ci_lo=round(teacher['overall_ci'][0], 4), ci_hi=round(teacher['overall_ci'][1], 4),
                 sec_per_case=round(teacher['infer_sec_mean'], 1),
                 peak_gb=round(teacher.get('peak_gb_max', teacher['peak_gb_mean']), 2), n_cases=teacher['n_cases'])]
    for s in students:
        rows.append(dict(model=f"UNet3D ch={s['ch']} (student)", classes=6,
                         params_m=round(s['params'] / 1e6, 4),
                         dice=round(s['overall_mean'], 4),
                         ci_lo=round(s['overall_ci'][0], 4), ci_hi=round(s['overall_ci'][1], 4),
                         sec_per_case=round(s['infer_sec_mean'], 1),
                         peak_gb=round(s.get('peak_gb_max', s['peak_gb_mean']), 2), n_cases=s['n_cases']))

    print(f"  {'模型':<26}{'类数':>5}{'参数M':>10}{'Dice':>9}{'95%CI':>18}"
          f"{'秒/例':>8}{'峰值GB':>9}")
    for r in rows:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"  {r['model']:<26}{r['classes']:>5}{r['params_m']:>10.4f}{r['dice']:>9.4f}"
              f"{ci:>18}{r['sec_per_case']:>8.1f}{r['peak_gb']:>9.2f}")

    t = rows[0]
    print("\n  === 相对教师（逐例配对检验）===")
    for s in students:
        r = next(x for x in rows[1:] if x['params_m'] == round(s['params'] / 1e6, 4))
        pr = paired_test(f"ch{s['ch']}")
        print(f"  {r['model']:<26} 参数 1:{t['params_m']/r['params_m']:>6.0f}   "
              f"提速 {t['sec_per_case']/r['sec_per_case']:.2f}×   "
              f"内存 {r['peak_gb']/t['peak_gb']:.2f}×")
        if pr is None:
            print("      逐例数据缺失，无法做配对检验")
            continue
        print(f"      配对差值（学生 − 教师）: {pr['mean_diff']:+.4f}  "
              f"95%CI [{pr['ci_lo']:+.4f}, {pr['ci_hi']:+.4f}]  n={pr['n']} 对")
        print(f"      {pr['verdict']}")
        if pr['wilcoxon_p'] is not None:
            print(f"      Wilcoxon signed-rank p = {pr['wilcoxon_p']:.4g}")

    with open(os.path.join(RESULTS, "seg3d_tradeoff.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # 权衡曲线：横轴参数量（对数），纵轴 Dice，误差棒为 bootstrap CI
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    px = [r['params_m'] for r in rows]
    dy = [r['dice'] for r in rows]
    lo = [r['dice'] - r['ci_lo'] for r in rows]
    hi = [r['ci_hi'] - r['dice'] for r in rows]
    axes[0].errorbar(px[1:], dy[1:], yerr=[lo[1:], hi[1:]], fmt='o-', capsize=4,
                     lw=2, label='UNet3D (student, 5 lobes)')
    axes[0].errorbar(px[:1], dy[:1], yerr=[lo[:1], hi[:1]], fmt='s', capsize=4,
                     ms=10, color='tab:red', label='organs.onnx (teacher, 25 classes)')
    axes[0].set_xscale('log'); axes[0].set_xlabel('parameters (millions, log scale)')
    axes[0].set_ylabel('lung-lobe Dice'); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
    axes[0].set_title('Accuracy vs model size (bars = bootstrap 95% CI)')

    axes[1].plot([r['sec_per_case'] for r in rows[1:]], dy[1:], 'o-', lw=2, label='student')
    axes[1].plot([rows[0]['sec_per_case']], dy[:1], 's', ms=10, color='tab:red', label='teacher')
    axes[1].set_xlabel('inference seconds per case (CPU, same z-slab strategy)')
    axes[1].set_ylabel('lung-lobe Dice'); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    axes[1].set_title('Accuracy vs inference cost')
    fig.suptitle('Study IV: how much accuracy does shrinking the model cost?', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "seg3d_tradeoff.png"), dpi=140)
    plt.close(fig)
    print("\n    → results/seg3d_tradeoff.csv + seg3d_tradeoff.png")

    print("\n  【解读这张表时不可省略的三点】")
    print("  1. 教师 25 类通用 vs 学生 5 类肺叶——任务不同，参数差里有一部分是任务窄")
    print("  2. 教师用 TotalSegmentator 全量训练，学生只用 207 例——数据量差距同样计入 Dice")
    print("  3. 只有时间与内存两栏是纯模型差异：同 CPU、同 z 分块、同测法，逐项对齐过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
