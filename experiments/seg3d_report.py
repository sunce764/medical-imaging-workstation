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
    students = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "seg3d_student_ch*.json"))):
        students.append(json.load(open(p)))
    students.sort(key=lambda d: d['params'])
    return teacher, students


def main():
    teacher, students = collect()
    if teacher is None:
        return 1
    if not students:
        print("  尚无学生模型结果，请先跑 seg3d_train.py + seg3d_eval.py"); return 1

    rows = [dict(model="organs.onnx (teacher)", classes=25, params_m=round(TEACHER_PARAMS / 1e6, 2),
                 dice=round(teacher['overall_mean'], 4),
                 ci_lo=round(teacher['overall_ci'][0], 4), ci_hi=round(teacher['overall_ci'][1], 4),
                 sec_per_case=round(teacher['infer_sec_mean'], 1),
                 peak_gb=round(teacher['peak_gb_mean'], 2), n_cases=teacher['n_cases'])]
    for s in students:
        rows.append(dict(model=f"UNet3D ch={s['ch']} (student)", classes=6,
                         params_m=round(s['params'] / 1e6, 4),
                         dice=round(s['overall_mean'], 4),
                         ci_lo=round(s['overall_ci'][0], 4), ci_hi=round(s['overall_ci'][1], 4),
                         sec_per_case=round(s['infer_sec_mean'], 1),
                         peak_gb=round(s['peak_gb_mean'], 2), n_cases=s['n_cases']))

    print(f"  {'模型':<26}{'类数':>5}{'参数M':>10}{'Dice':>9}{'95%CI':>18}"
          f"{'秒/例':>8}{'峰值GB':>9}")
    for r in rows:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"  {r['model']:<26}{r['classes']:>5}{r['params_m']:>10.4f}{r['dice']:>9.4f}"
              f"{ci:>18}{r['sec_per_case']:>8.1f}{r['peak_gb']:>9.2f}")

    t = rows[0]
    print("\n  === 相对教师 ===")
    for r in rows[1:]:
        d_dice = r['dice'] - t['dice']
        # CI 是否重叠：不重叠才谈得上「差异显著」，否则只能说「未发现差异」
        overlap = not (r['ci_hi'] < t['ci_lo'] or t['ci_hi'] < r['ci_lo'])
        print(f"  {r['model']:<26} 参数 1:{t['params_m']/r['params_m']:>6.0f}   "
              f"Dice {d_dice:+.4f}   提速 {t['sec_per_case']/r['sec_per_case']:.2f}×   "
              f"内存 {r['peak_gb']/t['peak_gb']:.2f}×   "
              f"{'CI 重叠（未发现显著差异）' if overlap else 'CI 不重叠（差异显著）'}")

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
