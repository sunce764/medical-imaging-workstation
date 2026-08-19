# =============================================================================
# 研究四（教师基线）：organs.onnx 在多例测试集上的真实 Dice
# ---------------------------------------------------------------------------
# 两个目的，一次跑完：
#   1. 给自训练的学生模型一个诚实的对标值（参数量/推理时间/峰值内存/Dice 四维权衡）
#   2. 把 seg_validate.py 的 n=1 补成 n=测试集全量，并给 bootstrap 置信区间
#
# 【为什么必须重测，而不是沿用 0.92】
#   现有的 Dice≈0.92 出自**单例**。单例既无法给区间，也很可能偏乐观——正好落在
#   一例边界清晰、器官俱全的扫描上。本项目自己的 README 就把「n=1」列为已知限制。
#   重测很可能得到低于 0.92 的数字，那才是真实值。
#
# 推理复刻自 ai_engine._run_onnx_multiorgan（与 seg_validate.py 同一套：
# clip[-1000,400] 归一化 + 沿 z 的 DZ=32 滑窗 + 输出取 argmax），故测的是产品行为。
#
# 用法：python experiments/seg3d_teacher.py [--split test] [--limit N]
# 产出：results/seg3d_teacher_dice.csv（逐例逐器官）+ 汇总打印
# =============================================================================

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from constants import MODEL_PATH  # noqa: E402

CACHE = os.path.join(HERE, ".seg3d_cache")
RESULTS = os.path.join(HERE, "results")

# 本研究的目标器官：5 个肺叶。真值标签 10–14 与 organs.onnx 的输出标签一致
# （TotalSegmentator 117 类的 1–21 段等同 class_map_part_organs，已由 seg_mapping.md 实测确证）
LUNG_LOBES = {10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
              12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right",
              14: "lung_lower_lobe_right"}


def load_zhw(path):
    """载入 nii，规范到 RAS，再转成 GUI 的 (Z=上下, H=前后, W=左右) 轴序。

    轴序必须与 GUI 一致，否则推理出的标签虽然「形状对」但解剖全错——
    本项目在别处已经栽过一次「看形状对、语义全错」。
    """
    import nibabel as nib
    v = nib.as_closest_canonical(nib.load(path))
    return np.transpose(np.asanyarray(v.dataobj), (2, 1, 0))


def make_session():
    """建一次 InferenceSession，供逐例推理复用。

    【为什么必须建在循环外】原实现把建图放在 run_onnx 内部，57 例就建 57 次图，
    建图开销全部计进「教师推理秒数」；而学生侧（seg3d_eval）的模型只在循环外建
    一次。两个数并排放进权衡表，比的是「教师含建图 vs 学生不含」，系统性地偏向
    学生。seg3d_bench 早就用预热排除了这项开销，此处补齐同一口径。
    """
    import onnxruntime as ort
    so = ort.SessionOptions(); so.enable_cpu_mem_arena = False
    return ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])


def run_onnx(volume_hu, sess=None, quiet=False):
    """复刻 ai_engine._run_onnx_multiorgan 的预处理与滑窗，返回 (标签图, 峰值内存GB)。

    sess 为 None 时自建（保持单次调用可用），批量评估务必从外部传入复用的 session。
    """
    import resource
    norm = np.clip(volume_hu, -1000, 400).astype(np.float32)
    norm = (norm + 1000.0) / 1400.0
    Z, H, W = norm.shape
    if sess is None:
        sess = make_session()
    iname = sess.get_inputs()[0].name
    ph, pw = (-H) % 32, (-W) % 32
    seg = np.zeros((Z, H, W), dtype=np.uint8)
    for z0 in range(0, Z, 32):
        z1 = min(z0 + 32, Z)
        blk = norm[z0:z1]
        pd = (-blk.shape[0]) % 32
        if pd or ph or pw:
            blk = np.pad(blk, ((0, pd), (0, ph), (0, pw)), mode="constant")
        out = sess.run(None, {iname: blk[np.newaxis, np.newaxis]})[0][0]
        seg[z0:z1] = out.argmax(0).astype(np.uint8)[:z1 - z0, :H, :W]
        del out
        if not quiet:
            print(f"      z {z1}/{Z}", end="\r")
    # ru_maxrss 的单位随平台而异：macOS 是**字节**，Linux 是 KB。写死一种会让数字
    # 差 1024 倍，而「4124.9 GB」这种荒谬值反倒容易发现，真正危险的是差 1024 倍后
    # 仍看似合理的情形。故按平台分别换算，统一返回 GB。
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
    return seg, peak_gb


def dice(a, b):
    inter = int(np.logical_and(a, b).sum())
    s = int(a.sum() + b.sum())
    return (2.0 * inter / s) if s else float('nan')     # 两者皆空 → 无定义，不算 0


def bootstrap_ci(vals, n_boot=2000, seed=0):
    """逐例 Dice 的 bootstrap 95% CI。n=1 时无法给区间——这正是要补掉的弱点。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 2:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    means = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(MODEL_PATH):
        print(f"  缺少 {MODEL_PATH}（含外部权重 organs.onnx.data）"); return 1

    sys.path.insert(0, HERE)
    from seg3d_data import split as make_split
    cases = make_split()[a.split]
    if a.limit:
        cases = cases[:a.limit]
    print(f"\n  教师基线：organs.onnx 在 {a.split} 集 {len(cases)} 例上的逐例 Dice")
    print(f"  目标：5 个肺叶（真值标签 {sorted(LUNG_LOBES)}）\n")

    # 【ru_maxrss 的语义陷阱】它是**进程生命周期内**的峰值，单调不减：逐例 append
    # 得到的是非递减序列，对它取 mean 既不是「每例峰值」也不是「全程峰值」，
    # 只是一条爬升曲线的平均高度，没有可解释的含义。真正有意义的是最大值＝全程峰值。
    # mean 仍保留，仅为兼容此前已产出的 JSON（那些数字就是这么来的，不重跑、不改写）。
    sess = make_session()          # 建一次复用：把建图开销排除在逐例计时之外
    rows, times, peaks = [], [], []
    for i, cid in enumerate(cases, 1):
        img = load_zhw(os.path.join(CACHE, f"{cid}_img.nii.gz"))
        gt = load_zhw(os.path.join(CACHE, f"{cid}_msk.nii.gz")).astype(np.int32)
        present = [lid for lid in LUNG_LOBES if (gt == lid).any()]
        if not present:
            print(f"  [{i}/{len(cases)}] {cid}: 真值无肺叶，跳过")
            continue
        t0 = time.perf_counter()
        seg, peak = run_onnx(img, sess=sess, quiet=True)
        dt = time.perf_counter() - t0
        times.append(dt); peaks.append(peak)
        ds = {}
        for lid in present:
            d = dice(seg == lid, gt == lid)
            ds[lid] = d
            rows.append(dict(case=cid, label=lid, organ=LUNG_LOBES[lid], dice=round(d, 4)))
        md = float(np.nanmean(list(ds.values())))
        print(f"  [{i}/{len(cases)}] {cid}  shape={img.shape}  {dt:.0f}s  "
              f"在场 {len(present)}/5 叶  平均 Dice={md:.3f}")
        sys.stdout.flush()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "seg3d_teacher_dice.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'label', 'organ', 'dice'])
        w.writeheader(); w.writerows(rows)

    print(f"\n  ===== 汇总（{len(set(r['case'] for r in rows))} 例）=====")
    print(f"  {'标签':>4} {'器官':<24}{'n':>4}{'平均Dice':>10}{'95%CI':>18}")
    summary = []
    for lid, name in LUNG_LOBES.items():
        vals = [r['dice'] for r in rows if r['label'] == lid]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        summary.append(dict(label=lid, organ=name, n=len(vals),
                            mean_dice=round(float(np.mean(vals)), 4),
                            ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        print(f"  {lid:>4} {name:<24}{len(vals):>4}{np.mean(vals):>10.4f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")
    allv = [r['dice'] for r in rows]
    lo, hi = bootstrap_ci(allv)
    print(f"\n  五叶总体：平均 Dice = {np.mean(allv):.4f}，95% CI [{lo:.4f}, {hi:.4f}]（n={len(allv)} 叶次）")
    print(f"  推理耗时：{np.mean(times):.0f} ± {np.std(times):.0f} s/例（CPU）")
    print(f"  峰值内存：全程 {max(peaks):.1f} GB"
          f"（逐例 ru_maxrss 均值 {np.mean(peaks):.1f} GB，仅兼容旧产物，勿作每例峰值解读）")
    with open(os.path.join(RESULTS, "seg3d_teacher_summary.json"), 'w') as f:
        json.dump({'per_organ': summary, 'overall_mean': float(np.mean(allv)),
                   'overall_ci': [lo, hi], 'n_lobe_instances': len(allv),
                   'n_cases': len(set(r['case'] for r in rows)),
                   # 落盘病例清单：下游做教师-学生对比时，「同规模」不等于「同一批」，
                   # 而逐例配对比较要求后者。只有把清单写出来，把关才验得了。
                   'cases': sorted(set(r['case'] for r in rows)),
                   'infer_sec_mean': float(np.mean(times)),
                   'peak_gb_max': float(max(peaks)),
                   'peak_gb_mean': float(np.mean(peaks))}, f, indent=1)
    print("    → results/seg3d_teacher_dice.csv + seg3d_teacher_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
