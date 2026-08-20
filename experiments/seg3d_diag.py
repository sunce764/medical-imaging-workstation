# =============================================================================
# 研究四（失败模式定位）：轻量学生模型学会了「是不是肺」，没学会「是哪个叶」
# ---------------------------------------------------------------------------
# 【这个脚本回答什么】自训练的学生模型五叶 Dice 极低（~0.13）。低的原因有三种，
#       后续该怎么做完全取决于是哪一种，所以必须先分开：
#         ① 几乎全预测背景        → 类别不平衡/损失配置问题，再训也白搭
#         ② 分得出肺、分不出叶    → 任务需要全局解剖位置，patch 视野给不了
#         ③ 位置对、边界糊        → 纯训练不足，加时间就有用
#       做法：把预测按「肺 vs 背景」和「五叶之间」两个口径分别算 Dice。若前者高、
#       后者低，就是 ②——而 ② 与 ①③ 的补救方向完全相反。
#
# 【为什么不用 val patch-Dice 判断】训练时报的那个数是 patch 级、且一半 patch 围绕
#       肺叶中心采（前景过采样），既被偏置又噪声大（12×bs2=24 个 patch 估 5 类）。
#       此处走整卷 zslab_infer，与教师同一分块口径。
#
# 【为什么用 val 而不是 test】test 集留给最终评估，诊断阶段反复看它会污染独立性。
#
# 用法：python experiments/seg3d_diag.py --ckpt results/seg3d_w8d3.pt
#       python experiments/seg3d_diag.py plot ch8d3      由已有 CSV 重绘，不重跑推理
#       python experiments/seg3d_diag.py rf              实测各 depth 的有效感受野
#       python experiments/seg3d_diag.py extent          实测真值里各肺叶的 z 跨度
# 产出：results/seg3d_diag_{tag}.csv（逐例逐类）+ .json（汇总）+ .png（两联图）
# =============================================================================

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")

NAMES = {1: 'upper_L', 2: 'lower_L', 3: 'upper_R', 4: 'middle_R', 5: 'lower_R'}


def _scale_bars():
    """尺度条的数字一律从 results/seg3d_geom.json 实测读取，**不写死**。

    此前这里硬编码 [42, 78, 192, 768]，其中 768 还是照 512² 想当然算的（实测中位
    380mm）。硬编码的图会在结构或数据一变时继续画出旧数字——这正是 model_card
    那轮已经栽过一次的坑，那次的结论就是「改成从产物实时读」。
    """
    gp = os.path.join(RESULTS, "seg3d_geom.json")
    if not os.path.exists(gp):
        return None
    g = json.load(open(gp))
    bars = [(f"student depth={d}\n({v[1]:.0f}mm ERF, xy)", v[1])
            for d, v in sorted(g['erf_mm'].items())]
    bars.append((f"student patch\n({g['patch_mm']:.0f}mm)", g['patch_mm']))
    bars.append((f"teacher full xy\n(median {g['teacher_fov_mm_median']:.0f}mm)",
                 g['teacher_fov_mm_median']))
    return bars, g


def plot(tag):
    """由已有 CSV/JSON 重绘，不重跑推理。"""
    cp = os.path.join(RESULTS, f"seg3d_diag_{tag}.csv")
    jp = os.path.join(RESULTS, f"seg3d_diag_{tag}.json")
    if not (os.path.exists(cp) and os.path.exists(jp)):
        print(f"  缺产物 seg3d_diag_{tag}.csv/.json，先跑 seg3d_diag.py --ckpt <权重>")
        return 1
    meta = json.load(open(jp))
    with open(cp, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))

    # 左：视野尺度。用对数横轴，否则 42 与 768 挤在一起看不出量级差
    sb = _scale_bars()
    if sb is None:
        print("  缺 results/seg3d_geom.json，先跑 seg3d_diag.py rf"); return 1
    bars, g = sb
    labs = [t for t, _ in bars]
    vals = [v for _, v in bars]
    cols = ['#3B7DD8'] * (len(bars) - 2) + ['#9DBEEA', '#1B7F4B']
    ax[0].barh(range(len(vals)), vals, color=cols, height=.6)
    for i, v in enumerate(vals):
        ax[0].text(v * 1.06, i, f"{v} mm", va='center', fontsize=9)
    # 参考线：肺叶实测 z 跨度中位数 18–108mm（见正文），区分肺叶需覆盖这个量级
    ax[0].plot([g['teacher_fov_mm_min'], g['teacher_fov_mm_max']],
               [len(vals) - 1] * 2, color='#1B7F4B', lw=1.2, alpha=.6,
               marker='|', ms=8)   # 教师视野逐例差异大，标出全距而非只给中位
    ax[0].axvline(108, color='#C0392B', ls='--', lw=1.3,
                  label='largest lobe extent (108 mm, measured)')
    ax[0].set_yticks(range(len(vals))); ax[0].set_yticklabels(labs, fontsize=8.5)
    ax[0].set_xscale('log'); ax[0].set_xlim(20, 1600)
    ax[0].set_xlabel('field of view / effective receptive field  (mm, log scale)')
    ax[0].set_title('What the model can see', fontsize=11)
    ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(axis='x', alpha=.3)

    # 右：逐叶「在场却零预测」的比例
    nv = meta['never_predicted']
    names = list(nv)
    frac = [nv[k][0] / nv[k][1] if nv[k][1] else 0 for k in names]
    cols2 = ['#C0392B' if f > 0.5 else '#3B7DD8' for f in frac]
    ax[1].bar(range(len(names)), frac, color=cols2, width=.62)
    for i, k in enumerate(names):
        ax[1].text(i, frac[i] + .02, f"{nv[k][0]}/{nv[k][1]}", ha='center', fontsize=9)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, fontsize=9, rotation=15)
    ax[1].set_ylim(0, 1.12); ax[1].set_ylabel('fraction of cases with ZERO voxels predicted')
    ax[1].axhline(1.0, color='#888', ls=':', lw=1)
    ax[1].set_title(f"Three lobes are never predicted at all  (n={meta['n_cases']} val cases)",
                    fontsize=11)
    ax[1].grid(axis='y', alpha=.3)

    lm, llo, lhi = meta['dice_lung']
    om, olo, ohi = meta['dice_lobe']
    fig.suptitle(f"Student ch={meta['ch']} depth={meta['depth']} ({meta['params']:,} params): "
                 f"lung-vs-background Dice {lm:.3f} [{llo:.3f}, {lhi:.3f}]  vs  "
                 f"between-lobe Dice {om:.3f} [{olo:.3f}, {ohi:.3f}]  —  {lm/om:.1f}x gap",
                 fontsize=10.5, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(RESULTS, f"seg3d_diag_{tag}.png"), dpi=140)
    print(f"  {len(rows)} 例 → results/seg3d_diag_{tag}.png")
    return 0


def rf(depths=(2, 3), ch=8, n_init=8, frac=0.9):
    """实测有效感受野：中心输出体素对输入的梯度，取包含 frac 梯度质量的直径。

    【为什么不用卷积公式推】那算的是**理论**感受野（梯度非零的最大范围），是上限。
    Luo et al. 2016 指出实际梯度呈高斯衰减，有效范围远小于理论值——本项目第一次
    正是按公式推出 ~29 体素、又按「梯度非零」测出 100% 覆盖，两个数都没用。
    取 90% 梯度质量的直径才是模型真正在看的范围。多次随机初始化平均，避免单次偶然。
    """
    import torch
    from seg3d_train import PATCH, build_net
    pz, py, px = PATCH
    c = (pz // 2, py // 2, px // 2)
    print(f"\n  patch={PATCH}  ch={ch}  {n_init} 次随机初始化平均  阈值={frac:.0%} 梯度质量\n")
    print(f"  {'depth':<8}{'z':>10}{'y':>10}{'x':>10}   （直径 mm，1.5mm 各向同性）")
    print("  " + "-" * 46)
    out = {}
    for d in depths:
        acc = None
        for k in range(n_init):
            torch.manual_seed(k)
            net = build_net(ch, d).eval()
            x = torch.zeros(1, 1, pz, py, px, requires_grad=True)
            net(x)[0, :, c[0], c[1], c[2]].sum().backward()
            g = x.grad[0, 0].abs().numpy()
            acc = g if acc is None else acc + g
        g = acc / n_init
        dia = []
        for ax, (cc, n) in enumerate(zip(c, PATCH, strict=True)):
            prof = g.sum(axis=tuple(i for i in range(3) if i != ax))
            prof = prof / prof.sum()
            ssum, r = prof[cc], 0
            while ssum < frac and r < n:
                r += 1
                ssum = prof[max(0, cc - r):min(n - 1, cc + r) + 1].sum()
            dia.append(2 * r * 1.5)
        out[d] = dia
        print(f"  depth={d:<2}{dia[0]:>9.0f}{dia[1]:>10.0f}{dia[2]:>10.0f}")
    # 教师每步的视野＝该例的面内最大边×spacing。**不能假定 512²**——本数据集面内
    # 中位仅 253 体素，按 512 写会把教师视野夸大到 768mm（实测中位 380mm）。
    # 只读 header 不加载数据体，297 例秒级完成。
    import glob

    import nibabel as nib
    ms = sorted(glob.glob(os.path.join(HERE, ".seg3d_cache", "*_msk.nii.gz")))
    geom = {}
    if ms:
        sh = np.array([nib.load(q).shape for q in ms])
        sps = {tuple(round(float(x), 3) for x in nib.load(q).header.get_zooms()[:3]) for q in ms}
        xy = np.maximum(sh[:, 0], sh[:, 1])
        geom = dict(n_cases=len(ms), spacings=sorted(map(list, sps)),
                    teacher_fov_mm_median=float(np.median(xy) * 1.5),
                    teacher_fov_mm_min=float(xy.min() * 1.5),
                    teacher_fov_mm_max=float(xy.max() * 1.5),
                    patch_mm=float(py * 1.5))
        print(f"\n  数据集 {len(ms)} 例，spacing {sorted(map(list, sps))}")
        print(f"  教师每步视野（面内最大边×spacing）中位 {np.median(xy)*1.5:.0f}mm，"
              f"范围 {xy.min()*1.5:.0f}–{xy.max()*1.5:.0f}mm")
        print(f"  学生 patch = {py*1.5:.0f}mm")
    out_json = os.path.join(RESULTS, "seg3d_geom.json")
    json.dump(dict(erf_mm={str(k): v for k, v in out.items()}, ch=ch,
                   n_init=n_init, frac=frac, **geom),
              open(out_json, 'w'), indent=1)
    print("    → results/seg3d_geom.json")
    return out


def extent(n=12):
    """实测真值里各肺叶的 z 跨度——用来判断 patch 的 z=32（48mm）够不够。

    【口径提醒】各行的病例子集不同：一个肺叶只在包含它的病例里被统计，
    故「全肺」的中位数可能小于某个单叶的中位数（含下叶的病例往往扫描范围更大）。
    这不是矛盾，是不同子集的中位数，不可横向相减。
    """
    import glob

    from seg3d_train import LOBES, load_zhw
    CACHE = os.path.join(HERE, ".seg3d_cache")
    msks = sorted(glob.glob(os.path.join(CACHE, "*_msk.nii.gz")))[:n]
    if not msks:
        print(f"  缺数据（{CACHE}），先跑 seg3d_data.py fetch"); return 1
    per = {v: [] for v in LOBES}
    lung = []
    for p in msks:
        gt = load_zhw(p)
        if not np.isin(gt, LOBES).any():
            continue
        z = np.where(np.isin(gt, LOBES).any(axis=(1, 2)))[0]
        lung.append(z.max() - z.min() + 1)
        for v in LOBES:
            zz = np.where((gt == v).any(axis=(1, 2)))[0]
            if len(zz):
                per[v].append(zz.max() - zz.min() + 1)
    print(f"\n  {len(lung)} 例（1.5mm 各向同性）。各行子集不同，勿横向相减。\n")
    print(f"  {'结构':<12}{'例数':>6}{'z 跨度中位数':>14}{'mm':>8}{'相对 patch z=32':>18}")
    print("  " + "-" * 60)
    for v in LOBES:
        if per[v]:
            m = float(np.median(per[v]))
            print(f"  {NAMES[v - 9]:<12}{len(per[v]):>6}{m:>12.0f} 层"
                  f"{m*1.5:>7.0f}{m/32:>15.1f}×")
    m = float(np.median(lung))
    print(f"  {'全肺':<12}{len(lung):>6}{m:>12.0f} 层{m*1.5:>7.0f}{m/32:>15.1f}×")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        if len(sys.argv) > 2:
            return plot(sys.argv[2])
        import glob
        avail = sorted(os.path.basename(x)[11:-4]
                       for x in glob.glob(os.path.join(RESULTS, 'seg3d_diag_*.csv')))
        print("  用法：seg3d_diag.py plot <tag>\n  现有 tag：" + ", ".join(avail))
        return 1
    if len(sys.argv) > 1 and sys.argv[1] == 'rf':
        rf(); return 0
    if len(sys.argv) > 1 and sys.argv[1] == 'extent':
        return extent()
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val', help='诊断默认走 val，勿用 test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='', help='产物名后缀；缺省由 ckpt 的结构+训练量推出')
    ap.add_argument('--infer', default='sliding', choices=['sliding', 'zslab'],
                    help='sliding=按训练同尺寸 patch 滑窗（测精度用，默认）；'
                         'zslab=整幅 z 分块（与教师同分块，测成本用）')
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from seg3d_data import split as make_split
    from seg3d_eval import sliding_infer, zslab_infer
    from seg3d_teacher import bootstrap_ci
    from seg3d_train import PATCH, build_net, prep_case

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    depth = ck.get('depth', 2)
    net = build_net(ck['ch'], depth)
    net.load_state_dict(ck['state']); net.eval()
    dev = torch.device('cpu')          # 与教师同平台，时间才可比
    # 产物名必须同时含结构与训练量：曾经只按 ch/depth 命名，导致同一结构的短训与
    # 长训两次实验写到同一组文件，后跑的把先跑的覆盖掉（靠 git 才找回）。
    tot = ck.get('total_steps')
    if a.tag:
        tag = a.tag
    elif tot:
        tag = f"ch{ck['ch']}d{depth}_{tot}s_{a.infer}"
    else:
        tag = f"ch{ck['ch']}d{depth}_{a.infer}"
        print(f"  ⚠ 该 ckpt 未记训练量（早于此字段），产物名回退为 {tag}，"
              f"可能覆盖同结构的旧结果；需要区分请显式传 --tag")
    print(f"\n  模型 ch={ck['ch']} depth={depth} 参数={ck['params']:,} "
          f"best_ep={ck['best_ep']} val_patch_dice={ck['val_patch_dice']:.4f}")

    cases = make_split()[a.split]
    if a.limit:
        cases = cases[:a.limit]
    rows, lung_d, lobe_d = [], [], []
    # 记录每个叶「在场却零预测」的次数——这是失败模式 ② 最直接的证据
    absent = {k: 0 for k in NAMES}
    present = {k: 0 for k in NAMES}

    for i, cid in enumerate(cases, 1):
        img, gt = prep_case(cid)
        if not (gt > 0).any():
            print(f"  [{i}/{len(cases)}] {cid}: 真值无肺叶，跳过"); continue
        t0 = time.perf_counter()
        pred = (sliding_infer(net, img, PATCH, dev, overlap=0.25)
                if a.infer == 'sliding' else zslab_infer(net, img, PATCH[0], dev))
        dt = time.perf_counter() - t0

        # ① 肺 vs 背景：把五叶合并成一个前景类
        tb, pb = (gt > 0), (pred > 0)
        s = tb.sum() + pb.sum()
        d_lung = float(2.0 * (tb & pb).sum() / s) if s else float('nan')

        # ② 五叶之间：逐叶 Dice，只统计真值中在场的叶
        ds = {}
        for k in NAMES:
            t, p = (gt == k), (pred == k)
            if not t.any():
                continue
            present[k] += 1
            if not p.any():
                absent[k] += 1
            ss = t.sum() + p.sum()
            ds[k] = float(2.0 * (t & p).sum() / ss)
        d_lobe = float(np.mean(list(ds.values()))) if ds else float('nan')

        lung_d.append(d_lung); lobe_d.append(d_lobe)
        rows.append(dict(case=cid, n_lobe_present=len(ds),
                         dice_lung=round(d_lung, 4), dice_lobe_mean=round(d_lobe, 4),
                         **{f"dice_{NAMES[k]}": (round(ds[k], 4) if k in ds else '')
                            for k in NAMES},
                         sec=round(dt, 1)))
        print(f"  [{i}/{len(cases)}] {cid}  肺={d_lung:.4f}  五叶={d_lobe:.4f}  {dt:.0f}s")
        sys.stdout.flush()

    if not rows:
        print("  无有效病例"); return 1

    # bootstrap_ci 只返回 (lo, hi)，均值自己算——与 seg3d_teacher 同一实现，口径一致
    lm, om = float(np.mean(lung_d)), float(np.mean(lobe_d))
    llo, lhi = bootstrap_ci(lung_d)
    olo, ohi = bootstrap_ci(lobe_d)
    print(f"\n  === {len(rows)} 例（{a.split} 集）===")
    print(f"  肺 vs 背景   Dice {lm:.4f}  95%CI [{llo:.4f}, {lhi:.4f}]")
    print(f"  五叶之间     Dice {om:.4f}  95%CI [{olo:.4f}, {ohi:.4f}]")
    print(f"  两者之比     {lm/om:.1f}×\n")
    print("  逐叶「在场却一个体素都没预测」的例数：")
    for k in NAMES:
        if present[k]:
            print(f"    {NAMES[k]:<10}{absent[k]:>3}/{present[k]:<3} 例")

    with open(os.path.join(RESULTS, f"seg3d_diag_{tag}.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS, f"seg3d_diag_{tag}.json"), 'w') as f:
        json.dump(dict(ckpt=os.path.basename(a.ckpt), ch=ck['ch'], depth=depth,
                       params=ck['params'], split=a.split, n_cases=len(rows),
                       dice_lung=[lm, llo, lhi], dice_lobe=[om, olo, ohi],
                       never_predicted={NAMES[k]: [absent[k], present[k]] for k in NAMES}),
                  f, indent=1)
    print(f"\n    → results/seg3d_diag_{tag}.csv + .json")
    plot(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
