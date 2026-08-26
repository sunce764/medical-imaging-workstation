# =============================================================================
# 研究二的样本量扩充：把 21 器官 Dice 从 n=1 扩到多例
# ---------------------------------------------------------------------------
# 动机：研究二用一例带真值的公开 CT 实测出模型的标签方案，并给出 21 器官平均 Dice ≈ 0.92。
#       出处判断只需一例（恒等对角线是定性证据），但**那个 0.92 一直是 n=1**，
#       README、技术报告与模型说明卡都在如实标注这一点。本项目已经两次看到单例的
#       危险，且方向相反：
#         · 肺叶单例 0.956–0.991 → 57 例 0.887（单例偏**乐观**）；
#         · spacing 修复单例 +0.064 → 20 例 +0.155（单例偏**保守**）。
#       结论不是「单例总是偏乐观」，而是**单例的偏向事先不可知**。故必须实测。
#
# 方法：随机抽 n 例（固定 seed，可复现），走**产品引擎** ai_engine 推理，与未经改动的
#       真值逐器官算 Dice。数据原生 1.5mm 各向同性 = 模型训练 spacing，引擎的 5% 阈值
#       会判定无需重采样，因此这里测到的与直通路径等价，可直接与研究二的单例基线比较。
#
# 【已提交 CSV 的时点：末窗回移（2a50e37）之前】本脚本不自带推理实现，而是**实时调用
#       ai_engine**。已提交的那批 CSV 生成于 ai_engine 仍用 `for z0 in range(0, Z, DZ)`
#       的时候，末块补零；产品其后改成回移到 [Z-DZ, Z)。于是出现一个必须写明的后果：
#       **当前源码已不再逐步复现这份已提交产物**——今天重跑会走当前引擎，得到的是另一
#       条路径上的数。故不得再写「本结果代表产品当前行为」。本轮未重跑（需 ONNX 推理）。
#
# 口径（两种均值不可混为一谈）：
#   · **患者级**：先在每例内部对该例在场的器官取平均，再跨例统计 → 主结论，
#     与研究二单例数字同口径，可直接对照；
#   · **器官级**：每个器官跨例统计 → 用来看哪些器官稳定、哪些方差大。
#   一个器官只在部分病例中在场（扫描范围不同），故两者的样本量本就不同。
#
# 用法：python experiments/seg_multi.py [n] [seed]        默认 20 例，seed=0
#       python experiments/seg_multi.py plot              由已有 CSV 重绘，不重跑推理
# 产出：results/seg_multi.csv + seg_multi_per_organ.csv + seg_multi.png
# =============================================================================

import csv
import glob
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from seg_validate import TS_TOTAL, dice, load_zhw  # noqa: E402  复用同一套口径

RESULTS = os.path.join(_HERE, "results")
CACHE = os.path.join(_HERE, ".seg3d_cache")


def boot_ci(vals, seed=0, n_boot=5000):
    """均值的 bootstrap 95% CI。样本数 < 3 时不报 CI——三个点撑不起区间估计。"""
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float(a.mean()) if a.size else float('nan'), float('nan'), float('nan')
    rs = np.random.RandomState(seed)
    b = [a[rs.randint(0, a.size, a.size)].mean() for _ in range(n_boot)]
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run(n_cases, seed):
    import ai_engine
    imgs = sorted(glob.glob(os.path.join(CACHE, "*_img.nii.gz")))
    if not imgs:
        print(f"  缺数据（{CACHE}），获取方式见 seg_validate.py 头部"); return 1
    rng = np.random.RandomState(seed)
    pick = [imgs[i] for i in rng.permutation(len(imgs))[:n_cases]]
    rows, per_organ = [], {}
    for k, ip in enumerate(pick, 1):
        cid = os.path.basename(ip).replace("_img.nii.gz", "")
        gp = ip.replace("_img", "_msk")
        if not os.path.exists(gp):
            continue
        try:
            img, gt = load_zhw(ip), load_zhw(gp)
        except Exception as ex:
            print(f"  [{k}/{len(pick)}] {cid} 读取失败：{ex}"); continue
        labs = [int(v) for v in np.unique(gt) if 0 < v <= 24]
        if not labs:
            print(f"  [{k}/{len(pick)}] {cid} 无在场器官，跳过"); continue
        got = {}
        t0 = time.perf_counter()
        eng = ai_engine.AutoAIEngineThread(img.astype(np.float32),
                                           lambda m, ms, _g=got: _g.update(m=m),
                                           spacing=(1.5, 1.5, 1.5))
        eng._run_body()
        dt = time.perf_counter() - t0
        pred = got.get('m')
        if pred is None or pred.shape != gt.shape:
            print(f"  [{k}/{len(pick)}] {cid} 输出异常，跳过"); continue
        ds = {}
        for v in labs:
            d = dice(pred == v, gt == v)
            ds[v] = d
            per_organ.setdefault(v, []).append(d)
        mean = float(np.mean(list(ds.values())))
        rows.append(dict(case=cid, n_organ=len(labs), mean_dice=round(mean, 4),
                         min_organ=TS_TOTAL.get(min(ds, key=ds.get), '?'),
                         min_dice=round(min(ds.values()), 4), sec=round(dt, 1)))
        print(f"  [{k}/{len(pick)}] {cid}  {len(labs):2d}器官  均值 {mean:.4f}   "
              f"最差 {rows[-1]['min_organ']} {rows[-1]['min_dice']:.3f}   {dt:.0f}s")
    if not rows:
        print("  无有效病例"); return 1

    pm, plo, phi = boot_ci([r['mean_dice'] for r in rows], seed)
    with open(os.path.join(RESULTS, "seg_multi.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        # 患者级均值与 CI 作为汇总行落盘。**CI 必须由实验产出，不能让下游重算**——
        # 模型说明卡曾自己用 random.Random 再算一遍，与本脚本的 np.random.RandomState
        # 产生不同的重采样序列，CI 下界差了 0.001。两套实现必然漂移，单一数据源才对。
        # case 前缀 "#" 标记为汇总行，读取端据此排除（否则会被当成第 N+1 个病例）。
        w.writerow({})
        w.writerow(dict(case=f"# summary n={len(rows)}", n_organ='',
                        mean_dice=f"{pm:.4f}", min_organ=f"ci={plo:.4f}..{phi:.4f}",
                        min_dice='', sec=''))
    with open(os.path.join(RESULTS, "seg_multi_per_organ.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['label', 'organ', 'n_cases_present', 'mean_dice', 'ci_lo', 'ci_hi', 'min_dice'])
        for v in sorted(per_organ, key=lambda x: -np.mean(per_organ[x])):
            m, lo, hi = boot_ci(per_organ[v], seed)
            w.writerow([v, TS_TOTAL.get(v, str(v)), len(per_organ[v]), f"{m:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}", f"{min(per_organ[v]):.4f}"])
    _report(rows, per_organ, seed)
    return 0


def _report(rows, per_organ, seed=0):
    pm, plo, phi = boot_ci([r['mean_dice'] for r in rows], seed)
    print(f"\n  === 患者级（{len(rows)} 例）===")
    print(f"  平均 Dice {pm:.4f}  95%CI [{plo:.4f}, {phi:.4f}]")
    print(f"  逐例范围 [{min(r['mean_dice'] for r in rows):.4f}, "
          f"{max(r['mean_dice'] for r in rows):.4f}]")
    print("\n  === 器官级（跨例，仅列在场 ≥3 例者）===")
    for v in sorted(per_organ, key=lambda x: np.mean(per_organ[x])):
        vals = per_organ[v]
        if len(vals) < 3:
            continue
        m, lo, hi = boot_ci(vals, seed)
        flag = '  ← 弱' if m < 0.75 else ''
        print(f"    {TS_TOTAL.get(v, v):<16}{len(vals):>3}例  {m:.3f} [{lo:.3f}, {hi:.3f}]{flag}")
    print("\n    → results/seg_multi.csv + seg_multi_per_organ.csv")


def plot():
    p = os.path.join(RESULTS, "seg_multi.csv")
    q = os.path.join(RESULTS, "seg_multi_per_organ.csv")
    if not (os.path.exists(p) and os.path.exists(q)):
        print("  缺产物，先跑 seg_multi.py"); return 1
    with open(p, encoding='utf-8-sig') as f:
        # 【必须跳过空行与汇总行】seg_multi.csv 末尾有一个 ',,,,,'  分隔行和一行
        # '# summary n=20,,0.9090,...'。直接 float(r['mean_dice']) 会在空行上抛
        # ValueError——README 里文档化的 `seg_multi.py plot` 因此在已提交的产物上
        # 直接崩掉。汇总行的 Dice 本身是合法浮点，靠 '#' 标记排除，否则它会被当成
        # 第 21 个病例算进 CI。seg_spacing.py:257 一直是这么做的，这里漏了。
        cases = []
        for r in csv.DictReader(f):
            cid = (r.get('case') or '').strip()
            if not cid or cid.startswith('#'):
                continue
            try:
                cases.append((cid, float(r['mean_dice'])))
            except (TypeError, ValueError):
                continue
    with open(q, encoding='utf-8-sig') as f:
        orgs = [(r['organ'], int(r['n_cases_present']), float(r['mean_dice']),
                 float(r['ci_lo']), float(r['ci_hi'])) for r in csv.DictReader(f)]
    vals = [v for _, v in cases]
    m, lo, hi = boot_ci(vals)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    order = np.argsort(vals)
    ax[0].barh(range(len(vals)), [vals[i] for i in order], color='#3B7DD8', height=.72)
    ax[0].axvline(m, color='#C0392B', ls='--', lw=1.4,
                  label=f'{len(vals)}-case mean {m:.3f}  95%CI [{lo:.3f}, {hi:.3f}]')
    ax[0].axvline(0.9219, color='#1B7F4B', ls=':', lw=1.6, label='single-case baseline 0.922 (n=1)')
    ax[0].set_yticks(range(len(vals)))
    ax[0].set_yticklabels([cases[i][0] for i in order], fontsize=7)
    ax[0].set_xlabel('per-case mean Dice over the organs present in that case')
    ax[0].set_xlim(0, 1); ax[0].grid(axis='x', alpha=.3); ax[0].legend(fontsize=8, loc='lower right')
    ax[0].set_title('Patient level — same definition as the single-case figure')

    orgs = sorted(orgs, key=lambda t: t[2])
    y = range(len(orgs))
    # 在场例数 <3 的器官不给 CI（三个点撑不起区间估计），画成灰色空心点。
    # 不能直接和 n=20 的实心点并列——那会让读者以为两者同样可信。
    solid = [(i, o) for i, o in enumerate(orgs) if o[1] >= 3]
    thin = [(i, o) for i, o in enumerate(orgs) if o[1] < 3]
    if solid:
        ax[1].errorbar([o[2] for _, o in solid], [i for i, _ in solid],
                       xerr=[[o[2] - o[3] for _, o in solid], [o[4] - o[2] for _, o in solid]],
                       fmt='o', ms=4, capsize=3, lw=1.2, color='#3B7DD8')
    if thin:
        ax[1].scatter([o[2] for _, o in thin], [i for i, _ in thin], s=34,
                      facecolors='none', edgecolors='#999', linewidths=1.2,
                      label='n < 3: no CI, not comparable')
        ax[1].legend(fontsize=7.5, loc='lower left')
    ax[1].axvline(0.75, color='#C0392B', ls=':', lw=1, alpha=.7)
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([f"{o[0]} (n={o[1]})" for o in orgs], fontsize=7.5)
    ax[1].set_xlabel('Dice across cases (bars = bootstrap 95% CI)')
    ax[1].set_xlim(0, 1); ax[1].grid(axis='x', alpha=.3)
    ax[1].set_title('Organ level — which organs are actually reliable')
    fig.suptitle('Study II at scale: the 21-organ Dice is no longer a single-case number',
                 fontsize=12.5, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "seg_multi.png"), dpi=140)
    plt.close(fig)
    print(f"  {len(vals)} 例  患者级 {m:.4f} [{lo:.4f}, {hi:.4f}]")
    print("    → results/seg_multi.png")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == 'plot':
        sys.exit(plot())
    sys.exit(run(int(a[0]) if a else 20, int(a[1]) if len(a) > 1 else 0))
