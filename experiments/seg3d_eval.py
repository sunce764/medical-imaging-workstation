# =============================================================================
# 研究四（整卷评估）：学生模型在测试集上的 Dice / 推理时间 / 峰值内存
# ---------------------------------------------------------------------------
# 【为什么不能拿训练时的 patch-Dice 当结论】
#   训练里报的是 patch 级 Dice，而 patch 有一半是围绕肺叶中心采的——那是被过采样
#   偏置过的分布，数值必然高于真实。整卷推理才是产品实际做的事，也才是与教师
#   （organs.onnx 整卷滑窗）唯一可比的口径。
#
# 【口径必须与教师逐项对齐，否则「学生比教师差多少」这个数没有意义】
#   · 同一批测试病例（患者级划分，seed 固定）
#   · 同样的 Dice 定义：两者皆空记为无定义（nan）而非 0
#   · 同样的峰值内存测法与平台换算
#   · 同样只统计真值中在场的肺叶
#
# 用法：python experiments/seg3d_eval.py --ckpt results/seg3d_w8.pt
# 产出：results/seg3d_student_ch{ch}[d{depth}].csv + 汇总 JSON
#       （depth≠2 时文件名带 d{depth}，与 seg3d_train 的权重命名一一对应；
#        否则不同深度的评估结果会写到同一个文件互相覆盖）
# =============================================================================

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CACHE = os.path.join(HERE, ".seg3d_cache")
RESULTS = os.path.join(HERE, "results")


def zslab_infer(net, img, dz, dev):
    """**与教师完全相同的分块策略**：整 xy 平面、沿 z 每 dz 层一块、无重叠。

    【为什么必须对齐分块策略】学生是全卷积网络，吃任意尺寸，本可以按训练时的
    32×128×128 patch 加 25% 重叠来推理。但那样对 145×320×320 的体积要跑 54 块、
    实际处理 28.3M 体素，而教师只跑 5 块、处理 16.4M——**学生因重叠多算了 1.7 倍**。
    实测后果：学生参数少 365 倍，μs/体素反而比教师高 9%。那比的是推理策略，
    不是模型。故此处照搬 ai_engine 的 z 分块，两边只剩模型这一个变量。

    边长补到 8 的倍数：网络有两次 2× 下采样，非整除会在上采样后对不齐。
    """
    import torch
    Z, H, W = img.shape
    ph, pw = (-H) % 8, (-W) % 8
    seg = np.zeros((Z, H, W), np.uint8)
    with torch.no_grad():
        for z0 in range(0, Z, dz):
            z1 = min(z0 + dz, Z)
            blk = img[z0:z1]
            pd = (-blk.shape[0]) % 8
            if pd or ph or pw:
                blk = np.pad(blk, ((0, pd), (0, ph), (0, pw)), mode='constant')
            o = net(torch.from_numpy(np.ascontiguousarray(blk[None, None])).float().to(dev))
            seg[z0:z1] = o[0].argmax(0).cpu().numpy().astype(np.uint8)[:z1 - z0, :H, :W]
    return seg


def sliding_infer(net, img, patch, dev, overlap=0.25):
    """沿三轴滑窗推理整卷，重叠区按累加平均融合。

    保留此实现供对照：它在小显存下更省内存，但因重叠会多算 1.7 倍体素，
    与教师的 z 分块口径不可比（见 zslab_infer 的说明）。正式评估用 zslab_infer。
    """
    import torch
    from seg3d_train import N_CLASS
    Z, H, W = img.shape
    pz, py, px = patch
    # 体积小于 patch 时补零，推理后裁回
    pad = [max(0, pz - Z), max(0, py - H), max(0, px - W)]
    if any(pad):
        img = np.pad(img, ((0, pad[0]), (0, pad[1]), (0, pad[2])), mode='constant')
    Zp, Hp, Wp = img.shape
    step = [max(1, int(p * (1 - overlap))) for p in patch]
    zs = list(range(0, max(1, Zp - pz + 1), step[0]))
    ys = list(range(0, max(1, Hp - py + 1), step[1]))
    xs = list(range(0, max(1, Wp - px + 1), step[2]))
    # 保证最后一块贴到边界，否则末端窄条永远采不到
    if zs[-1] != Zp - pz: zs.append(Zp - pz)
    if ys[-1] != Hp - py: ys.append(Hp - py)
    if xs[-1] != Wp - px: xs.append(Wp - px)
    acc = np.zeros((N_CLASS, Zp, Hp, Wp), np.float32)
    cnt = np.zeros((Zp, Hp, Wp), np.float32)
    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    blk = img[z0:z0 + pz, y0:y0 + py, x0:x0 + px][None, None]
                    o = net(torch.from_numpy(np.ascontiguousarray(blk)).float().to(dev))
                    acc[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += o[0].cpu().numpy()
                    cnt[z0:z0 + pz, y0:y0 + py, x0:x0 + px] += 1
    seg = acc.argmax(0).astype(np.uint8)
    return seg[:Z, :H, :W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--infer', default='sliding', choices=['sliding', 'zslab'],
                    help='sliding=按训练 patch 尺寸滑窗（测精度用，默认）；'
                         'zslab=整幅 xy 沿 z 分块（与教师同分块，测成本用）')
    ap.add_argument('--tag', default='', help='产物名后缀；缺省由 ckpt 推出')
    a = ap.parse_args()

    import resource

    import torch
    torch.set_num_threads(1)
    from seg3d_data import split as make_split
    from seg3d_teacher import LUNG_LOBES, bootstrap_ci, dice
    from seg3d_train import LOBES, build_net, prep_case

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    # 学生模型在 CPU 上评时间：与教师（onnxruntime CPU）同平台才可比。
    # 用 MPS 测出来的秒数与教师的 CPU 秒数放同一张表，是在比硬件不是比模型。
    dev = torch.device('cpu')
    depth = ck.get('depth', 2)                       # 旧 ckpt 无 depth 键，默认 2
    net = build_net(ck['ch'], depth).to(dev)
    net.load_state_dict(ck['state']); net.eval()
    npar = sum(p.numel() for p in net.parameters())
    print(f"  学生模型 ch={ck['ch']}  参数 {npar/1e6:.4f}M  patch={tuple(ck['patch'])}")
    print(f"  设备={dev}（与教师同为 CPU，保证时间可比）\n")

    cases = make_split()[a.split]
    if a.limit:
        cases = cases[:a.limit]
    # 【ru_maxrss 的语义陷阱】它是**进程生命周期内**的峰值，单调不减：逐例 append
    # 得到的是非递减序列，对它取 mean 既不是「每例峰值」也不是「全程峰值」，
    # 只是一条爬升曲线的平均高度，没有可解释的含义。真正有意义的是最大值＝全程峰值。
    # mean 仍保留，仅为兼容此前已产出的 JSON（那些数字就是这么来的，不重跑、不改写）。
    rows, times, peaks, vox_counts = [], [], [], []
    for i, cid in enumerate(cases, 1):
        img, lab = prep_case(cid)          # 与训练同一套预处理与标签重映射
        present = [k for k in range(1, len(LOBES) + 1) if (lab == k).any()]
        if not present:
            print(f"  [{i}/{len(cases)}] {cid}: 真值无肺叶，跳过")
            continue
        t0 = time.perf_counter()
        # 【两条推理路径，用途不同，不可混用】
        #   zslab  整幅 xy 沿 z 分块，与教师（ai_engine）同一分块口径 → 测**成本**
        #   sliding 按训练 patch 尺寸滑窗 → 测**精度**
        # 二者在同一份权重上相差 0.25 Dice：InstanceNorm3d 逐样本在空间维求统计量，
        # 而 HU 归一化后空气与补零同为 0，张量越大统计量偏得越狠，前景被压成背景。
        # 详见 seg3d_infer_bias.py 的五条对照。默认走 sliding，因为本脚本报的是 Dice。
        seg = (sliding_infer(net, img, tuple(ck['patch']), dev, 0.25)
               if a.infer == 'sliding' else zslab_infer(net, img, tuple(ck['patch'])[0], dev))
        dt = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
        times.append(dt); peaks.append(peak); vox_counts.append(int(np.prod(img.shape)))
        ds = []
        for k in present:
            d = dice(seg == k, lab == k)
            lid = LOBES[k - 1]
            ds.append(d)
            rows.append(dict(case=cid, label=lid, organ=LUNG_LOBES[lid], dice=round(d, 4)))
        print(f"  [{i}/{len(cases)}] {cid}  {img.shape}  {dt:.1f}s  "
              f"在场 {len(present)}/5  平均 Dice={np.nanmean(ds):.3f}")
        sys.stdout.flush()

    # 必须带 depth 与推理路径：同一 ch 的不同深度是不同模型；同一份权重在两条
    # 推理路径下相差 0.25 Dice，混进同一文件名必然被误读。与 seg3d_diag 同规则。
    base = f"ch{ck['ch']}" if depth == 2 else f"ch{ck['ch']}d{depth}"
    tot = ck.get('total_steps')
    if a.tag:
        tag = a.tag
    elif tot:
        tag = f"{base}_{tot}s_{a.infer}"
    else:
        tag = f"{base}_{a.infer}"
        print(f"  ⚠ 该 ckpt 未记训练量（早于该字段），产物名为 {tag}；"
              f"需区分训练量请显式传 --tag")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"seg3d_student_{tag}.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'label', 'organ', 'dice'])
        w.writeheader(); w.writerows(rows)

    print(f"\n  ===== 学生 {tag}（{len(set(r['case'] for r in rows))} 例）=====")
    print(f"  {'标签':>4} {'器官':<24}{'n':>4}{'平均Dice':>10}{'95%CI':>18}")
    per = []
    for lid, name in LUNG_LOBES.items():
        vals = [r['dice'] for r in rows if r['label'] == lid]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        per.append(dict(label=lid, organ=name, n=len(vals),
                        mean_dice=round(float(np.nanmean(vals)), 4),
                        ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        print(f"  {lid:>4} {name:<24}{len(vals):>4}{np.nanmean(vals):>10.4f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")
    allv = [r['dice'] for r in rows]
    lo, hi = bootstrap_ci(allv)
    us_per_vox = float(np.sum(times) / np.sum(vox_counts) * 1e6)
    print(f"\n  五叶总体 Dice = {np.nanmean(allv):.4f}，95% CI [{lo:.4f}, {hi:.4f}]"
          f"（n={len(allv)} 叶次）")
    print(f"  推理 {np.mean(times):.1f} ± {np.std(times):.1f} s/例   "
          f"{us_per_vox:.3f} μs/体素   全程峰值 {max(peaks):.2f} GB")

    out = dict(ch=ck['ch'], depth=depth, params=npar,
               # 推理口径必须随产物一起存：同一份权重在两条路径下相差 0.25 Dice，
               # 而教师基线走的是 zslab。产物若不自描述，下游只能靠文件名猜，
               # 而文件名是可以被改的。seg3d_report 据此拒绝混口径入权衡曲线。
               infer=a.infer,
               n_cases=len(set(r['case'] for r in rows)),
               cases=sorted(set(r['case'] for r in rows)),      # 供 report 校验是否同一批
               overall_mean=float(np.nanmean(allv)), overall_ci=[lo, hi],
               n_lobe_instances=len(allv), per_organ=per,
               infer_sec_mean=float(np.mean(times)), us_per_voxel=us_per_vox,
               peak_gb_max=float(max(peaks)), peak_gb_mean=float(np.mean(peaks)),
               val_patch_dice=ck.get('val_patch_dice'), best_ep=ck.get('best_ep'))
    with open(os.path.join(RESULTS, f"seg3d_student_{tag}.json"), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"    → results/seg3d_student_{tag}.csv + .json")

    # 与教师并排——两边口径已逐项对齐，这张表才是本研究的主结果
    tp = os.path.join(RESULTS, "seg3d_teacher_summary.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        print("\n  ===== 对标 organs.onnx =====")
        print(f"  {'':<10}{'参数':>10}{'Dice':>9}{'μs/体素':>11}{'峰值GB':>9}")
        print(f"  {'教师':<10}{31.2:>9.1f}M{t['overall_mean']:>9.4f}"
              f"{'—':>11}{t.get('peak_gb_max', t['peak_gb_mean']):>9.2f}")
        print(f"  {'学生':<10}{npar/1e6:>9.4f}M{out['overall_mean']:>9.4f}"
              f"{us_per_vox:>11.3f}{out['peak_gb_max']:>9.2f}")
        print(f"  参数比 1:{31.2e6/npar:.0f}   Dice 差 {t['overall_mean']-out['overall_mean']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
