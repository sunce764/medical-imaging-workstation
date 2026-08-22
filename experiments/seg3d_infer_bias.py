# =============================================================================
# 研究四（评估偏差定位）：一个输入尺寸依赖，把学生模型的五叶 Dice 压低了 12 倍
# ---------------------------------------------------------------------------
# 【背景】seg3d_diag 用 zslab_infer（整幅 xy、沿 z 分块）测出学生五叶 Dice 0.4903，
#       而训练时报的 val patch-Dice 是 0.8186。两者差 0.33——这个缺口本身就是
#       信号，当时没有深究。改用与训练同尺寸的滑窗推理后，同一份权重给出 0.7457。
#
# 【为什么需要这个脚本】"换个推理策略结果变好"只是相关。要把它变成结论，必须
#       逐个排除竞争解释。本脚本把五条互不依赖的论证做成可复现的子命令：
#
#         ab      A/B + 阴性对照      排除「重叠融合的功劳」
#         dose    剂量-反应           排除「滑窗机制本身让 Dice 虚高」「z 重叠的功劳」
#         pad     补零实验（核心）    排除全部内容类解释：内容一个体素都没变
#         norm    归一化统计量观测    从相关推进到机制
#         train   训练集对照          排除「泛化不足」
#
# 【机制】build_net 用 InstanceNorm3d，逐样本在空间维上求统计量。归一化后体外
#       空气（HU −1000）恰好是 0，补零也是 0；输入张量越大，近零区域占比越高，
#       统计量偏移越大，前景被压成背景。训练恒为 32×128×128 的实体素，推理却喂
#       整幅并补齐——两个分布不一致。学生没有任何数据增强，对此毫无抵抗力。
#
# 【尚未验证】教师（同款 nnU-Net + InstanceNorm，产品 ai_engine.py 同一策略）是否
#       也受影响。子命令 teacher 留了位置，但它要重跑 ONNX 推理（~100s/例），
#       按项目约定必须由用户显式发起，故需 --yes 才执行。
#
# 用法：python experiments/seg3d_infer_bias.py [ab|dose|pad|norm|train|all]
# 产出：results/seg3d_infer_bias_{子命令}.csv
# =============================================================================

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")

CKPT = os.path.join(RESULTS, "seg3d_w8d3.pt")
# 面内 W 远大于训练 patch 的 128；s0084/s0347 的 W ≤128，作阴性对照
VAL_BIG, VAL_SMALL = ['s0218', 's0330', 's0255', 's0182'], ['s0084', 's0347']
TRAIN_BIG, TRAIN_SMALL = ['s0075', 's0270', 's0285'], ['s0335', 's0172']
XY_DOSE = [128, 160, 192, 224, 256]


def _load():
    import torch
    from seg3d_train import build_net
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    net = build_net(ck['ch'], ck['depth'])
    net.load_state_dict(ck['state']); net.eval()
    return net, torch.device('cpu')


def _dice(gt, pr, n_cls=5):
    ds = []
    for k in range(1, n_cls + 1):
        t, p = gt == k, pr == k
        s = t.sum() + p.sum()
        if s:
            ds.append(2.0 * (t & p).sum() / s)
    return float(np.mean(ds)) if ds else float('nan')


def _write(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, f"seg3d_infer_bias_{name}.csv")
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"    → results/seg3d_infer_bias_{name}.csv")


def ab(cases=None, split_name='val'):
    """A/B：同一权重，只换推理策略。阴性对照排除「重叠融合的功劳」。"""
    from seg3d_eval import sliding_infer, zslab_infer
    from seg3d_train import PATCH, prep_case
    net, dev = _load()
    cases = cases or (VAL_BIG + VAL_SMALL)
    print(f"  {'case':<8}{'面内W':>7}{'zslab':>9}{'滑窗':>9}{'Δ':>9}{'秒':>7}")
    print("  " + "-" * 49)
    rows = []
    for c in cases:
        img, gt = prep_case(c)
        if not (gt > 0).any():
            continue
        t0 = time.perf_counter()
        a = _dice(gt, zslab_infer(net, img, PATCH[0], dev))
        b = _dice(gt, sliding_infer(net, img, PATCH, dev, 0.25))
        dt = time.perf_counter() - t0
        w = img.shape[2]
        note = '  ← 阴性对照' if w <= 128 else ''
        print(f"  {c:<8}{w:>7}{a:>9.4f}{b:>9.4f}{b - a:>+9.4f}{dt:>7.0f}{note}")
        sys.stdout.flush()
        rows.append([c, split_name, w, f"{a:.4f}", f"{b:.4f}", f"{b - a:+.4f}"])
    _write('ab' if split_name == 'val' else 'train',
           ['case', 'split', 'inplane_W', 'dice_zslab', 'dice_sliding', 'delta'], rows)
    return 0


def train():
    """训练集对照：模型见过的例子上差值是否同样大 → 排除「泛化不足」。

    仅用于这个对照，不产生任何对外报告的性能数字。
    """
    return ab(TRAIN_BIG + TRAIN_SMALL, 'train')


def dose():
    """剂量-反应：只改滑窗块的 xy，覆盖范围始终是整卷，信息不丢。

    裁剪式对照做不到这一点——裁剪会同时丢信息，两个方向的效应混在一起。
    """
    from seg3d_eval import sliding_infer, zslab_infer
    from seg3d_train import prep_case
    net, dev = _load()
    print("  " + ''.join(f"{x:>9}" for x in ['case', '面内W'] )
          + ''.join(f"{x:>9}" for x in XY_DOSE) + f"{'zslab':>9}")
    print("  " + "-" * (18 + 9 * len(XY_DOSE) + 9))
    rows = []
    for c in ['s0218', 's0182', 's0084']:
        img, gt = prep_case(c)
        vs = [_dice(gt, sliding_infer(net, img, (32, xy, xy), dev, 0.25)) for xy in XY_DOSE]
        zs = _dice(gt, zslab_infer(net, img, 32, dev))
        note = '  ← 阴性对照（整幅即 128）' if img.shape[2] <= 128 else ''
        print(f"  {c:>9}{img.shape[2]:>9}" + ''.join(f"{v:>9.4f}" for v in vs)
              + f"{zs:>9.4f}{note}")
        sys.stdout.flush()
        rows.append([c, img.shape[2]] + [f"{v:.4f}" for v in vs] + [f"{zs:.4f}"])
    _write('dose', ['case', 'inplane_W'] + [f"xy{x}" for x in XY_DOSE] + ['zslab'], rows)
    return 0


def pad():
    """补零实验：内容一个体素都没变，只在右下补零，看前景还剩多少。

    这是最干净的一条：s0084 整幅即 128²，放大到 256² 不引入任何新内容。
    若前景仍被抹掉，则一切「内容类」解释（体外空气、全局上下文、信息量）出局。
    """
    import torch
    from seg3d_train import prep_case
    net, dev = _load()
    img, gt = prep_case('s0084')
    z0 = int(np.argmax(np.isin(gt, [1, 2, 3, 4, 5]).sum(axis=(1, 2))))
    z0 = max(0, min(img.shape[0] - 32, z0 - 16))
    a = img[z0:z0 + 32]
    rows = []
    print(f"  s0084  内容固定为 {a.shape}，只改右下补零量\n")
    print(f"  {'张量 xy':>9}{'补零占比':>10}{'前景体素':>11}{'与原样一致率':>14}")
    print("  " + "-" * 46)
    base = None
    with torch.no_grad():
        for xy in [128, 160, 192, 224, 256]:
            b = np.pad(a, ((0, 0), (0, xy - 128), (0, xy - 128)), mode='constant')
            p = net(torch.from_numpy(b[None, None]).float())[0][:, :, :128, :128].argmax(0).numpy()
            if base is None:
                base = p
            fg = int((p > 0).sum())
            agree = float((p == base).mean())
            frac = 1 - (128 ** 2) / (xy ** 2)
            print(f"  {xy:>9}{frac:>10.1%}{fg:>11,}{agree:>14.1%}")
            rows.append([xy, f"{frac:.3f}", fg, f"{agree:.4f}"])
    _write('pad', ['tensor_xy', 'zero_pad_fraction', 'foreground_voxels', 'agreement'], rows)
    return 0


def norm():
    """直接观测 InstanceNorm3d 各层输入的统计量随张量尺寸如何漂移。"""
    import torch
    from seg3d_train import prep_case
    net, dev = _load()
    img, gt = prep_case('s0084')
    z0 = int(np.argmax(np.isin(gt, [1, 2, 3, 4, 5]).sum(axis=(1, 2))))
    z0 = max(0, min(img.shape[0] - 32, z0 - 16))
    a = img[z0:z0 + 32]
    b = np.pad(a, ((0, 0), (0, 128), (0, 128)), mode='constant')
    st = {}

    def hook(name):
        def h(m, i, o):
            st.setdefault(name, []).append((float(i[0].mean()), float(i[0].std())))
        return h
    hs = [m.register_forward_hook(hook(f"norm{i}"))
          for i, m in enumerate(net.modules()) if isinstance(m, torch.nn.InstanceNorm3d)]
    with torch.no_grad():
        net(torch.from_numpy(a[None, None]).float())
        net(torch.from_numpy(b[None, None]).float())
    for h in hs:
        h.remove()
    print("  s0084  内容相同，128² vs 256²（补零）\n")
    print(f"  {'层':<9}{'128² mean':>12}{'256² mean':>12}{'128² std':>11}{'256² std':>11}{'std 比':>9}")
    print("  " + "-" * 64)
    rows = []
    for k, v in st.items():
        (m1, s1), (m2, s2) = v
        print(f"  {k:<9}{m1:>12.4f}{m2:>12.4f}{s1:>11.4f}{s2:>11.4f}{s2 / s1:>9.2f}×")
        rows.append([k, f"{m1:.4f}", f"{m2:.4f}", f"{s1:.4f}", f"{s2:.4f}", f"{s2 / s1:.3f}"])
    _write('norm', ['layer', 'mean_128', 'mean_256', 'std_128', 'std_256', 'std_ratio'], rows)
    return 0


TEACHER_XY = [128, 192, 256]          # 教师 5 级下采样，每维须为 32 的倍数


def _teacher_sliding(sess, norm, patch, overlap=0.25):
    """教师的滑窗推理：按 patch 分块、累加 25 类 logits 后 argmax。

    产品 ai_engine._run_onnx_multiorgan 与 seg3d_teacher.run_onnx 都是整幅 xy、
    沿 z 分块、逐块独立 argmax。此处只改「每次前向的张量多大」，其余不动。
    """
    iname = sess.get_inputs()[0].name
    Z, H, W = norm.shape
    pz, py, px = patch
    pd = [max(0, pz - Z), max(0, py - H), max(0, px - W)]
    if any(pd):
        norm = np.pad(norm, ((0, pd[0]), (0, pd[1]), (0, pd[2])), mode='constant')
    Zp, Hp, Wp = norm.shape
    step = [max(1, int(p * (1 - overlap))) for p in patch]
    zs = list(range(0, max(1, Zp - pz + 1), step[0]))
    ys = list(range(0, max(1, Hp - py + 1), step[1]))
    xs = list(range(0, max(1, Wp - px + 1), step[2]))
    for lst, tot, p_ in ((zs, Zp, pz), (ys, Hp, py), (xs, Wp, px)):
        if lst[-1] != tot - p_:
            lst.append(tot - p_)
    acc = np.zeros((25, Zp, Hp, Wp), np.float32)
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                blk = norm[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
                out = sess.run(None, {iname: blk[None, None]})[0][0]
                acc[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += out
                del out
    seg = acc.argmax(0).astype(np.uint8)
    del acc
    return seg[:Z, :H, :W]


def _teacher_full_zov(sess, norm, dz=32, overlap=0.25):
    """整幅 xy（与产品一致）+ z 方向重叠、累加 logits。

    2×2 正交里的 B 格：只改 z 策略，xy 张量尺寸与产品现状完全相同，
    因此它与 A 的差值就是 z 分块独自造成的损失，不掺 xy 的份。
    """
    iname = sess.get_inputs()[0].name
    Z, H, W = norm.shape
    ph, pw = (-H) % 32, (-W) % 32
    if ph or pw:
        norm = np.pad(norm, ((0, 0), (0, ph), (0, pw)), mode='constant')
    Hp, Wp = norm.shape[1], norm.shape[2]
    step = max(1, int(dz * (1 - overlap)))
    zs = list(range(0, max(1, Z - dz + 1), step))
    if zs[-1] != max(0, Z - dz):
        zs.append(max(0, Z - dz))
    Zp = max(Z, dz)
    if Zp > Z:
        norm = np.pad(norm, ((0, Zp - Z), (0, 0), (0, 0)), mode='constant')
    acc = np.zeros((25, Zp, Hp, Wp), np.float32)
    for z0 in zs:
        blk = norm[z0:z0 + dz]
        out = sess.run(None, {iname: blk[None, None]})[0][0]
        acc[:, z0:z0 + dz] += out
        del out
    seg = acc.argmax(0).astype(np.uint8)
    del acc
    return seg[:Z, :H, :W]


def _teacher_xy_noov(sess, norm, xy, dz=32):
    """块 xy + z 无重叠 + 逐块独立 argmax。2×2 正交里的 C 格。"""
    iname = sess.get_inputs()[0].name
    Z, H, W = norm.shape
    ph, pw = (-H) % xy, (-W) % xy
    if ph or pw:
        norm = np.pad(norm, ((0, 0), (0, ph), (0, pw)), mode='constant')
    Hp, Wp = norm.shape[1], norm.shape[2]
    seg = np.zeros((Z, Hp, Wp), np.uint8)
    for z0 in range(0, Z, dz):
        z1 = min(z0 + dz, Z)
        for y0 in range(0, Hp, xy):
            for x0 in range(0, Wp, xy):
                blk = norm[z0:z1, y0:y0 + xy, x0:x0 + xy]
                pd = (-blk.shape[0]) % 32
                if pd:
                    blk = np.pad(blk, ((0, pd), (0, 0), (0, 0)), mode='constant')
                out = sess.run(None, {iname: blk[None, None]})[0][0]
                seg[z0:z1, y0:y0 + xy, x0:x0 + xy] = \
                    out.argmax(0).astype(np.uint8)[:z1 - z0]
                del out
    return seg[:, :H, :W]


def grid():
    """2×2 正交：把 xy 尺寸与 z 分块两个因素分开。

    A 整幅+z无重叠（产品现状） / B 整幅+z重叠 / C 块256+z无重叠 / D 块256+z重叠
    B−A 是 z 独自的贡献，C−A 是 xy 独自的贡献，D 是两者合力。
    """
    if '--yes' not in sys.argv:
        print("  这会重跑 ONNX 推理。确认请加 --yes"); return 1
    import resource

    from seg3d_teacher import make_session, run_onnx
    from seg3d_train import LOBES, prep_case
    sess = make_session()
    XY = 256
    print(f"  {'case':<8}{'面内W':>7}{'A 现状':>9}{'B 仅z重叠':>11}"
          f"{'C 仅块xy':>10}{'D 两者':>9}{'B−A':>8}{'C−A':>8}{'峰值GB':>8}{'秒':>7}")
    print("  " + "-" * 78)
    rows = []
    for c in ['s0218', 's0182', 's0084']:
        img, gt = prep_case(c)
        hu = img * 1400.0 - 1000.0
        norm = ((np.clip(hu, -1000, 400) + 1000.0) / 1400.0).astype(np.float32)

        def remap(seg):
            out = np.zeros(seg.shape, np.uint8)
            for k, lid in enumerate(LOBES, start=1):
                out[seg == lid] = k
            return out

        t0 = time.perf_counter()
        a = _dice(gt, remap(run_onnx(hu, sess=sess, quiet=True)[0]))
        b = _dice(gt, remap(_teacher_full_zov(sess, norm)))
        cc = _dice(gt, remap(_teacher_xy_noov(sess, norm, XY)))
        d = _dice(gt, remap(_teacher_sliding(sess, norm, (32, XY, XY))))
        dt = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pk = rss / (1024 ** 3) if sys.platform == 'darwin' else rss / (1024 ** 2)
        print(f"  {c:<8}{img.shape[2]:>7}{a:>9.4f}{b:>11.4f}{cc:>10.4f}{d:>9.4f}"
              f"{b - a:>+8.4f}{cc - a:>+8.4f}{pk:>8.1f}{dt:>7.0f}")
        sys.stdout.flush()
        rows.append([c, img.shape[2], f"{a:.4f}", f"{b:.4f}", f"{cc:.4f}",
                     f"{d:.4f}", f"{b - a:+.4f}", f"{cc - a:+.4f}", f"{pk:.2f}"])
    _write('grid', ['case', 'inplane_W', 'A_product', 'B_z_overlap_only',
                    'C_xy_block_only', 'D_both', 'delta_B_A', 'delta_C_A',
                    'peak_gb_max'], rows)
    return 0


def teacher():
    """G1：教师（同款 nnU-Net + InstanceNorm，产品同一推理策略）是否也被压低。

    若是，则「0.35M 学生逼近 31.2M 教师」是错的——涨的是两边；而且产品
    ai_engine.py 现在的分割质量本身就被这个策略压着，且产品跑 512² 临床 DICOM，
    比本数据集（面内中位 253）更大，按剂量-反应的方向问题只会更重。
    """
    if '--yes' not in sys.argv:
        print("  这会重跑 ONNX 推理（~分钟级/例）。确认请加 --yes"); return 1
    from seg3d_teacher import make_session, run_onnx
    from seg3d_train import LOBES, prep_case
    sess = make_session()
    print(f"  {'case':<8}{'面内W':>7}{'整幅':>9}"
          + ''.join(f"{'块' + str(x):>9}" for x in TEACHER_XY) + f"{'秒':>7}")
    print("  " + "-" * (24 + 9 * len(TEACHER_XY) + 7))
    rows = []
    for c in ['s0218', 's0182', 's0084']:
        img, gt = prep_case(c)
        hu = img * 1400.0 - 1000.0            # prep_case 已归一化，教师要 HU 输入
        norm = np.clip(hu, -1000, 400).astype(np.float32)
        norm = (norm + 1000.0) / 1400.0
        t0 = time.perf_counter()

        def remap(seg):
            out = np.zeros(seg.shape, np.uint8)
            for k, lid in enumerate(LOBES, start=1):
                out[seg == lid] = k
            return out

        full = _dice(gt, remap(run_onnx(hu, sess=sess, quiet=True)[0]))
        vs = [_dice(gt, remap(_teacher_sliding(sess, norm, (32, xy, xy))))
              for xy in TEACHER_XY]
        dt = time.perf_counter() - t0
        note = '  ← 阴性对照' if img.shape[2] <= 128 else ''
        print(f"  {c:<8}{img.shape[2]:>7}{full:>9.4f}"
              + ''.join(f"{v:>9.4f}" for v in vs) + f"{dt:>7.0f}{note}")
        sys.stdout.flush()
        rows.append([c, img.shape[2], f"{full:.4f}"] + [f"{v:.4f}" for v in vs])
    _write('teacher', ['case', 'inplane_W', 'dice_fullplane']
           + [f"dice_xy{x}" for x in TEACHER_XY], rows)
    return 0





# ===== 产品线：57 例 test 集、三配置、全器官 ==================================
# A = 产品现状（整幅 xy、z 每 32 层无重叠、逐块独立 argmax）
# B = 只改 z：整幅 xy、z 25% 重叠、累加 logits 后 argmax
# D = 两者都改：块 256 + z 25% 重叠、累加
# C（块 xy 无重叠）已在 grid 中证明有害（s0218 −0.166），不是候选，不再跑。
#
# 【为什么要流式累加】全卷 25 类 float32 累加，最大一例（s0086 273×430×430）要
# 5.17GB，加 session 会顶到 8–10GB，正撞上 CLAUDE.md 记着的那条内存墙。
# 但 dz=32、step=24 时，任一 z 位置最多被 2 个块覆盖，只需保留 8 层尾巴：
# 内存从 O(Z) 降到 O(dz)，峰值 <1GB。correctness 由 --verify 对拍全量版保证。

def _zstream(block_fn, Z, Hp, Wp, dz=32, step=24):
    """沿 z 流式融合：block_fn(z0) 返回该 z 块的 (25, dz, Hp, Wp) logits。

    块 k 起于 z0=k*step、覆盖 [z0, z0+dz)。下一块起点是 z0+step，故 z<z0+step
    的位置此后不会再被写入，可以当场定稿；只有末尾 dz-step 层要留给下一块。
    """
    zs = list(range(0, max(1, Z - dz + 1), step))
    if zs[-1] != max(0, Z - dz):
        zs.append(max(0, Z - dz))
    seg = np.zeros((Z, Hp, Wp), np.uint8)
    # 末尾那块为贴边界，与前一块的间隔可能小于 step，此时某些 z 会被三块覆盖。
    # 故保留最近两个 tail，而不是想当然地只留一个；定稿边界取下一块的起点。
    # hist 必须存**未融合的原始 logits**：若把融合结果写回同一数组再入 hist，
    # 下一块会把上上块的贡献重复计入。s0347 末尾两块只隔 2 层，正是靠它暴露的。
    hist = []                                             # [(z0, 原始 logits), …] 最多 2
    for i, z0 in enumerate(zs):
        o = block_fn(z0)                                  # (25, dz, Hp, Wp) 原始
        end = min(zs[i + 1] if i + 1 < len(zs) else Z, Z)
        n = end - z0
        if n > 0:
            fused = o[:, :n].copy()                       # 只复制要定稿的那几层
            for pz, po in hist:
                ov = min(pz + dz, end) - z0               # 该历史块能覆盖到的层数
                if ov > 0:
                    fused[:, :ov] += po[:, z0 - pz:z0 - pz + ov]
            seg[z0:end] = fused.argmax(0).astype(np.uint8)
            del fused
        hist = (hist + [(z0, o)])[-2:]
    return seg


def _infer(sess, norm, mode, xy=256, dz=32):
    """mode: 'A' 产品现状 / 'B' 仅 z 重叠 / 'D' 块 xy + z 重叠。"""
    iname = sess.get_inputs()[0].name
    Z, H, W = norm.shape
    pad_to = 32 if mode != 'D' else xy
    ph, pw = (-H) % pad_to, (-W) % pad_to
    if ph or pw:
        norm = np.pad(norm, ((0, 0), (0, ph), (0, pw)), mode='constant')
    Hp, Wp = norm.shape[1], norm.shape[2]

    if mode == 'A':                                        # 逐块独立 argmax，无融合
        seg = np.zeros((Z, Hp, Wp), np.uint8)
        for z0 in range(0, Z, dz):
            z1 = min(z0 + dz, Z)
            blk = norm[z0:z1]
            pd = (-blk.shape[0]) % 32
            if pd:
                blk = np.pad(blk, ((0, pd), (0, 0), (0, 0)), mode='constant')
            out = sess.run(None, {iname: blk[None, None]})[0][0]
            seg[z0:z1] = out.argmax(0).astype(np.uint8)[:z1 - z0]
            del out
        return seg[:, :H, :W]

    def block_fn(z0):
        blk = norm[z0:z0 + dz]
        if blk.shape[0] < dz:
            blk = np.pad(blk, ((0, dz - blk.shape[0]), (0, 0), (0, 0)), mode='constant')
        if mode == 'B':
            return sess.run(None, {iname: blk[None, None]})[0][0]
        acc = np.zeros((25, dz, Hp, Wp), np.float32)       # D：xy 方向也重叠融合
        sx = max(1, int(xy * 0.75))
        ys = list(range(0, max(1, Hp - xy + 1), sx))
        xs = list(range(0, max(1, Wp - xy + 1), sx))
        for lst, tot in ((ys, Hp), (xs, Wp)):
            if lst[-1] != tot - xy:
                lst.append(tot - xy)
        for y0 in ys:
            for x0 in xs:
                out = sess.run(None, {iname: blk[None, None, :, y0:y0 + xy,
                                                x0:x0 + xy]})[0][0]
                acc[:, :, y0:y0 + xy, x0:x0 + xy] += out
                del out
        return acc

    return _zstream(block_fn, Z, Hp, Wp, dz)[:, :H, :W]


# TotalSegmentator v2 class_map_part_organs，恒等映射（见 seg_validate 实测确证）
ORGANS = {
    1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'liver',
    6: 'stomach', 7: 'pancreas', 8: 'adrenal_gland_right', 9: 'adrenal_gland_left',
    10: 'lung_upper_lobe_left', 11: 'lung_lower_lobe_left', 12: 'lung_upper_lobe_right',
    13: 'lung_middle_lobe_right', 14: 'lung_lower_lobe_right', 15: 'esophagus',
    16: 'trachea', 17: 'thyroid_gland', 18: 'small_bowel', 19: 'duodenum',
    20: 'colon', 21: 'urinary_bladder', 22: 'prostate',
    23: 'kidney_cyst_left', 24: 'kidney_cyst_right',
}


def bench():
    """产品线主实验：test 集全例 × 一个推理配置 × 全部 24 器官。

    【为什么一次只跑一个配置】ru_maxrss 是进程生命期高水位、单调非减，同一进程里
    跑完三个配置只能读到一个累计值——上一轮审计修过这个坑，这里靠「一配置一进程」
    从结构上避免，而不是靠记得。

    【为什么要断点续】单配置数小时，中途任何中断都不该让已算的例子作废。
    每例算完立刻追加一行，重启时跳过 CSV 里已有的 case。
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('bench')
    ap.add_argument('--config', required=True, choices=['A', 'B', 'D'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--yes', action='store_true')
    a = ap.parse_args()
    if not a.yes:
        print("  这会重跑 ONNX 推理（数小时）。确认请加 --yes"); return 1

    import resource

    from seg3d_data import split as make_split
    from seg3d_teacher import make_session
    from seg3d_train import CACHE, HU_CLIP, load_zhw

    out_p = os.path.join(RESULTS, f"seg3d_infer_bias_bench_{a.config}.csv")
    done = set()
    if os.path.exists(out_p):
        with open(out_p, encoding='utf-8-sig') as f:
            done = {r['case'] for r in csv.DictReader(f)}
        print(f"  断点续：{out_p} 已有 {len(done)} 例，跳过")
    else:
        with open(out_p, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['case', 'config', 'organ_id', 'organ',
                                    'dice', 'gt_voxels', 'sec'])

    cases = make_split()[a.split]
    if a.limit:
        cases = cases[:a.limit]
    todo = [c for c in cases if c not in done]
    sess = make_session()
    print(f"  配置 {a.config}  {a.split} 集 {len(cases)} 例，待跑 {len(todo)}\n")

    for i, c in enumerate(todo, 1):
        ip = os.path.join(CACHE, f"{c}_img.nii.gz")
        mp = os.path.join(CACHE, f"{c}_msk.nii.gz")
        if not (os.path.exists(ip) and os.path.exists(mp)):
            print(f"  [{i}/{len(todo)}] {c}: 缺文件，跳过"); continue
        hu = load_zhw(ip).astype(np.float32)
        gt = load_zhw(mp).astype(np.int32)
        norm = ((np.clip(hu, *HU_CLIP) - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])
                ).astype(np.float32)
        del hu
        t0 = time.perf_counter()
        pred = _infer(sess, norm, a.config)
        dt = time.perf_counter() - t0
        del norm

        rows, ds = [], []
        for k, name in ORGANS.items():
            t = (gt == k)
            nt = int(t.sum())
            if not nt:
                continue                      # 该器官不在场：不算 0，不进分母
            p = (pred == k)
            d = 2.0 * int(np.logical_and(t, p).sum()) / (nt + int(p.sum()))
            rows.append([c, a.config, k, name, f"{d:.4f}", nt, f"{dt:.1f}"])
            ds.append(d)
        with open(out_p, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(rows)
        del gt, pred
        print(f"  [{i}/{len(todo)}] {c}  {len(ds)} 器官  均值 {np.mean(ds):.4f}  {dt:.0f}s")
        sys.stdout.flush()

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pk = rss / (1024 ** 3) if sys.platform == 'darwin' else rss / (1024 ** 2)
    print(f"\n  配置 {a.config} 完成。本进程峰值内存 {pk:.2f} GB")
    # 【峰值必须落盘】此前它只 print 到终端，于是 README 里的 8.44 / 9.09 GB
    # 无法由任何已提交产物核验，第三方要验证只能重跑 59 例 × 2 配置。写成
    # 独立 sidecar 而不是加进 bench CSV 的列：那份 CSV 是逐器官一行、且靠
    # 已有 case 集合做断点续跑，加列会让旧文件与新文件的表头不一致。
    # ru_maxrss 是进程生命期高水位，因此这个数只在「一个配置一个进程」时有意义。
    peak_p = os.path.join(RESULTS, 'seg3d_infer_bias_bench_peak.csv')
    new_file = not os.path.exists(peak_p)
    with open(peak_p, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['config', 'split', 'n_cases', 'peak_gb'])
        w.writerow([a.config, a.split, len(cases), f"{pk:.2f}"])
    print(f"    → {os.path.basename(out_p)}  /  {os.path.basename(peak_p)}")
    return 0


def main():
    cmds = {'ab': ab, 'dose': dose, 'pad': pad, 'norm': norm, 'train': train,
            'teacher': teacher, 'grid': grid, 'bench': bench}
    c = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if c == 'all':
        for k in ['pad', 'norm', 'ab', 'dose', 'train']:
            print(f"\n=== {k} ===")
            cmds[k]()
        return 0
    if c not in cmds:
        print(f"  用法：seg3d_infer_bias.py [{'|'.join(cmds)}|all]"); return 1
    return cmds[c]()


if __name__ == '__main__':
    raise SystemExit(main())
