# =============================================================================
# 模型说明卡（model card）——把「这个 AI 有多可信」从 markdown 搬进产品界面
#
# 动机：本项目做了不少别人不做的诚实功课（出处靠实测推断而非元数据、单例 Dice
#       被 57 例修正、幻觉率被真实测量），但这些全写在 README 与技术报告里，
#       打开 GUI 只看得见漂亮的彩色分割。一个只展示成功、不展示适用边界的工具，
#       在科研与临床语境下都不该被信任。
#
# 设计：纯字符串组装，不依赖 Qt——沿用本项目「纯计算抽成无 Qt 模块」的范式，
#       可进 SKIP_REAL_DATA 子集单测。数据一律从已跑出的实验产物读取，
#       读不到就如实说「未提供」，绝不硬编码一个好看的数字。
# =============================================================================

import csv
import json
import os

# 【产物读取的异常捕获为何这么宽】这些文件是实验产物，可能缺失、换过表头、或写到
# 一半被中断。实测踩到两层：csv.DictReader 在列缺失时给出 None，float(None) 抛的是
# **TypeError** 而非 ValueError；文件含 NUL 字节（截断写入的典型形态）时 csv 模块抛
# 的是它自己的 **csv.Error**，不属于任何内置异常类型。少任何一项，用户点一下
# 「模型说明卡」按钮就会崩——而这张卡片的全部意义正是在于可被信任。
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "experiments", "results")


def _read_json(name, require=()):
    """读产物 JSON；文件缺失/损坏，或缺少 require 里任一字段时返回 None。

    【「能解析」不等于「能用」】此前只挡住了完全无法解析的情形（NUL 字节、非法 JSON），
    合法 JSON 但字段名对不上时照样返回 dict，消费端 lobe['overall_mean'] 直接 KeyError
    冲到界面——用户点一下「模型说明卡」就崩。而字段改名正是实验脚本演进时最常见的形态，
    比文件损坏常见得多。require 让 `if lobe:` 这个既有守卫真正兜得住：字段不全就当作
    「产物不在场」，走已经写好的降级文案，而不是半路炸掉。
    """
    try:
        with open(os.path.join(_RESULTS, name), encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError, TypeError, csv.Error):
        return None
    if not isinstance(d, dict) or any(k not in d for k in require):
        return None
    return d


def _read_multi():
    """读多例配对验证产物，返回 (n, 直通均值, 引擎均值, 差值均值, 全改善数)。

    单例的修复效果（s0029 上 +0.064）事后被证明是**最不显著**的一例，20 例上是 +0.155。
    这正是本项目一贯的教训：单例既可能偏乐观，也可能偏保守，方向事先不可知。
    """
    p = os.path.join(_RESULTS, "seg_spacing_fix_multi.csv")
    try:
        with open(p, encoding='utf-8-sig') as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get('case') or '').strip() and not r['case'].strip().startswith('#')]
        pairs = [(float(r['dice_direct']), float(r['dice_engine'])) for r in rows]
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None
    if len(pairs) < 2:
        return None
    dr = [a for a, _ in pairs]; en = [b for _, b in pairs]
    gains = [b - a for a, b in pairs]
    return (len(pairs), sum(dr) / len(dr), sum(en) / len(en),
            sum(gains) / len(gains), sum(1 for g in gains if g > 0))


def _read_json_row(name):
    """读单行 CSV 产物的第一行，返回 dict；文件缺失或为空时返回 None。"""
    try:
        with open(os.path.join(_RESULTS, name), encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        return rows[0] if rows else None
    except (OSError, ValueError, TypeError, csv.Error):
        return None


def _read_spacing():
    """读 spacing 消融产物，返回 (按 spacing 排序的 [(spacing, mean_dice)], 最差的两个器官)。

    卡片里的每一个数字都必须来自产物文件：初版把 0.922→0.881→0.799 直接写死在
    字符串里，与本模块「不硬编码」的原则自相矛盾——重跑消融得到别的数值时，
    卡片会安静地继续显示旧数字。
    """
    p = os.path.join(_RESULTS, "seg_spacing.csv")
    q = os.path.join(_RESULTS, "seg_spacing_per_organ.csv")
    try:
        with open(p, encoding='utf-8-sig') as f:
            pts = sorted((float(r['spacing']), float(r['mean_dice'])) for r in csv.DictReader(f))
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None, []
    if len(pts) < 2:
        return None, []
    worst = []
    try:
        with open(q, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        cols = [c for c in rows[0] if c.startswith('dice@')]
        cols.sort(key=lambda c: float(c[5:-2]))
        for r in rows[:2]:                      # 该文件已按末端 Dice 升序，前两行即最差
            worst.append((r['organ'], [float(r[c]) for c in cols]))
    except (OSError, ValueError, KeyError, IndexError, TypeError, csv.Error):
        worst = []
    return pts, worst


def _read_multi_organ():
    """读多例 21 器官验证产物，返回 (例数, 患者级均值, CI 下界, CI 上界, 最弱器官列表)。

    有它就不再报单例数字：研究二的 0.92 长期是 n=1，而本项目两次看到单例失真且方向
    相反（肺叶单例偏乐观、spacing 修复单例偏保守）。多例在场时以多例为准，单例仅作
    对照保留在文案里。
    """
    p = os.path.join(_RESULTS, "seg_multi.csv")
    q = os.path.join(_RESULTS, "seg_multi_per_organ.csv")
    try:
        with open(p, encoding='utf-8-sig') as f:
            vals = [float(r['mean_dice']) for r in csv.DictReader(f)
                    if (r.get('case') or '').strip() and not r['case'].strip().startswith('#')]
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None
    if len(vals) < 3:
        return None
    n = len(vals)
    mean = sum(vals) / n
    # CI 只从产物的汇总行读，**绝不在这里重算**。初版自己跑了一遍 bootstrap，用的是
    # random.Random 而实验脚本用 np.random.RandomState——同一个 seed、不同的重采样
    # 序列，CI 下界差了 0.001。同一个统计量有两套实现就一定会漂移，而卡片的立足点
    # 正是「每个数字都来自实验产物」。读不到汇总行就不报 CI，也不猜。
    lo = hi = float('nan')
    try:
        with open(p, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                cid = (r.get('case') or '').strip()
                if cid.startswith('#') and 'ci=' in (r.get('min_organ') or ''):
                    a_, b_ = r['min_organ'].split('ci=')[1].split('..')
                    lo, hi = float(a_), float(b_)
                    mean = float(r['mean_dice'])       # 均值同样以产物为准
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        lo = hi = float('nan')
    weak, best = [], None
    try:
        with open(q, encoding='utf-8-sig') as f:
            rows = [r for r in csv.DictReader(f) if int(r['n_cases_present']) >= 3]
        for r in rows:
            if float(r['mean_dice']) < 0.75:
                weak.append((r['organ'], float(r['mean_dice']), int(r['n_cases_present'])))
        if rows:   # 最强器官也必须现读——它同样会进卡片文案，写死一样是硬编码
            b_ = max(rows, key=lambda r: float(r['mean_dice']))
            best = (b_['organ'], float(b_['mean_dice']))
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        weak, best = [], None
    return n, mean, lo, hi, weak[:3], best


def _read_seg_dice():
    """读研究二的逐器官 Dice（单例 s0029）。返回 (在场器官数, 平均 Dice)。"""
    p = os.path.join(_RESULTS, "seg_dice.csv")
    try:
        with open(p, encoding='utf-8-sig') as f:
            ds = [float(r['dice']) for r in csv.DictReader(f)
                  if int(r['our_label']) > 0 and float(r['dice']) > 0]
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None
    return (len(ds), sum(ds) / len(ds)) if ds else None


def build_model_card(is_english=False):
    """组装模型说明卡的 HTML 文本。

    三段固定结构：出处（怎么知道的）→ 验证到什么程度 → 已知局限。
    第三段刻意放在最后且不折叠：适用边界比准确率更该被读到。
    """
    seg = _read_seg_dice()
    mo = _read_multi_organ()
    lobe = _read_json("seg3d_teacher_summary.json",
                      require=("overall_mean", "overall_ci", "n_cases"))
    en = is_english
    L = []

    L.append(f"<b>{'Model provenance' if en else '模型出处'}</b><br>")
    L.append(
        "TotalSegmentator v2 <code>class_map_part_organs</code> (24 organs + background), "
        "nnU-Net v2 PlainConvUNet, 3D full-resolution.<br>"
        "The shipped ONNX carries <i>no</i> metadata naming its origin. The label scheme above was "
        "not read off a label file — it was <b>measured</b>: running the model on a public CT "
        "that ships with ground truth and computing the label-overlap confusion matrix yields a "
        "clean diagonal. What that measures is the <b>mapping</b>, which is all the software uses; "
        "that these weights are that exact upstream release follows from it by inference, not by "
        "any published checksum.<br><br>"
        if en else
        "TotalSegmentator v2 <code>class_map_part_organs</code>（24 器官 + 背景），"
        "nnU-Net v2 PlainConvUNet，3D 全分辨率。<br>"
        "随附的 ONNX <i>没有任何</i>标明出处的元数据。上述标签方案不是从标签文件里读来的，"
        "而是<b>实测</b>出来的：拿一例自带真值的公开 CT 跑推理，与真值算标签重叠混淆矩阵，"
        "得到干净的对角线。被实测的是<b>标签映射</b>，也正是软件唯一依赖的部分；"
        "「这份权重即该上游 release」是由此推出的，没有上游公布的校验和可对。<br><br>")

    L.append(f"<b>{'Measured accuracy' if en else '实测准确度'}</b><br>")
    if mo:
        nc, mm, lo_, hi_, weak, _best = mo
        wtxt = ("；最弱的是 " + "、".join(f"{o} {v:.2f}（{k} 例）" for o, v, k in weak)) if weak else ""
        wen = ("; weakest: " + ", ".join(f"{o} {v:.2f} (n={k})" for o, v, k in weak)) if weak else ""
        L.append(
            f"Multi-organ: patient-level mean Dice <b>{mm:.3f}</b> "
            f"[{lo_:.3f}, {hi_:.3f}] across <b>{nc} cases</b>"
            + (f" (single-case baseline was {seg[1]:.3f})" if seg else "") + f"{wen}.<br>"
            if en else
            f"多器官：<b>{nc} 例</b>患者级平均 Dice <b>{mm:.3f}</b> [{lo_:.3f}, {hi_:.3f}]"
            + (f"（单例基线为 {seg[1]:.3f}）" if seg else "") + f"{wtxt}。<br>")
    elif seg:
        n, mean = seg
        L.append(
            f"Multi-organ: mean Dice <b>{mean:.3f}</b> over {n} organs present "
            f"— <b>on a single case (n=1)</b>.<br>"
            if en else
            f"多器官：{n} 个在场器官平均 Dice <b>{mean:.3f}</b>"
            f"——<b>但这是单例结果（n=1）</b>。<br>")
    else:
        L.append("Multi-organ validation results not found in experiments/results.<br>"
                 if en else "未在 experiments/results 找到多器官验证结果。<br>")
    # overall_ci 必须是可下标的二元组：标量或 null 会让 ci[0] 抛 TypeError。
    if lobe and not (isinstance(lobe['overall_ci'], (list, tuple)) and len(lobe['overall_ci']) >= 2):
        lobe = None
    if lobe:
        m, ci, nc = lobe['overall_mean'], lobe['overall_ci'], lobe['n_cases']
        L.append(
            f"Lung lobes: Dice <b>{m:.3f}</b> [{ci[0]:.3f}, {ci[1]:.3f}] over <b>{nc} cases</b>. "
            f"Note this is <i>lower</i> than the single-case figure — n=1 was optimistic, "
            f"which is exactly why the larger run was done.<br><br>"
            if en else
            f"肺叶：<b>{nc} 例</b>上 Dice <b>{m:.3f}</b> [{ci[0]:.3f}, {ci[1]:.3f}]。"
            f"注意它<i>低于</i>单例结果——单例是偏乐观的，扩样本正是为了纠正这一点。<br><br>")
    else:
        L.append("Lung-lobe multi-case baseline not found.<br><br>"
                 if en else "未找到肺叶多例基线结果。<br><br>")

    L.append(f"<b style='color:#C0392B'>{'Known limitations' if en else '已知局限'}</b><br>")

    fix = _read_json_row("seg_spacing_fix.csv")
    multi = _read_multi()
    pts, worst = _read_spacing()
    if fix and pts:
        # 已按 nnU-Net 契约重采样：这一条从「局限」升级为「已处理，且效果实测」。
        # 仍然留在本段，因为读者需要知道这一步存在、为什么必要、以及还剩什么没验证。
        base_d, end_d = pts[0][1], pts[-1][1]
        loss = 100 * (1 - end_d / base_d)
        # 【倍率也要现算】同段的 base_d / end_d 都从 seg_spacing.csv 实读，唯独倍率曾是
        # 字面量「一倍 / twice」。当前 CSV 是 1.5 → 3.0 恰好成立，但消融范围一改（补测
        # 4.0mm，或补测比 1.5 更细的一档使 pts[0] 不再是训练 spacing），Dice 会跟着更新
        # 而倍率不会——正是 _read_spacing 的 docstring 点名要避免的「重跑得到别的数值时，
        # 卡片安静地继续显示旧数字」。下面的 elif 分支一直是现算的，此处对齐。
        ratio = pts[-1][0] / pts[0][0] if pts[0][0] else float('nan')
        spacing_zh = (
            "<b>1. 体素间距（voxel spacing）已按 nnU-Net 契约重采样——必要性与效果均实测。</b>"
            f"不做这一步的代价：spacing 偏离训练值 {ratio:g} 倍时平均 Dice 由 {base_d:.3f} 掉到 "
            f"{end_d:.3f}（−{loss:.0f}%）。现推理前先还原到训练 spacing，同一份失配输入下 "
            + (f"{multi[0]} 例配对验证下 Dice 由 <b>{multi[1]:.3f} 回升到 {multi[2]:.3f}</b>"
               f"（平均 {multi[3]:+.3f}，{multi[4]}/{multi[0]} 例全部改善）。"
               if multi else
               f"Dice 由 <b>{fix['dice_direct']} 回升到 {fix['dice_engine']}</b>"
               f"（找回差距的 {fix['pct_of_gap_recovered']}%，仅 1 例）。")
            + "<b>仍未直接验证的是更细 spacing 一侧</b>——该侧是降采样、信息本就充足，"
            "原理上比已测的变粗侧更有利，但没有带真值的细 spacing 数据可算 Dice。"
            "<b>这一步也有代价，不是纯赚</b>：蒙版边界是在 1.5mm 网格上决定的，映射回更细的"
            "原分辨率后呈阶梯状（0.713mm 数据上实测边界平台约 2 像素）——结构级准确度上升，"
            "而像素级边界精度下降。另：扫描范围过大时重采样会被跳过以免内存溢出，"
            "此时 spacing 失配依旧存在。<br>")
        spacing_en = (
            "<b>1. Voxel spacing is now resampled per nnU-Net's contract — both the need and "
            "the effect are measured.</b> Skipping it costs real accuracy: mean Dice falls "
            f"{base_d:.3f} → {end_d:.3f} (−{loss:.0f}%) at {ratio:g}× the training spacing. With "
            + (f"resampling in place, mean Dice across {multi[0]} paired cases goes <b>{multi[1]:.3f} → "
             f"{multi[2]:.3f}</b> ({multi[3]:+.3f} on average, improving in {multi[4]}/{multi[0]}). "
             if multi else
             f"resampling in place, the same mismatched input recovers from <b>{fix['dice_direct']} "
             f"to {fix['dice_engine']}</b> ({fix['pct_of_gap_recovered']}% of the gap, n=1). ")
            + "<b>The finer-spacing side is still not directly validated</b> — that direction is "
            "downsampling, where information is already sufficient, so it should fare better "
            "than the coarser side measured here, but no ground-truth fine-spacing case was "
            "available. <b>The step is not free</b>: mask boundaries are decided on the 1.5 mm "
            "grid and become stair-stepped when mapped back to a finer original resolution "
            "(≈2-pixel plateaus measured on 0.713 mm data) — structural accuracy up, "
            "pixel-level boundary precision down. Resampling is also skipped for very large "
            "scan ranges to avoid running out of memory, leaving the mismatch in place there.<br>")
    elif pts:
        base_sp, base_d = pts[0]
        end_sp, end_d = pts[-1]
        curve = " → ".join(f"{d:.3f}" for _, d in pts)
        sps = " / ".join(f"{s:g}" for s, _ in pts)
        loss = 100 * (1 - end_d / base_d)
        ratio = end_sp / base_sp
        detail = "; ".join(f"{o} {' → '.join(f'{v:.2f}' for v in ds)}" for o, ds in worst)
        spacing_zh = (
            "<b>1. 体素间距（voxel spacing）已按契约重采样——不做的代价已实测，不是推测。</b>"
            "nnU-Net 的推理契约第一步是重采样到训练 spacing，本工具现已这样做"
            "（<code>ai_engine.TARGET_SPACING</code>）。下面这条曲线是【不做】的代价："
            f"在同一例带真值数据上的消融（<code>experiments/seg_spacing.py</code>）显示，"
            f"{sps}mm 下平均 Dice 依次为 <b>{curve}</b>——"
            f"<b>spacing 变为 {ratio:g} 倍时掉 {loss:.0f}%</b>。"
            + (f"小器官最先垮且并非单调（{detail}）。" if detail else "")
            + f"受内存所限只测得到变粗方向，比 {base_sp:g}mm 更细的 spacing"
              "（如本机 RIDER 数据的 0.713mm）推理需 50GB 以上，仍属<b>未测量</b>。<br>")
        spacing_en = (
            "<b>1. Voxel spacing is resampled per nnU-Net's contract — the cost of "
            "<i>not</i> doing so is measured, not hypothetical.</b> The contract begins by "
            "resampling to the training spacing, and this tool now does "
            "(<code>ai_engine.TARGET_SPACING</code>). The curve below is what skipping it costs. "
            "An ablation on the same ground-truth case "
            f"(<code>experiments/seg_spacing.py</code>) shows mean Dice going <b>{curve}</b> at "
            f"{sps} mm — a <b>{loss:.0f}% loss at {ratio:g}× the training spacing</b>. "
            + (f"Small structures fail first and not monotonically ({detail}). " if detail else "")
            + f"Only the coarser direction was testable; spacings finer than {base_sp:g} mm "
              "(such as the 0.713 mm RIDER series) need >50 GB to infer and remain "
              "<b>unmeasured</b>.<br>")
    else:
        # 【产物缺失≠产品没做这件事】此前这两句写的是「本工具直接按原始 spacing 送入」，
        # 而 ai_engine.TARGET_SPACING 早已实现重采样——读不到 CSV 是一次 I/O 事件，
        # 不能据此对产品当前行为下断言。这里只说消融数据不在场。
        spacing_zh = ("<b>1. 体素间距（voxel spacing）的代价尚未在本机测量。</b>本工具已按 "
                      "nnU-Net 契约把体积重采样到训练 spacing（<code>ai_engine.TARGET_SPACING"
                      "</code>）；但这一步究竟值多少 Dice、哪些器官最先垮，需跑 "
                      "<code>experiments/seg_spacing.py</code> 才有数——本机尚未测量。<br>")
        spacing_en = ("<b>1. The cost of voxel spacing has not been measured here.</b> This tool "
                      "does resample to nnU-Net's training spacing "
                      "(<code>ai_engine.TARGET_SPACING</code>); what that step is worth in Dice, "
                      "and which organs fail first without it, comes from "
                      "<code>experiments/seg_spacing.py</code>, which has not been run here.<br>")

    # 第 2 条随样本量状态而变：多例结果在场时，"n=1" 已不成立，此处必须换成仍然
    # 成立的限制——否则卡片上半段写着 20 例、下半段还挂着 n=1，自相矛盾（截图时发现）。
    if mo:
        nc2, weak2, best2 = mo[0], mo[4], mo[5]
        spread = (f"（最强 {best2[0]} {best2[1]:.2f} 与最弱 {weak2[0][0]} {weak2[0][1]:.2f}，"
                  f"相差 {best2[1] - weak2[0][1]:.2f}）") if (weak2 and best2) else ""
        limit2_zh = (f"<b>2. {nc2} 例仍是小样本，且器官间差异远大于总体数字所示。</b>"
                     f"全部来自同一个公开数据集（TotalSegmentator-CT-Lite，1.5mm 各向同性），"
                     f"未覆盖其他扫描协议与设备{spread}。总体 Dice 说明不了<i>某一个</i>器官"
                     f"在<i>你这一例</i>上是否可信。<br>")
        limit2_en = (f"<b>2. {nc2} cases is still a small sample, and per-organ reliability "
                     f"varies far more than the aggregate.</b> All cases come from one public "
                     f"dataset (TotalSegmentator-CT-Lite, 1.5 mm isotropic); other protocols and "
                     f"scanners are untested"
                     + (f" ({best2[0]} {best2[1]:.2f} vs {weak2[0][0]} {weak2[0][1]:.2f})" if (weak2 and best2) else "")
                     + ". An aggregate Dice says little about whether "
                     "<i>a particular</i> organ is trustworthy in <i>your</i> study.<br>")
    else:
        limit2_zh = "<b>2. 多器官那个数字是 n=1。</b>一例样本撑不起普适结论。<br>"
        limit2_en = ("<b>2. The multi-organ figure is n=1.</b> One case cannot support a "
                     "general claim.<br>")

    # 【方位维度必须在卡片上出现】上面每一个 Dice 都产自 experiments 的 RAS 输入路径，
    # 而产品从 DICOM 读入时面内两轴与模型相反。2026-08-27 之前不翻转，成对器官标签整体
    # 互换（实测肺叶五标签 Dice 全 0.000、肝 0.181）；修复后有回归测试，但上列数字并未
    # 在产品自己的 DICOM 路径上重测。卡片是直接展示给用户的，这条不能只写在仓库文档里。
    axis_en = ("<b>3. These Dice figures were measured on RAS input.</b> The product reads DICOM, "
               "whose two in-plane axes run opposite to the model's. Volumes are flipped as a pair "
               "before inference (corrected 2026-08-27, covered by a regression test), but the "
               "numbers above come from the experiments' RAS path and have not been re-measured on "
               "the product's own path.<br>")
    axis_zh = ("<b>3. 上列 Dice 测自 RAS 方位的输入。</b>产品从 DICOM 读入，其面内两轴与模型相反；"
               "推理前后已成对翻转（2026-08-27 修正，有回归测试覆盖），但上述数字来自 experiments "
               "的 RAS 路径，并未在产品自己的 DICOM 路径上重测。<br>")
    L.append(
        spacing_en + limit2_en + axis_en +
        "<b>4. Not a medical device.</b> Educational and research use only; never for diagnosis."
        if en else
        spacing_zh + limit2_zh + axis_zh +
        "<b>4. 非医疗器械。</b>仅供教学与科研，绝不可用于诊断。")
    return "".join(L)


def card_title(is_english=False):
    return "Model card — provenance & limits" if is_english else "模型说明卡 —— 出处与适用边界"
