"""按病例聚类重算总体 Dice 的 95% 置信区间（研究四的诚实性修正）。

问题：seg3d_teacher.py / seg3d_eval.py 的 `overall_ci` 把所有 (病例, 肺叶) 行摊平成
一个列表后做 i.i.d. 自助重采样。但同一病例的 5 个肺叶不是独立观测——它们共享扫描
质量、spacing、病理与标注者。有效样本量接近病例数而非叶次数，故 i.i.d. 口径给出的
区间**系统性偏窄**。

正确做法是**聚类（按病例）自助法**：以放回方式抽 N 个病例，中签病例带上它全部的
肺叶行，再求均值。理论依据见 Field & Welsh, J. R. Statist. Soc. B 69(3):369-390
(2007)——cluster bootstrap 在 transformation 与 random-effect 两种模型下都相合，
而 residual（此处即 i.i.d.）只在前者下相合。

本脚本**只读输入 CSV，不做任何推理**，结果写入 `results/cluster_ci.json`。
该文件是**已提交产物**，重跑会覆写它——早先的说明写作「不覆盖任何既有产物」，
那是错的，并且掩盖了一个真实缺陷：`seg3d_student_ch8.csv` 被 .gitignore 排除，
在任何干净 clone 上都不存在，而旧实现遇缺失只 print 一句「跳过」便继续，随后把
只剩 3 个键的结果覆盖到已提交的 4 键文件上。于是**任何照文档执行复现的人都会
静默销毁一份已提交证据**。现在的契约是：
  · 必需输入缺失 → 立即失败退出，不写任何东西；
  · 输入 CSV 缺列或无有效行 → 立即失败退出（此前是 KeyError / ValueError 裸抛）；
  · 本地专有输入（local_only）缺失 → 跳过该条，但随后拒绝缩小既有产物；
  · 既有目标不可读 / 不是 JSON / schema 不对 → 立即失败退出。**这条是补上去的**：
    原先 `except (OSError, ValueError): prev_keys = set()` 把坏文件当成「没有既有
    产物」，于是**恰恰在文件已损坏时把上面那道防缩小的闸门整个关掉**，实测干净
    clone + 半截 JSON 会静默从 4 键覆写成 3 键并 exit 0；
  · 写盘前比对键集，只允许保持或新增，绝不允许减少；
  · 落盘走同目录临时文件 → flush → fsync → os.replace 原子替换。原先直接
    `open(dest, "w")`，该调用**先截断后写**，写到一半失败就把已提交产物留成 0 字节。

用法（须在 dicom_gui 环境内）：python experiments/cluster_ci.py
"""
import csv
import json
import os
import tempfile

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# 四个带 (case, label, dice) 结构的输入。local_only=True 表示该产物被 .gitignore
# 排除、只存在于产出它的那台机器上——干净 clone 必然缺它，这不是错误，但会让输出
# 不完整，故写盘时由键集比对兜住。
TARGETS = [("seg3d_teacher_dice.csv", False),
           ("seg3d_student_ch8.csv", True),
           ("seg3d_student_ch8d3_33600s_sliding.csv", False),
           ("seg3d_student_ch8d3_33600s_zslab.csv", False)]


def _read(path):
    """读 (case, dice) 对；跳过非有限值。CSV 带 BOM，故用 utf-8-sig。

    缺列或一行有效数据都没有时**明确失败**，不返回空表：空表会一路走到
    np.mean([]) → nan 与 np.concatenate([]) → ValueError，前者把 nan 写进已提交
    产物，后者裸抛一个与病因无关的 traceback。两种都比不上在此处指名道姓。
    """
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = set(rd.fieldnames or ())
        missing = sorted({"case", "dice"} - cols)
        if missing:
            raise SystemExit(f"输入 {os.path.basename(path)} 缺列 {missing}"
                             f"（实有 {sorted(cols)}）——已中止，未写出任何文件。")
        for r in rd:
            try:
                d = float(r["dice"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(d):
                rows.append((r["case"], d))
    if not rows:
        raise SystemExit(f"输入 {os.path.basename(path)} 没有任何有效的有限 dice 行"
                         f"——已中止，未写出任何文件。")
    return rows


def _prev_keys(dest):
    """读既有产物的键集；不可读 / 非 JSON / schema 不符一律失败退出。

    【绝不把坏文件当成没有文件】防缩小的闸门只在这里拿得到 prev_keys 时才生效。
    早先这里吞掉 OSError/ValueError 并回落到空集，等于在文件已经损坏的情况下
    自动放行覆写——最需要保护的那一刻反而没有保护。
    """
    if not os.path.exists(dest):
        return None
    try:
        with open(dest, encoding="utf-8") as f:
            prev = json.load(f)
    except OSError as e:
        raise SystemExit(f"已有 {dest} 无法读取（{e}）——已中止，该文件逐字节未动。") from None
    except ValueError as e:
        raise SystemExit(f"已有 {dest} 不是合法 JSON（{e}）——已中止，该文件逐字节未动。\n"
                         f"请先人工确认它是否已损坏；本脚本不会覆盖读不懂的既有产物。") from None
    if not isinstance(prev, dict) or not all(isinstance(v, dict) for v in prev.values()):
        raise SystemExit(f"已有 {dest} 的 schema 不是 {{产物名: {{指标}}}}"
                         f"（顶层为 {type(prev).__name__}）——已中止，该文件逐字节未动。")
    return set(prev)


def _atomic_write_json(dest, obj):
    """同目录临时文件 → flush → fsync → os.replace。失败则清掉临时文件，目标不动。

    同目录是硬要求：os.replace 跨文件系统会抛 OSError，而 /tmp 与仓库常不同卷。
    """
    d = os.path.dirname(os.path.abspath(dest))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".cluster_ci.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        # 目标此刻仍是原样：截断只会发生在临时文件上
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    for name, local_only in TARGETS:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            if not local_only:
                # 必需输入缺失属环境异常，继续算出来的是残缺结果，不得落盘
                raise SystemExit(f"缺少必需输入 {name}（{path}）——已中止，未写出任何文件。")
            print(f"  跳过（本地专有产物，未入库）：{name}")
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
    # 【绝不缩小已提交产物】干净 clone 上 seg3d_student_ch8.csv 必然缺失，此时 out
    # 只有 3 个键。若照写，就把已提交的 4 键文件删掉了一条——而且当事人毫不知情。
    prev_keys = _prev_keys(dest)          # 读不懂就在这里退出，不会走到写盘
    if prev_keys is not None:
        lost = sorted(prev_keys - set(out))
        if lost:
            raise SystemExit(
                f"拒绝覆盖 {dest}：现有文件含本次未能产出的条目 {lost}。\n"
                f"这些条目的输入是本地专有产物（见 TARGETS 的 local_only），"
                f"在干净 clone 上不存在。已中止，既有文件逐字节未动。")
    _atomic_write_json(dest, out)
    print(f"\n已写出 {dest}（覆写该已提交产物；键集经比对未减少）")


if __name__ == "__main__":
    main()
