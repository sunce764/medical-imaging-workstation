# =============================================================================
# AI 分割量化验证：用 TotalSegmentator 标注数据测 organs.onnx 的 Dice 并"测出"标签映射
# ---------------------------------------------------------------------------
# 动机：organs.onnx 的 25 类 label→器官 映射一直是推断的（无官方 dataset.json）。
#       本脚本在一例带真值标注的公开 CT（TotalSegmentator-CT-Lite，CC-BY-4.0）上，
#       用 GUI 所用的同一套预处理+滑窗推理跑模型，再与真值逐标签算重叠，
#       从而 (1) 客观量化分割质量(Dice)，(2) 用混淆矩阵"测出"而非"猜出"标签映射。
#
# 数据不入库（需自备）。获取方式（HTTP Range 只抽一例，约 42MB）：
#   from remotezip import RemoteZip
#   base="https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite/resolve/main"
#   RemoteZip(base+"/Images.zip").read("Images/s0029.nii.gz")  -> s0029_img.nii.gz
#   RemoteZip(base+"/Masks.zip").read("Masks/s0029.nii.gz")    -> s0029_msk.nii.gz
#
# 复现性限制（如实记录）：上面 URL 用的是 /resolve/main —— main 是【可变分支引用】。
#   上游若更新数据集，重抓到的内容可能与产出 results/seg_dice.csv 时所用的不是同一份，
#   而本仓库未记录当时那两个 .nii.gz 的校验和（原始文件已不在本地，无法事后补记）。
#   要严格复现，请改用固定 revision（HF 支持 /resolve/<commit-sha>/...）并记录所得文件
#   的 SHA256 后再与本仓库结果比对；已提交的 Dice 数值对应的是当时 main 分支的内容。
#
# 用法：
#   python experiments/seg_validate.py <img.nii.gz> <mask.nii.gz>
# 产出：experiments/results/ 下 seg_confusion.png + seg_dice.csv + seg_mapping.md
# =============================================================================

import csv
import os
import sys

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from constants import MODEL_PATH  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

# TotalSegmentator v2 "total"（117 类）中与本模型相关的胸腹器官整数→名称，
# 用于把真值标签翻成器官名（其 1-21 与官方 class_map_part_organs 完全一致）。
TS_TOTAL = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_R", 9: "adrenal_L",
    10: "lung_upper_L", 11: "lung_lower_L", 12: "lung_upper_R",
    13: "lung_middle_R", 14: "lung_lower_R", 15: "esophagus", 16: "trachea",
    17: "thyroid", 18: "small_bowel", 19: "duodenum", 20: "colon",
    21: "urinary_bladder",
    # 22-24 在原表中缺失，导致多例统计里这三类只显示为 "?"（本例真值恰无它们，
    # 单例研究里没暴露）。身份由已实测的 class_map_part_organs 映射方案确定。
    22: "prostate", 23: "kidney_cyst_L", 24: "kidney_cyst_R",
    51: "heart", 52: "aorta",
}


def load_zhw(path):
    """载入 nii，规范到 RAS，再转成 GUI 的 (Z=上下, H=前后, W=左右) 轴序。"""
    v = nib.as_closest_canonical(nib.load(path))   # -> RAS: (R, A, S)
    arr = np.asanyarray(v.dataobj)
    return np.transpose(arr, (2, 1, 0))            # (S, A, R) = (Z, H, W)


def run_onnx(volume_hu):
    """复刻 ai_engine._run_onnx_multiorgan：clip[-1000,400] 归一化 + 沿 z 的 DZ=32 滑窗。

    【与当前产品的已知差异，勿再写作「严格复刻」】产品自 2a50e37 起把末窗回移到
    [Z-DZ, Z)（原先末块由 pad(mode='constant') 补零，而 HU 归一化后 0 就是空气；
    实测 194 层时末块只有 2 层真实数据、30 层合成空气）。本函数**仍是回移前的写法**，
    故本脚本产出的是**历史（修复前）证据**，差异只落在末块。具体受此限定的已提交
    产物是 seg_dice、seg_mapping、seg_confusion 三份——它们的 accuracy 只能称作
    historical evidence，不得写成对当前 shipped path 的验证。有意不改：既有结果全部
    在这条路径上产出，改它会让已提交的 CSV 与脚本对不上，而重跑需要 ONNX 推理。
    """
    import onnxruntime as ort
    norm = np.clip(volume_hu, -1000, 400).astype(np.float32)
    norm = (norm + 1000.0) / 1400.0
    Z, H, W = norm.shape
    so = ort.SessionOptions(); so.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ph, pw = (-H) % 32, (-W) % 32
    seg = np.zeros((Z, H, W), dtype=np.uint8)
    DZ = 32
    for z0 in range(0, Z, DZ):
        z1 = min(z0 + DZ, Z)
        blk = norm[z0:z1]
        pd = (-blk.shape[0]) % 32
        if pd or ph or pw:
            blk = np.pad(blk, ((0, pd), (0, ph), (0, pw)), mode="constant")
        out = sess.run(None, {iname: blk[np.newaxis, np.newaxis]})[0][0]
        seg[z0:z1] = out.argmax(0).astype(np.uint8)[:z1 - z0, :H, :W]
        del out
        print(f"    推理 z {z1}/{Z}", end="\r")
    print()
    return seg


def dice(a_mask, b_mask):
    inter = int(np.logical_and(a_mask, b_mask).sum())
    s = int(a_mask.sum() + b_mask.sum())
    return (2.0 * inter / s) if s else 0.0


def main():
    if len(sys.argv) < 3:
        print("用法: python experiments/seg_validate.py <img.nii.gz> <mask.nii.gz>")
        sys.exit(1)
    img_path, msk_path = sys.argv[1], sys.argv[2]
    print(f"载入影像 {os.path.basename(img_path)} / 真值 {os.path.basename(msk_path)}")
    vol = load_zhw(img_path)
    gt = load_zhw(msk_path)
    assert vol.shape == gt.shape, f"影像与真值尺寸不一致 {vol.shape} vs {gt.shape}"
    print(f"体积 {vol.shape}  HU[{vol.min():.0f},{vol.max():.0f}]")

    # 推理结果缓存到影像同目录，避免调试出图时重复 ~3min 的 CPU 推理
    cache = img_path + ".pred.npy"
    if os.path.exists(cache):
        pred = np.load(cache)
        print(f"复用缓存推理 {os.path.basename(cache)}")
    else:
        print("运行 organs.onnx（GUI 同款滑窗推理）...")
        pred = run_onnx(vol)
        np.save(cache, pred)
    our_labels = [int(u) for u in np.unique(pred) if u != 0]
    print(f"模型输出标签: {our_labels}")

    # === 混淆矩阵：每个"我们的标签"内部，真值各器官的体素占比 → 测出解剖身份 ===
    gt_ids = [g for g in TS_TOTAL if (gt == g).any()]
    rows_csv = [["our_label", "best_gt_id", "best_gt_organ", "dice", "iou",
                 "our_voxels", "gt_voxels"]]
    mapping = []
    conf = np.zeros((len(our_labels), len(gt_ids)), dtype=np.float32)  # 行=我们标签, 列=真值器官, 值=Dice
    for i, u in enumerate(our_labels):
        um = (pred == u)
        un = int(um.sum())
        best = (0, None, 0.0, 0.0)
        for j, g in enumerate(gt_ids):
            gm = (gt == g)
            d = dice(um, gm)
            conf[i, j] = d
            inter = int(np.logical_and(um, gm).sum())
            union = int(np.logical_or(um, gm).sum())
            iou = inter / union if union else 0.0
            if d > best[2]:
                best = (g, TS_TOTAL[g], d, iou)
        g, gname, d, iou = best
        if gname is None:   # 本例真值未包含该器官（如 prostate/kidney_cyst 在此例缺席）
            gname = "(absent in GT)"
        rows_csv.append([u, g, gname, round(d, 4), round(iou, 4), un,
                         int((gt == g).sum()) if g else 0])
        mapping.append((u, gname, d))
        print(f"    our#{u:2d} -> {gname:16s} (GT#{g})  Dice={d:.3f}  IoU={iou:.3f}  vox={un}")

    # CSV
    with open(os.path.join(RESULTS, "seg_dice.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_csv)
    print(f"  wrote {os.path.join(RESULTS, 'seg_dice.csv')}")

    # 混淆热图
    fig, ax = plt.subplots(figsize=(max(7, len(gt_ids) * 0.5), max(5, len(our_labels) * 0.42)))
    im = ax.imshow(conf, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gt_ids))); ax.set_xticklabels([TS_TOTAL[g] for g in gt_ids], rotation=90, fontsize=8)
    ax.set_yticks(range(len(our_labels))); ax.set_yticklabels([f"our#{u}" for u in our_labels], fontsize=8)
    ax.set_xlabel("TotalSegmentator ground-truth organ"); ax.set_ylabel("organs.onnx output label")
    ax.set_title("Label overlap (Dice) — recovering the organs.onnx label map")
    for i in range(len(our_labels)):
        for j in range(len(gt_ids)):
            if conf[i, j] > 0.15:
                ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                        color="white" if conf[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, label="Dice")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "seg_confusion.png"), dpi=140); plt.close(fig)
    print(f"  wrote {os.path.join(RESULTS, 'seg_confusion.png')}")

    # 恢复出的映射（markdown）
    good = [m for m in mapping if m[2] >= 0.2]
    with open(os.path.join(RESULTS, "seg_mapping.md"), "w") as f:
        f.write("# organs.onnx 标签映射（由 TotalSegmentator 真值实测恢复）\n\n")
        f.write("数据: 单例 TotalSegmentator-CT-Lite（1.5mm 各向同性，thorax-abdomen-pelvis）。\n")
        f.write("方法: GUI 同款滑窗推理 → 与真值逐标签 Dice，取每个输出标签重叠最大的真值器官。\n\n")
        f.write("| our label | 实测器官 | Dice |\n|---|---|---|\n")
        for u, name, d in mapping:
            flag = "" if d >= 0.2 else "  ·低置信"
            f.write(f"| {u} | {name}{flag} | {d:.3f} |\n")
        f.write(f"\n可信匹配(Dice≥0.2): {len(good)}/{len(mapping)}。\n")
        md = float(np.mean([m[2] for m in mapping])) if mapping else 0.0
        f.write(f"平均 Dice: {md:.3f}。\n")
    print(f"  wrote {os.path.join(RESULTS, 'seg_mapping.md')}")
    print("完成。")


if __name__ == "__main__":
    main()
