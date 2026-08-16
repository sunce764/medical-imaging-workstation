# =============================================================================
# 研究四（训练）：轻量 3D U-Net 分割 5 个肺叶，对标 TotalSegmentator
# ---------------------------------------------------------------------------
# 问题：organs.onnx 是 31.2M 参数、整卷推理 ~99s（实测 1.62 μs/体素）、峰值 4–8.8GB。
#       把模型压小，Dice 掉多少？这个交换在什么场景可接受？
#
# 【为什么这个问题值得问】本项目的 GUI 曾因整卷推理 OOM 而不得不沿 z 分块 + 关闭
#       CPU 内存池来压峰值——那是**绕过**问题。本研究从根上换掉它，并把代价量化。
#
# 【训练标签用真值，不用教师输出】
#   用 TotalSegmentator 的预测当标签（知识蒸馏）也能训，但那样学生的 Dice 衡量的是
#   「模仿教师有多像」，不是「分割有多准」——教师自身相对真值也不是 1.0，误差会被
#   静默吞掉。本研究训练与验证**一律用数据集自带的真值标注**。
#
# 用法：
#   python experiments/seg3d_train.py --ch 8            # 一个容量配置
#   python experiments/seg3d_train.py --ch 8 --epochs 30 --steps 120
# 产出：results/seg3d_w{ch}.pt（权重，已 gitignore）+ 训练日志
# =============================================================================

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CACHE = os.path.join(HERE, ".seg3d_cache")
RESULTS = os.path.join(HERE, "results")

# 5 个肺叶 → 网络输出 6 类（含背景）。真值标签 10–14 重映射到 1–5。
LOBES = [10, 11, 12, 13, 14]
N_CLASS = len(LOBES) + 1
HU_CLIP = (-1000.0, 400.0)          # 与 ai_engine 的预处理一致
PATCH = (32, 128, 128)              # (z, y, x)。z 取 32 与教师的滑窗块高一致


def load_zhw(path):
    """载入 nii → RAS → (Z, H, W)，与 GUI 及教师基线同一轴序。"""
    import nibabel as nib
    v = nib.as_closest_canonical(nib.load(path))
    return np.transpose(np.asanyarray(v.dataobj), (2, 1, 0))


def prep_case(cid):
    """返回 (归一化影像 float32, 重映射标签 uint8)。标签 10–14 → 1–5，其余归 0。"""
    img = load_zhw(os.path.join(CACHE, f"{cid}_img.nii.gz")).astype(np.float32)
    gt = load_zhw(os.path.join(CACHE, f"{cid}_msk.nii.gz")).astype(np.int32)
    img = np.clip(img, *HU_CLIP)
    img = (img - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])
    lab = np.zeros(gt.shape, np.uint8)
    for k, lid in enumerate(LOBES, start=1):
        lab[gt == lid] = k
    return img, lab


class PatchSampler:
    """按病例惰性加载并缓存，按 patch 采样。

    【前景过采样】肺叶只占体积的一小部分，纯随机采 patch 会有大半是全背景，
    梯度几乎全来自背景，模型很快学会「全预测背景」——Dice 却因背景占比高而
    看着不差。故一半 patch 强制以某个肺叶体素为中心。
    """

    def __init__(self, cases, seed=0, cache_n=8, fg_ratio=0.5):
        self.cases = list(cases)
        self.rng = np.random.RandomState(seed)
        self.cache, self.cache_n, self.fg_ratio = {}, cache_n, fg_ratio

    def _get(self, cid):
        if cid not in self.cache:
            if len(self.cache) >= self.cache_n:          # 简单 FIFO，控内存
                self.cache.pop(next(iter(self.cache)))
            self.cache[cid] = prep_case(cid)
        return self.cache[cid]

    def sample(self, n):
        xs, ys = [], []
        pz, py, px = PATCH
        while len(xs) < n:
            cid = self.cases[self.rng.randint(len(self.cases))]
            img, lab = self._get(cid)
            Z, H, W = img.shape
            if Z < pz or H < py or W < px:               # 体积小于 patch，跳过
                continue
            if self.rng.rand() < self.fg_ratio and lab.any():
                idx = np.argwhere(lab > 0)
                cz, cy, cx = idx[self.rng.randint(len(idx))]
                z0 = int(np.clip(cz - pz // 2, 0, Z - pz))
                y0 = int(np.clip(cy - py // 2, 0, H - py))
                x0 = int(np.clip(cx - px // 2, 0, W - px))
            else:
                z0 = self.rng.randint(Z - pz + 1)
                y0 = self.rng.randint(H - py + 1)
                x0 = self.rng.randint(W - px + 1)
            xs.append(img[z0:z0 + pz, y0:y0 + py, x0:x0 + px])
            ys.append(lab[z0:z0 + pz, y0:y0 + py, x0:x0 + px])
        return (np.stack(xs)[:, None].astype(np.float32),
                np.stack(ys).astype(np.int64))


def build_net(ch):
    import torch.nn as nn

    def blk(ci, co):
        return nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1), nn.InstanceNorm3d(co, affine=True),
            nn.LeakyReLU(0.01, True),
            nn.Conv3d(co, co, 3, padding=1), nn.InstanceNorm3d(co, affine=True),
            nn.LeakyReLU(0.01, True))

    import torch

    class UNet3D(nn.Module):
        """两次下采样的精简 3D U-Net。层数固定，只调通道数 ch 来做容量权衡。"""

        def __init__(self, c):
            super().__init__()
            self.e1, self.e2, self.e3 = blk(1, c), blk(c, c * 2), blk(c * 2, c * 4)
            self.u2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2); self.d2 = blk(c * 4, c * 2)
            self.u1 = nn.ConvTranspose3d(c * 2, c, 2, 2);     self.d1 = blk(c * 2, c)
            self.out = nn.Conv3d(c, N_CLASS, 1)
            self.pool = nn.MaxPool3d(2)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)

    return UNet3D(ch)


def dice_loss(logits, target, eps=1.0):
    """软 Dice（不含背景类）+ 交叉熵，是分割的标准组合。

    只用 CE 会被背景主导（肺叶体素占比低）；只用 Dice 在早期梯度不稳。
    """
    import torch.nn.functional as F
    p = F.softmax(logits, 1)
    t = F.one_hot(target, N_CLASS).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (p * t).sum(dims)[1:]                 # [1:] 跳过背景
    denom = p.sum(dims)[1:] + t.sum(dims)[1:]
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch', type=int, default=8, help='基础通道数，决定容量')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--steps', type=int, default=120, help='每 epoch 的 step 数')
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    from seg3d_data import split as make_split
    sp = make_split()
    tr = PatchSampler(sp['train'], seed=a.seed)
    va = PatchSampler(sp['val'], seed=a.seed + 1, fg_ratio=0.5)

    net = build_net(a.ch).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    print(f"  设备={dev}  容量 ch={a.ch}  参数 {npar/1e6:.3f}M  "
          f"patch={PATCH}  {a.epochs}ep × {a.steps}step × bs{a.bs}")
    print(f"  训练 {len(sp['train'])} 例 / 验证 {len(sp['val'])} 例（患者级划分）\n")

    best, t0 = {'dice': -1.0}, time.perf_counter()
    for ep in range(1, a.epochs + 1):
        net.train(); tot = 0.0
        te = time.perf_counter()
        for _ in range(a.steps):
            x, y = tr.sample(a.bs)
            xt = torch.from_numpy(x).to(dev); yt = torch.from_numpy(y).to(dev)
            opt.zero_grad()
            lo = net(xt)
            loss = dice_loss(lo, yt) + F.cross_entropy(lo, yt)
            loss.backward(); opt.step()
            tot += float(loss)
        sch.step()
        # 验证：patch 级 Dice（整卷 Dice 在 seg3d_eval.py 里做，那才是可与教师比的口径）
        net.eval(); ds = []
        with torch.no_grad():
            for _ in range(12):
                x, y = va.sample(a.bs)
                pr = net(torch.from_numpy(x).to(dev)).argmax(1).cpu().numpy()
                for k in range(1, N_CLASS):
                    pk, tk = (pr == k), (y == k)
                    s = pk.sum() + tk.sum()
                    if s:
                        ds.append(2.0 * (pk & tk).sum() / s)
        vd = float(np.mean(ds)) if ds else 0.0
        if vd > best['dice']:
            best = {'dice': vd, 'ep': ep,
                    'state': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        print(f"  ep{ep:>3}/{a.epochs}  loss={tot/a.steps:.4f}  val patch-Dice={vd:.4f}"
              f"  {time.perf_counter()-te:.0f}s")
        sys.stdout.flush()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"seg3d_w{a.ch}.pt")
    torch.save({'state': best['state'], 'ch': a.ch, 'n_class': N_CLASS, 'lobes': LOBES,
                'patch': PATCH, 'best_ep': best['ep'], 'val_patch_dice': best['dice'],
                'params': npar, 'seed': a.seed}, out)
    print(f"\n  最佳 ep{best['ep']}  val patch-Dice={best['dice']:.4f}")
    print(f"  权重 → results/seg3d_w{a.ch}.pt   总耗时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
