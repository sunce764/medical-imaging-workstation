# =============================================================================
# 研究三：学习式稀疏角重建 —— 它到底恢复了什么，又编造了什么
# ---------------------------------------------------------------------------
# 目的：研究一量化出传统方法的天花板（误差随视角饱和、最优滤波器随剂量翻转）。
#       本研究接着问：用一个自实现的小网络做 FBP 后处理，能否突破线性滤波
#       「保细节 vs 压伪影」的固有权衡？代价是什么？
#
# 为什么不只报 RMSE：稀疏角重建里，把图抹平就能降 RMSE，而抹掉的恰恰是最该
#       看清的小病灶。故本研究用三层指标，缺一层就会自欺：
#         全局  RMSE / SSIM
#         细节  病灶对比度保留率、条形栅格的调制度传递（CTF）
#         安全  背景条纹强度、**虚假结构检出率（幻觉）**
#
# 为什么幻觉是主结论之一：传统方法（FBP/ART）在稀疏角下产生的是**伪影**——难看，
#       但医生看得出不可信；学习式重建产生的是**幻觉**——图很干净、像真的解剖，
#       而那里本来什么都没有。临床上这是两种性质完全不同的失效。
#
# 被测代码：正演与解析重建全部直接调用 recon.py 中 GUI 所用的同一批函数。
# 数据：随机模体家族，由代码生成、种子固定，**不使用任何患者数据**，别人跑得出同样结果。
#
# 用法：
#   python experiments/recon_dl.py            # 全跑（矩阵 + 幻觉 + 分布外 + 分辨率）
#   python experiments/recon_dl.py matrix      # 只跑视角矩阵
#   python experiments/recon_dl.py halluc ood res   # 按需单跑（复用已存权重）
#   python experiments/recon_dl.py export           # 把 20 视角模型导出为 ONNX 供 GUI 使用
#
# 产出：experiments/results/recon_dl_*.{png,csv}（只新增，不覆盖已有产物）
# 依赖：torch（见 requirements-experiments.txt）。App 运行不需要它——
#       模型若要进 GUI，走 ONNX 导出，由已有的 onnxruntime 推理。
# =============================================================================

import csv
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon  # noqa: E402  被测的产品代码

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SIZE = 128
VIEWS = [15, 20, 30, 45, 60]
# 三个种子段互不重叠——模体实例级隔离，杜绝同一实例跨集出现
SEED_TRAIN, SEED_VAL, SEED_TEST, SEED_PAIR = (0, 100_000), (100_000, 110_000), \
                                             (110_000, 120_000), (200_000, 210_000)

if HAS_TORCH:
    torch.set_num_threads(1)
    DEV = torch.device('mps' if torch.backends.mps.is_available()
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))


# ---------------------------------------------------------------------------
# 模体家族：随机椭圆 + 小高对比结构（"病灶"）
# ---------------------------------------------------------------------------
def _ellipse(n, cy, cx, ay, ax, ang, val):
    yy, xx = np.mgrid[0:n, 0:n]
    yy = (yy - n / 2) / (n / 2); xx = (xx - n / 2) / (n / 2)
    c, s = np.cos(ang), np.sin(ang)
    yr = (yy - cy) * c + (xx - cx) * s
    xr = -(yy - cy) * s + (xx - cx) * c
    return ((yr / ay) ** 2 + (xr / ax) ** 2 <= 1.0) * val


def _fov(img, n):
    """圆形视野外置零，与 radon(circle=True) 的处理域一致。"""
    yy, xx = np.mgrid[0:n, 0:n]
    img = img.copy()
    img[(((yy - n / 2) / (n / 2)) ** 2 + ((xx - n / 2) / (n / 2)) ** 2) > 1.0] = 0.0
    return np.clip(img, 0, 1).astype(np.float32)


def make_phantom(seed, n=SIZE, n_lesion=None):
    """随机模体。固定 Shepp-Logan 会被网络直接背下来，测出的就不是重建能力。"""
    rng = np.random.RandomState(seed)
    img = _ellipse(n, 0, 0, rng.uniform(0.80, 0.92), rng.uniform(0.68, 0.86),
                   rng.uniform(-0.2, 0.2), 0.35)
    for _ in range(rng.randint(3, 8)):
        img += _ellipse(n, rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45),
                        rng.uniform(0.12, 0.38), rng.uniform(0.12, 0.38),
                        rng.uniform(0, np.pi), rng.uniform(-0.22, 0.28))
    lesions = []
    for _ in range(rng.randint(0, 4) if n_lesion is None else n_lesion):
        cy, cx = rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40)
        r, val = rng.uniform(0.030, 0.065), rng.uniform(0.25, 0.45)
        img += _ellipse(n, cy, cx, r, r, 0.0, val)
        lesions.append((cy, cx, r, val))
    return _fov(img, n), lesions


def forward_fbp(img, n_views, filt='ramp'):
    """正演到稀疏角弦图再 FBP 回来，全程调产品代码。

    注意 compute_fbp 返回 (未滤波BP, 滤波后FBP)——取错第一个会拿到 BP，
    尺度大两个数量级且本来就糊，视角数的影响会被完全淹没（开发时踩过）。
    """
    theta = recon.make_theta(180.0, n_views)
    sino = recon.compute_sinogram(img, theta)
    _, fbp = recon.compute_fbp(sino, theta, filt)
    return fbp.astype(np.float32)


# ---------------------------------------------------------------------------
# 网络：残差 U-Net（预测伪影再相减）
# ---------------------------------------------------------------------------
if HAS_TORCH:
    def _blk(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True),
            nn.LeakyReLU(0.01, True))

    class ResUNet(nn.Module):
        """自实现，不用 MONAI/nnU-Net——本研究的目的之一就是自己搭网络。

        残差而非直接预测干净图：稀疏角伪影是稀疏的高频条纹，比整幅解剖好学；
        直接预测等于让网络重新合成解剖，幻觉风险更高。
        输出层零初始化 → 训练起点恒等于「原样输出 FBP」，「网络改进了多少」有干净的零基准。
        """

        def __init__(self, c=32):
            super().__init__()
            self.e1, self.e2, self.e3 = _blk(1, c), _blk(c, c * 2), _blk(c * 2, c * 4)
            self.bott = _blk(c * 4, c * 8)
            self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2); self.d3 = _blk(c * 8, c * 4)
            self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2); self.d2 = _blk(c * 4, c * 2)
            self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2);     self.d1 = _blk(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1); self.pool = nn.MaxPool2d(2)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            b = self.bott(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(b), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return x - self.out(d1)


# ---------------------------------------------------------------------------
# 三层指标
# ---------------------------------------------------------------------------
def lesion_contrast(recon_img, truth, lesions, n=SIZE):
    """病灶对比度保留率。用「核心均值 − 紧邻环带均值」，避免整体亮度漂移被误算成对比度变化。

    上限是 1.0（与真值一致），不是 FBP 实测的 0.95——0.95 本身已被伪影压低，
    网络超过它属于恢复，不是「增强」（开发时曾据此误判为幻觉）。
    """
    yy, xx = np.mgrid[0:n, 0:n]
    out = []
    for k, les in enumerate(lesions):
        for cy, cx, r, _ in les:
            d2 = ((yy - n / 2) / (n / 2) - cy) ** 2 + ((xx - n / 2) / (n / 2) - cx) ** 2
            core, ring = d2 <= r * r, (d2 > (1.6 * r) ** 2) & (d2 <= (2.4 * r) ** 2)
            if core.sum() < 3 or ring.sum() < 3:
                continue
            ct = truth[k, 0][core].mean() - truth[k, 0][ring].mean()
            if abs(ct) > 1e-6:
                out.append((recon_img[k, 0][core].mean() - recon_img[k, 0][ring].mean()) / ct)
    return float(np.mean(out)) if out else float('nan')


def evaluate(pred, truth, lesions):
    from skimage.metrics import structural_similarity as ssim
    bg = truth < 0.01
    return dict(
        rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
        ssim=float(np.mean([ssim(truth[i, 0], pred[i, 0],
                                 data_range=float(truth[i, 0].max() - truth[i, 0].min()) or 1.0)
                            for i in range(len(truth))])),
        lesion=lesion_contrast(pred, truth, lesions),
        streak=float(pred[bg].std()) if bg.any() else float('nan'))


def build_set(seed_range, n, n_views, filt):
    xs, ys, les = [], [], []
    for i in range(n):
        img, lz = make_phantom(seed_range[0] + i)
        xs.append(forward_fbp(img, n_views, filt)); ys.append(img); les.append(lz)
    return np.stack(xs)[:, None], np.stack(ys)[:, None], les


def _predict(net, x, bs=16):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            t = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).float().to(DEV)
            outs.append(net(t).cpu().numpy())
    return np.concatenate(outs)


def train_one(n_views, ntrain=600, nval=80, epochs=40, bs=8, ch=32, lr=1e-3, seed=0):
    """训练一个后处理网络。

    输入固定用 ramp-FBP：hann 已在滤波阶段把高频连同细节一并滤掉，丢失的信息
    网络无从恢复；ramp 保留信息但留下条纹，那才是可学的部分。
    """
    # 【torch 侧的种子也必须固定】此前只有 build_set 里的 RandomState 固定了模体
    # 数据，权重初始化与下面的 torch.randperm 打乱都没固定——同一条命令重跑会得到
    # 另一份权重。seg3d_train.py:223 一直是这么做的，此处是遗漏。
    # 【范围】本行只约束今后的运行。已提交的 recon_dl_* 产物跑在它加入之前，且
    # 【尚未】用固定种子重跑并与那批产物比对过——因此既不声称二者一致，也不对
    # 二者有多接近作任何断言，那需要一次实际的重跑作证据。README 与本目录
    # README 均按此口径写明。
    torch.manual_seed(seed)
    xtr, ytr, _ = build_set(SEED_TRAIN, ntrain, n_views, 'ramp')
    xva, yva, lva = build_set(SEED_VAL, nval, n_views, 'ramp')
    net = ResUNet(ch).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr = torch.from_numpy(xtr).to(DEV); Ytr = torch.from_numpy(ytr).to(DEV)
    best = {'rmse': float('inf')}
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); nn.functional.mse_loss(net(Xtr[idx]), Ytr[idx]).backward(); opt.step()
        sch.step()
        if ep % 5 == 0:
            net.eval()
            mv = evaluate(_predict(net, xva), yva, lva)
            # checkpoint 只按【验证集】选，测试集全程不参与选择——避免选择偏差
            if mv['rmse'] < best['rmse']:
                best = dict(mv); best['ep'] = ep
                best['state'] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best['state']); net.eval()
    return net, best['ep']


# ---------------------------------------------------------------------------
# 实验 A：视角矩阵 —— 5 档视角 × {FBP-hann, FBP-ramp, +CNN}
# ---------------------------------------------------------------------------
def exp_matrix(nval=80):
    """ART/SIRT 不进本表：它们需显式系统矩阵，lstsq 成本把可用尺寸限制在 ~64²，
    与本实验的 128² 不可直接比。混进同一张表会制造「可比」的假象。"""
    rows, nets = [], {}
    for nv in VIEWS:
        t = time.perf_counter()
        net, ep = train_one(nv, nval=nval)
        nets[nv] = net
        # 存权重：专项实验（幻觉/分布外/分辨率）可独立复跑，不必每次重训 10 分钟
        torch.save({'state': net.state_dict(), 'ch': 32, 'views': nv, 'best_ep': ep},
                   os.path.join(RESULTS, f"recon_dl_w{nv}.pt"))
        xr, yt, lt = build_set(SEED_TEST, nval, nv, 'ramp')
        xh, _, _ = build_set(SEED_TEST, nval, nv, 'hann')   # 同一批模体，只换滤波器
        res = {'FBP-hann': evaluate(xh, yt, lt), 'FBP-ramp': evaluate(xr, yt, lt),
               '+CNN': evaluate(_predict(net, xr), yt, lt)}
        for meth, m in res.items():
            rows.append(dict(views=nv, method=meth, **{k: round(float(v), 5) for k, v in m.items()}))
        print(f"  [{nv:>2} 视角] ep{ep} {time.perf_counter()-t:.0f}s  " + "  ".join(
            f"{k}:RMSE={v['rmse']:.4f}/病灶={v['lesion']:.3f}" for k, v in res.items()))
        sys.stdout.flush()
    _write_csv("recon_dl_matrix.csv", rows)
    _plot_matrix(rows)
    return rows, nets


# ---------------------------------------------------------------------------
# 实验 B：幻觉 —— 本来没有结构的地方，网络会不会造一个出来
# ---------------------------------------------------------------------------
def exp_hallucination(net, n_views, npair=60):
    """配对模体：同一背景，唯一差别是某处有没有病灶。拿【无病灶版】喂网络，
    看那个位置有没有长出东西。背景完全相同，故任何差异只能来自网络。

    必须单独做：矩阵里的「背景条纹 std」是全背景标准差，一个孤立假病灶会被
    上万个背景像素稀释，根本抓不到。
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    def signal(img, cy, cx, r):
        d2 = ((yy - SIZE / 2) / (SIZE / 2) - cy) ** 2 + ((xx - SIZE / 2) / (SIZE / 2) - cx) ** 2
        core, ring = d2 <= r * r, (d2 > (1.6 * r) ** 2) & (d2 <= (2.4 * r) ** 2)
        if core.sum() < 3 or ring.sum() < 3:
            return float('nan')
        return float(img[core].mean() - img[ring].mean())

    xs_wo, xs_wi, meta = [], [], []
    for s in range(SEED_PAIR[0], SEED_PAIR[0] + npair * 3):
        if len(meta) >= npair:
            break
        base, _ = make_phantom(s, n_lesion=0)
        rng = np.random.RandomState(s + 777)
        for _ in range(40):
            cy, cx, r = rng.uniform(-.35, .35), rng.uniform(-.35, .35), 0.045
            d2 = ((yy - SIZE / 2) / (SIZE / 2) - cy) ** 2 + ((xx - SIZE / 2) / (SIZE / 2) - cx) ** 2
            core = d2 <= r * r
            # 要求落在体内且局部均匀：否则测到的「幻觉」可能只是网络在恢复邻近的真实结构
            if core.sum() >= 5 and base[core].mean() > 0.15 and base[core].std() < 0.03:
                wi = _fov(base + _ellipse(SIZE, cy, cx, r, r, 0.0, 0.35), SIZE)
                xs_wi.append(forward_fbp(wi, n_views)); xs_wo.append(forward_fbp(base, n_views))
                meta.append((cy, cx, r, signal(wi, cy, cx, r), signal(xs_wo[-1], cy, cx, r)))
                break
    pw = _predict(net, np.stack(xs_wo)[:, None])
    pi = _predict(net, np.stack(xs_wi)[:, None])
    tw = np.array([m[3] for m in meta])
    fw = np.array([m[4] for m in meta])
    nw = np.array([signal(pw[k, 0], *meta[k][:3]) for k in range(len(meta))])
    ni = np.array([signal(pi[k, 0], *meta[k][:3]) for k in range(len(meta))])
    out = dict(n_pair=len(meta), truth_with=float(tw.mean()), fbp_without=float(fw.mean()),
               net_without=float(nw.mean()), net_with=float(ni.mean()),
               recovery_pct=float(ni.mean() / tw.mean() * 100),
               **{f"halluc_rate_{int(t*100)}pct": float((nw > t * tw).mean() * 100)
                  for t in (0.2, 0.3, 0.5)})
    print(f"  配对 {out['n_pair']}  真病灶信号 {out['truth_with']:+.4f} | "
          f"FBP无病灶处 {out['fbp_without']:+.4f} | 网络无病灶处 {out['net_without']:+.4f} | "
          f"真病灶恢复 {out['recovery_pct']:.0f}%")
    print("  幻觉率  " + "  ".join(f">{k.split('_')[-1]}: {v:.1f}%"
                                   for k, v in out.items() if k.startswith('halluc')))
    _write_csv("recon_dl_hallucination.csv", [out])
    return out


# ---------------------------------------------------------------------------
# 实验 C：分布外 —— 学到的是通用去伪影，还是记住了训练分布的形状
# ---------------------------------------------------------------------------
def _ood_sets(seed, n=SIZE):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    ny, nx = (yy - n / 2) / (n / 2), (xx - n / 2) / (n / 2)
    body = _ellipse(n, 0, 0, 0.86, 0.78, 0, 0.35)
    sq = body.copy()
    for _ in range(rng.randint(3, 6)):                      # 尖角直边：椭圆先验最易露馅处
        cy, cx = rng.uniform(-.4, .4), rng.uniform(-.4, .4)
        sq += ((np.abs(ny - cy) <= rng.uniform(.10, .28))
               & (np.abs(nx - cx) <= rng.uniform(.10, .28))) * rng.uniform(-.2, .3)
    pg = body.copy()                                        # 非凸多边形：边数与凹凸都不在训练分布里
    from matplotlib.path import Path as _MplPath
    pts = np.stack([ny.ravel(), nx.ravel()], 1)
    for _ in range(rng.randint(2, 5)):
        k = rng.randint(3, 7)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k)); rad = rng.uniform(.08, .26, k)
        cy, cx = rng.uniform(-.35, .35), rng.uniform(-.35, .35)
        poly = np.stack([cy + rad * np.sin(ang), cx + rad * np.cos(ang)], 1)
        pg += _MplPath(poly).contains_points(pts).reshape(n, n) * rng.uniform(-.2, .3)
    ln = body.copy()                                        # 高频细线：分辨率极限探针
    for _ in range(rng.randint(2, 4)):
        y0, gap = rng.randint(int(n * .25), int(n * .75)), int(rng.choice([3, 4, 6]))
        for k in range(6):
            r0 = y0 + k * gap
            if r0 + (1 if gap <= 4 else 2) < n:
                ln[r0:r0 + (1 if gap <= 4 else 2), int(n * .3):int(n * .7)] += 0.35
    return {'方块(尖角直边)': _fov(sq, n), '多边形(非凸)': _fov(pg, n),
            '细线栅格(高频)': _fov(ln, n)}


def exp_ood(net, n_views, in_gain, n_each=24):
    from skimage.metrics import structural_similarity as ssim
    names = list(_ood_sets(0).keys())
    rows = []
    for nm in names:
        T = np.stack([_ood_sets(s)[nm] for s in range(n_each)])[:, None]
        F = np.stack([forward_fbp(t[0], n_views) for t in T])[:, None]
        P = _predict(net, F)
        rf = float(np.sqrt(np.mean((F - T) ** 2))); rp = float(np.sqrt(np.mean((P - T) ** 2)))
        rows.append(dict(set=nm, rmse_fbp=round(rf, 5), rmse_cnn=round(rp, 5),
                         gain_pct=round((rf - rp) / rf * 100, 2),
                         ssim_fbp=round(float(np.mean([ssim(T[i, 0], F[i, 0], data_range=1.0)
                                                       for i in range(len(T))])), 4),
                         ssim_cnn=round(float(np.mean([ssim(T[i, 0], P[i, 0], data_range=1.0)
                                                       for i in range(len(T))])), 4)))
        print(f"  {nm:<16} RMSE {rf:.4f}→{rp:.4f}  降幅 {rows[-1]['gain_pct']:.1f}%")
    ratio = float(np.mean([r['gain_pct'] for r in rows]) / in_gain) if in_gain else float('nan')
    print(f"  分布内降幅 {in_gain:.1f}% → 分布外均值 {np.mean([r['gain_pct'] for r in rows]):.1f}%"
          f"  增益比 {ratio:.2f}（接近 1 = 通用去伪影；远小于 1 = 先验记忆）")
    _write_csv("recon_dl_ood.csv", rows + [dict(set='__ratio__', gain_pct=round(ratio, 3))])
    return rows, ratio


# ---------------------------------------------------------------------------
# 实验 D：分辨率极限 —— 条形栅格的调制度传递
# ---------------------------------------------------------------------------
def _peak_valley(a, bar_rows, gap_rows, x0, x1):
    """栅格的峰谷差。写成模块级函数而非循环内的 lambda——后者会捕获循环变量，
    是延迟绑定的经典陷阱（ruff B023 已在此拦下一次）。"""
    return float(a[bar_rows, x0:x1].mean() - a[gap_rows, x0:x1].mean())


def exp_resolution(net, n_views, periods=(4, 6, 8, 10, 12, 16, 20, 28), n_rep=8):
    """判据是 |CTF − 1| 而非 CTF 越大越好。

    ramp 滤波在锐边有过冲（Gibbs 振铃），CTF 可以 **大于 1**——那是失真不是优势。
    开发时初版按「越大越好」判读，把「网络校正了过冲」误报成「网络在抹平」，
    结论与事实完全相反。
    """
    rows = []
    for p in periods:
        half = max(1, p // 2)
        cf, cn, rf, rn = [], [], [], []
        for s in range(n_rep):
            rng = np.random.RandomState(s)
            img = _ellipse(SIZE, 0, 0, 0.86, 0.78, 0, 0.35).astype(np.float32)
            y0 = SIZE // 2 - (7 * p) // 2 + rng.randint(-4, 5)
            x0, x1 = int(SIZE * .30), int(SIZE * .70)
            bars = []
            for k in range(7):
                r0 = y0 + k * p
                if 0 <= r0 and r0 + half <= SIZE:
                    img[r0:r0 + half, x0:x1] += 0.35
                    # 记录条纹占据的【全部行】，不是起始行——起始行恰是边缘，而 ramp
                    # 过冲在边缘最强，只采边缘会把 CTF 系统性拉高（搬运时曾错成起始行，
                    # 周期16 的 CTF 从 1.19 虚高到 2.27）。
                    bars.extend(range(r0, r0 + half))
            img = _fov(img, SIZE)
            f = forward_fbp(img, n_views)
            o = _predict(net, f[None, None])[0, 0]
            bset = set(bars)
            gaps = [r + half for r in bars if r + half < SIZE and (r + half) not in bset]
            if not bars or not gaps:
                continue
            t = _peak_valley(img, bars, gaps, x0, x1)
            if abs(t) > 1e-6:
                cf.append(_peak_valley(f, bars, gaps, x0, x1) / t)
                cn.append(_peak_valley(o, bars, gaps, x0, x1) / t)
            rf.append(np.sqrt(np.mean((f - img) ** 2))); rn.append(np.sqrt(np.mean((o - img) ** 2)))
        if not cf:
            continue
        cf, cn, rf, rn = np.mean(cf), np.mean(cn), np.mean(rf), np.mean(rn)
        rows.append(dict(period_px=p, linewidth_px=half, lp_per_px=round(1 / p, 4),
                         ctf_fbp=round(float(cf), 4), ctf_cnn=round(float(cn), 4),
                         dev_fbp=round(abs(float(cf) - 1), 4), dev_cnn=round(abs(float(cn) - 1), 4),
                         rmse_fbp=round(float(rf), 5), rmse_cnn=round(float(rn), 5),
                         gain_pct=round((rf - rn) / rf * 100, 2)))
        print(f"  周期{p:>3}px 线宽{half:>2}px  CTF {cf:.3f}→{cn:.3f}  "
              f"|CTF-1| {abs(cf-1):.3f}→{abs(cn-1):.3f}  RMSE降幅 {rows[-1]['gain_pct']:.1f}%")
    better = sum(1 for r in rows if r['dev_cnn'] < r['dev_fbp'] - 0.01)
    print(f"  {better}/{len(rows)} 个频率上 CNN 的调制度更接近真值")
    _write_csv("recon_dl_resolution.csv", rows)
    _plot_resolution(rows)
    return rows


# ---------------------------------------------------------------------------
# 产出
# ---------------------------------------------------------------------------
def _write_csv(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RESULTS, name), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"    → results/{name}")


def _plot_matrix(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title, ylab) in zip(axes, [  # noqa: B905 — 两者等长，由下方列表保证
            ('rmse', 'RMSE (lower is better)', 'RMSE'),
            ('lesion', 'Lesion contrast retention (1.0 = ground truth)', 'retention'),
            ('streak', 'Background streak level (lower is better)', 'std')]):
        for meth, mk in (('FBP-hann', 'o--'), ('FBP-ramp', 's--'), ('+CNN', '^-')):
            ys = [next(r[key] for r in rows if r['views'] == v and r['method'] == meth)
                  for v in VIEWS]
            ax.plot(VIEWS, ys, mk, label=meth, lw=2)
        if key == 'lesion':
            ax.axhline(1.0, color='k', ls=':', lw=1, label='ground truth (ceiling)')
        ax.set_xlabel('number of projection views'); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle('Learned post-processing vs linear filtering: three-tier metrics for sparse-view CT', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "recon_dl_matrix.png"), dpi=140)
    plt.close(fig)
    print("    → results/recon_dl_matrix.png")


def _plot_resolution(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    p = [r['period_px'] for r in rows]
    axes[0].plot(p, [r['ctf_fbp'] for r in rows], 's--', label='FBP-ramp', lw=2)
    axes[0].plot(p, [r['ctf_cnn'] for r in rows], '^-', label='+CNN', lw=2)
    axes[0].axhline(1.0, color='k', ls=':', lw=1.2, label='ground truth (CTF = 1)')
    axes[0].set_xlabel('bar-pattern period (px)'); axes[0].set_ylabel('CTF')
    axes[0].set_title('Modulation transfer: CTF > 1 is ramp overshoot, not an advantage')
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
    axes[1].plot(p, [r['gain_pct'] for r in rows], 'o-', color='tab:green', lw=2)
    axes[1].set_xlabel('bar-pattern period (px)'); axes[1].set_ylabel('RMSE reduction (%)')
    axes[1].set_title('Gain decays toward the resolution limit')
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "recon_dl_resolution.png"), dpi=140)
    plt.close(fig)
    print("    → results/recon_dl_resolution.png")


def export_onnx(n_views=20, out=None):
    """把训练好的网络导出为 ONNX，供 GUI 用已有的 onnxruntime 推理。

    这样 App 不需要 torch——与 organs.onnx 同一套技术栈。
    动态 batch 与空间维：GUI 的重建实验室图像尺寸可变（16/32/64/128…），
    固定尺寸会让模型在非 128² 时直接拒绝加载。
    """
    wp = os.path.join(RESULTS, f"recon_dl_w{n_views}.pt")
    if not os.path.exists(wp):
        print(f"  缺少权重 {wp}，请先跑 `python experiments/recon_dl.py matrix`")
        return None
    ck = torch.load(wp, map_location='cpu', weights_only=False)
    net = ResUNet(ck['ch']).cpu().eval()
    net.load_state_dict({k: v.cpu() for k, v in ck['state'].items()})
    out = out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", f"recon_dl_v{n_views}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(
        net, torch.randn(1, 1, SIZE, SIZE), out,
        input_names=['fbp'], output_names=['recon'], opset_version=17,
        dynamic_axes={'fbp': {0: 'n', 2: 'h', 3: 'w'}, 'recon': {0: 'n', 2: 'h', 3: 'w'}})
    mb = os.path.getsize(out) / 1e6
    # 导出后必须核对数值：ONNX 与 torch 不一致会静默给出不同的重建图
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
        x = np.random.RandomState(0).rand(1, 1, SIZE, SIZE).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        got = sess.run(None, {'fbp': x})[0]
        d = float(np.abs(ref - got).max())
        print(f"  导出 {os.path.relpath(out)}  {mb:.1f}MB  "
              f"ONNX vs torch 最大偏差 {d:.2e} {'✓' if d < 1e-4 else '✗ 偏差过大'}")
        # 非 128² 也要能跑，否则 GUI 换图像尺寸就崩
        x64 = np.random.RandomState(1).rand(1, 1, 64, 64).astype(np.float32)
        sess.run(None, {'fbp': x64})
        print("  动态尺寸自检：64² 输入可推理 ✓")
    except ImportError:
        print(f"  导出 {os.path.relpath(out)}  {mb:.1f}MB（未装 onnxruntime，跳过数值核对）")
    return out


def _read_csv(name):
    with open(os.path.join(RESULTS, name), encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ('method', 'set'):
                try:
                    r[k] = float(v) if '.' in v else int(v)
                except (ValueError, TypeError):
                    pass
    return rows


def replot():
    """从已提交的 CSV 重画图。改图样式/语言不必重跑数小时的实验，
    也保证图与 CSV 恒为同一份数据。"""
    _plot_matrix(_read_csv("recon_dl_matrix.csv"))
    _plot_resolution(_read_csv("recon_dl_resolution.csv"))


def main():
    args_l = [a.lower() for a in sys.argv[1:]]
    if 'plot' in args_l:
        replot(); return 0
    if 'export' in args_l:
        if not HAS_TORCH:
            print("导出需要 torch"); return 1
        export_onnx(20); return 0
    if not HAS_TORCH:
        print("需要 torch，见 experiments/requirements-experiments.txt"); return 1
    args = [a.lower() for a in sys.argv[1:]] or ['matrix', 'halluc', 'ood', 'res']
    print(f"设备={DEV}  尺寸={SIZE}²  视角档={VIEWS}")
    nets, in_gain = {}, None
    if 'matrix' in args:
        print("\n=== 实验 A：视角矩阵 ===")
        rows, nets = exp_matrix()
        r20 = {m: next(r for r in rows if r['views'] == 20 and r['method'] == m)
               for m in ('FBP-ramp', '+CNN')}
        in_gain = (r20['FBP-ramp']['rmse'] - r20['+CNN']['rmse']) / r20['FBP-ramp']['rmse'] * 100
    if any(k in args for k in ('halluc', 'ood', 'res')):
        net = nets.get(20)
        if net is None:
            wp = os.path.join(RESULTS, "recon_dl_w20.pt")
            if os.path.exists(wp):
                ck = torch.load(wp, map_location='cpu', weights_only=False)
                net = ResUNet(ck['ch']).to(DEV); net.load_state_dict(ck['state']); net.eval()
                print(f"  复用已存权重 recon_dl_w20.pt（ep{ck['best_ep']}）")
            else:
                net = train_one(20)[0]
        if in_gain is None:
            xr, yt, lt = build_set(SEED_TEST, 80, 20, 'ramp')
            b, c = evaluate(xr, yt, lt), evaluate(_predict(net, xr), yt, lt)
            in_gain = (b['rmse'] - c['rmse']) / b['rmse'] * 100
        if 'halluc' in args:
            print("\n=== 实验 B：幻觉 ===");  exp_hallucination(net, 20)
        if 'ood' in args:
            print("\n=== 实验 C：分布外 ===");  exp_ood(net, 20, in_gain)
        if 'res' in args:
            print("\n=== 实验 D：分辨率极限 ===");  exp_resolution(net, 20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
