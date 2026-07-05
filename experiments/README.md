# 量化研究：重建 + AI 分割

本目录把主程序的两大 AI/算法能力从**功能实现**升级为**量化研究**，产出可复现、无患者数据的图表与指标：

- **研究一（重建）**：用 Shepp-Logan 标准体模测量 `recon.py` 重建质量随剂量/滤波器/算法的变化。
- **研究二（AI 分割）**：用带真值的公开 CT（TotalSegmentator-CT-Lite）实测 `organs.onnx` 的 Dice，并"测出"其 25 类标签映射。

> 被测对象都是产品代码本身——直接调用 GUI 所用的 `recon` 函数与 `ai_engine` 同款推理，无任何重写。

## 运行

```bash
python experiments/recon_study.py           # 实验 A + B（快，纯 FBP）
python experiments/recon_study.py c          # 实验 C（需构建系统矩阵，首次较慢，之后走磁盘缓存）
python experiments/recon_study.py a b c      # 全跑
```

产出写入 `experiments/results/`：每个实验一张 PNG 图 + 一份 CSV 原始数据。

> 复现依赖（App 之外）：`pip install -r experiments/requirements-experiments.txt`（matplotlib / nibabel / remotezip）。

## 方法

- **体模**：`skimage` Shepp-Logan，缩放至目标边长、归一化 `[0,1]`、施加与 `radon(circle=True)` 对齐的圆形掩码。掩码后的图即真值(GT)——正弦图只编码内切圆内信息，故仅在圆内比较。
- **正向模型**：`recon.compute_sinogram`（Radon 变换，`circle=True`）。
- **剂量代理**：投影角度数 `n_proj`（角度范围固定 180°）。投影越少 ≈ 剂量越低。
- **噪声模型（实验 C）**：Beer-Lambert + Poisson 光子统计。入射光子 `I0`，透射 `N ~ Poisson(I0·e^{-p})`，含噪投影 `p' = -ln(N/I0)`。`I0` 越小噪声越大。固定随机种子，可复现。
- **指标**：圆内 RMSE、NRMSE、SSIM（`data_range=1`）、PSNR。

## 发现

### A｜剂量–质量曲线（FBP Ram-Lak，256×256）
`exp_a_dose_quality.png`

RMSE 随投影数从 15 的 **0.222** 单调降到 360 的 **0.036**；SSIM 从 **0.35** 升到 **0.95**。
**约 180 投影后进入明显收益递减区**——继续加剂量对质量的边际提升很小。这为"够用即止"的剂量选择提供了量化依据。

### B｜滤波器最优选择随剂量翻转（256×256）
`exp_b_filters.png`

| n_proj | ramp(Ram-Lak) | shepp-logan | cosine | hamming | hann |
|---|---|---|---|---|---|
| 20（稀疏） | 0.176 | 0.167 | 0.154 | 0.145 | **0.143** |
| 180（稠密） | **0.037** | 0.040 | 0.048 | 0.054 | 0.055 |

**核心结论：不存在一个"最好"的滤波器，最优选择取决于剂量。**
低剂量/稀疏角度下，抑制高频噪声的切趾滤波器（hann/hamming）胜出；随剂量增加，锐利的 Ram-Lak 凭保真度反超。**交叉点在约 45–60 投影**。

### C｜光子噪声下解析 vs 迭代（64×64，I0=3×10⁴）
`exp_c_analytic_vs_iterative.png` · `exp_c_gallery.png`

| n_proj | FBP | DMR(最小二乘) | ART | SIRT |
|---|---|---|---|---|
| 30 | 0.099 | 0.151 | **0.069** | 0.083 |
| 60 | 0.089 | **0.611** | **0.054** | 0.076 |
| 90 | 0.087 | 0.090 | **0.050** | 0.075 |

1. **ART 全程最优**——非负约束 + 逐射线(Kaczmarz)更新起到隐式正则，噪声下最鲁棒。
2. **朴素最小二乘(DMR)在噪声下不稳定**，且在 60 投影处 RMSE 飙到 **0.611**：此时 64 探测器×60 投影 ≈ 3840 方程 ≈ 4096 未知数，系统接近方阵、**条件数最差**，噪声被最猛烈放大；30（欠定取最小范数解）与 90（超定有平均效应）反而更稳。这是病态逆问题的经典表现，非实现缺陷。
3. **SIRT 平稳保守**（~0.075），鲁棒性远优于 DMR，平滑代价是略逊 ART。

`exp_c_gallery.png` 的视觉对比与 RMSE 完全吻合：DMR 满屏椒盐噪声、FBP 有条纹伪影、ART 最干净、SIRT 最平滑。

## 重建研究一句话总结

在低剂量 CT 中，**"用什么算法/滤波器"的答案随剂量而变**：稀疏低剂量域应选带约束的迭代法(ART)或切趾滤波，充足剂量域解析法(FBP+Ram-Lak)即足够且更快。

---

# 研究二：AI 分割量化验证（`seg_validate.py`）

## 动机
`organs.onnx` 的 25 类 label→器官 映射长期只是**推断**（无官方 dataset.json）。本研究用带真值的公开 CT 客观测量分割质量，并**用混淆矩阵"测出"而非"猜出"映射**。

## 数据（不入库，需自备）
单例 **TotalSegmentator-CT-Lite**（CC-BY-4.0，1.5mm 各向同性，thorax-abdomen-pelvis 覆盖），用 HTTP Range 从 22GB 压缩包里**只抽一例**（约 42MB）：

```python
from remotezip import RemoteZip
base = "https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite/resolve/main"
open("s0029_img.nii.gz","wb").write(RemoteZip(base+"/Images.zip").read("Images/s0029.nii.gz"))
open("s0029_msk.nii.gz","wb").write(RemoteZip(base+"/Masks.zip").read("Masks/s0029.nii.gz"))
```

```bash
python experiments/seg_validate.py s0029_img.nii.gz s0029_msk.nii.gz
```

## 方法
影像规范到 RAS 再转成 GUI 的 (Z,H,W) 轴序 → `ai_engine` 同款预处理(clip[-1000,400] 归一化) + 沿 z 的 DZ=32 滑窗推理 → 与真值逐标签算 Dice/IoU，取每个输出标签重叠最大的真值器官（不预设映射为恒等，而是**测出**）。

## 发现
`exp` 产出：`seg_confusion.png`（混淆热图）· `seg_dice.csv` · `seg_mapping.md`

1. **映射被测出为完美恒等对角线**：our#k → TotalSegmentator 第 k 个器官，逐一命中。**由此确证模型 = TotalSegmentator v2 `class_map_part_organs`**（24 器官 + 背景，nnU-Net v2 导出），不再是"来源未知"。
2. **21 个在场器官平均 Dice ≈ 0.92**（肾/肺叶 0.97–0.99，甲状腺/胆囊等小器官 0.79–0.82），与 TotalSegmentator 官方公布水平一致——**同时验证了 GUI 推理管线正确**。
3. **纠正历史错标**：`5`=**肝**（旧推断误作"心脏"；模型无心脏/主动脉输出，二者在 TS 另一 part 编号 51/52，超出 0-24）；肺叶 `10,11`=**左**、`12,13,14`=**右**（旧标注左右判反系放射惯例镜像）。`models/organ_labels_candidate.json` 已据此改写为已确证映射。

| our# | 器官 | Dice | | our# | 器官 | Dice |
|---|---|---|---|---|---|---|
| 1 | 脾 | 0.97 | | 12 | 右肺上叶 | 0.97 |
| 2 | 右肾 | 0.99 | | 13 | 右肺中叶 | 0.96 |
| 3 | 左肾 | 0.98 | | 14 | 右肺下叶 | 0.99 |
| 5 | 肝 | 0.95 | | 16 | 气管 | 0.96 |
| 10 | 左肺上叶 | 0.99 | | 18 | 小肠 | 0.91 |
| 11 | 左肺下叶 | 0.99 | | 21 | 膀胱 | 0.87 |

## 分割研究一句话总结
不靠猜——**一例带真值公开 CT 就把模型身份、标签映射、管线正确性同时钉死**：organs.onnx 是 TotalSegmentator `class_map_part_organs`，平均 Dice≈0.92。
