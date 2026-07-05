# 医学影像工作站 Pro + 重建实验室

[![CI](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg)](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml)

基于 **PySide6 (Qt6)** 的桌面 CT 影像工作站，集成 **AI 多器官分割**、临床阅片工具与 **CT 断层重建教学实验室**。

> **定位声明**：本软件是**影像教学 / 科研工具**，**不是经认证的医疗器械，不得用于临床诊断**。AI 分割结果与器官定量为自动推断，仅供参考。

---

## Overview (English)

A desktop **CT imaging workstation** (PySide6/Qt6, ~4,700 lines of Python) that combines a clinical DICOM reader, a **from-scratch tomographic reconstruction laboratory**, and an **AI multi-organ segmentation** pipeline — built as a teaching/research tool (not a certified medical device).

Beyond the application, the repository contains **two reproducible quantitative studies** that turn the built-in algorithms into measured findings:

- **Low-dose reconstruction — dose–quality tradeoffs.** On the Shepp-Logan phantom, reconstruction error saturates beyond ≈180 views; the optimal FBP filter *inverts* with dose (smoothing filters win at sparse angles, the sharp Ram-Lak wins at dense angles); and under Poisson photon noise a constrained iterative solver (ART) is most robust while naive least-squares inversion becomes unstable near the square-system regime.
- **AI segmentation — provenance recovery & Dice validation.** By running the shipped ONNX model on one ground-truth-labelled public CT and computing a label-overlap confusion matrix, the undocumented model is identified as **TotalSegmentator v2 `class_map_part_organs`** (nnU-Net v2), with **mean Dice ≈ 0.92** over 21 organs — simultaneously validating the inference pipeline and correcting two label errors.

📄 **Read the [technical report](docs/technical_report.md)** (methods, figures, results) · 🧪 **Reproduce via [`experiments/`](experiments/README.md)**.

Highlights: mixin-decomposed architecture (6 cohesive modules), an 80-check offscreen-Qt regression suite, defensive DICOM handling, and reconstruction algorithms (Radon / FBP / DFR / ART / SIRT) implemented from first principles.

## 界面 · Screenshots

> 演示数据为公开的 **TotalSegmentator-CT-Lite**（CC-BY-4.0），**非患者数据、无 PHI**（患者栏已标注）。

**AI 多器官分割 · 轴位叠加**（右侧图例含各器官体积/HU 与免责声明）
![Axial view with AI multi-organ overlay](docs/img/gui_axial_segmentation.png)

**三平面 MPR · 十字线联动**（肺窗，5 个肺叶彩色叠加）
![Tri-planar MPR with linked cross-hairs](docs/img/gui_mpr_triplanar.png)

---

## 功能概览

### 临床阅片
- **DICOM 加载**：并行读盘，按解剖 Z 坐标排序；多序列目录自动取切片最多的序列
- **MPR 三平面**：横断 / 冠状 / 矢状面，十字线联动、切片/窗位同步
- **窗宽窗位**：三滑条 + 6 套临床预设（肺/纵隔/骨/血管/腹部/脑）+ 反色
- **测量工具**（左栏 9 个）：探针测 HU、卡尺测距、自由画笔、矩形/套索截取、3D 连通域追踪、**分割画笔/橡皮**、**椭圆 ROI 密度测量**
- **椭圆 ROI**：拖出椭圆读均值±SD / 最值 HU / 面积；可**拖动、缩放、删除**
- **标注**：切片专属 / 全局穿透，工程 JSON 存取
- **DICOM 四角叠加**：PACS 风格患者信息 / 窗位 / 切片，解剖方位字母 (A/P/R/L/S/I)
- **Cine 电影播放**：往返(bounce)连播、速度可调；**键盘翻片**（↑↓ / PgUp PgDn）
- **双序列随访对比**：加载既往序列并排显示，按 `ImagePositionPatient` **解剖配准**、切片/窗位联动

### AI 多器官分割
- 加载数据即**自动后台推理**（`models/organs.onnx`，25 类胸腹器官含肺叶）
- 结果彩色叠加 + **图例可点切换显隐** + **光标 HUD**（HU / 坐标 / 所在器官）
- **器官定量面板**：各器官体积(mL) / 平均 HU，导出 CSV
- **分割可编辑**：画笔补画（可选目标器官，计入其定量）/ 橡皮擦除，**Ctrl+Z 撤销**
- 无模型时自动降级为纯数学连通域算法

### 重建实验室（教学）
- 投影生成（Radon）：角度范围 60/120/180/360° + 采样密度 1×/2×/4×
- 解析重建：BP / FBP（5 种滤波器）/ DFR（傅里叶中心切片）
- 矩阵/迭代：DMR（最小二乘）/ ART / SIRT，误差图 + RMSE

### 合规
- **脱敏开关**：一键隐去屏幕与导出文件名中的患者身份（显示层）
- **AI 免责声明**：面板常驻 + 导出 CSV 内嵌

界面支持**中英双语**一键切换。

---

## 环境与运行

推荐使用专用 conda 环境 **`dicom_gui`**（Python 3.10）：

```bash
conda activate dicom_gui        # 或 conda env create -f environment.yml
python main.py                  # 空载启动
python main.py --data /path/to/dicom_dir   # 启动即加载指定 DICOM 目录（可选）
```

依赖：PySide6 · pydicom · numpy · scipy · scikit-image · onnxruntime（见 `requirements.txt` / `environment.yml`）。

启动默认**不加载任何数据**；用 `--data DIR` 指定 DICOM 目录，或运行后从界面「加载 DICOM 目录」选择。

---

## 目录结构

```
main.py            主窗口 MedicalViewer + 入口（加载 / 临床阅片渲染 / 窗位·工具·布局·AI 调度 / i18n / 键盘导航）
ui_builder.py      UiBuilderMixin —— 主窗口三栏布局与全部控件构建（左栏 / 视图栅格 / 右面板 / 两 Tab）
interaction.py     InteractionMixin —— Cine 电影播放 + MPR 联动/导航（十字线同步 / 换层换面 / HU 探针）
recon_lab.py       ReconLabMixin —— 重建实验室 UI 调度（投影 / BP / FBP / DFR / DMR / ART / SIRT）
compare_lab.py     CompareMixin —— 双序列随访对比（加载既往序列 / 解剖配准 / 联动）
annotation_lab.py  AnnotationMixin —— 标注/分割蒙版编辑/器官定量/工程持久化
ai_engine.py       AutoAIEngineThread —— 后台 AI 推理（滑窗 + 信号回调）
graphics_view.py   MedicalGraphicsView —— 影像交互视图 + ROIGraphicsItem
recon.py           纯计算重建算法（无 Qt 依赖）
constants.py       工具/平面常量 + 多器官调色板
style.qss          暗色主题
models/organs.onnx 分割模型（外部权重 organs.onnx.data 需单独放置，见下）
tests/test_gui.py  回归测试套件
experiments/       量化研究（重建剂量-质量权衡 + AI 分割 Dice 验证），产出图表/CSV，见其 README
```

---

## 模型说明（重要）

- `models/organs.onnx`（图，已入库）+ `models/organs.onnx.data`（**119MB 外部权重，未入库，需单独放置到 `models/` 下**）。
- **架构已确证：nnU-Net v2 `PlainConvUNet`（3D 全分辨率）**。由 ONNX 张量命名与结构逆向确认：`decoder.encoder.stages.*` + `decoder.seg_layers.*`（nnU-Net v2 深监督头命名）、6 级编码器通道 `[32,64,128,256,320,320]`（`max_features=320` 为 nnU-Net 默认）、5 级下采样（故输入 pad 到 2⁵=32 倍数）、InstanceNorm + LeakyReLU、25 类（24 器官 + 背景）、经 PyTorch 2.11 `torch.onnx` 导出。
- **出处与标签映射已确证（实测，非推断）**：模型 = **TotalSegmentator v2 `class_map_part_organs`**（24 器官 + 背景）。用一例带真值的公开 CT（TotalSegmentator-CT-Lite，1.5mm 各向同性）跑 GUI 同款推理并与真值逐标签算重叠，得到**完美恒等对角线**：our#k → 第 k 个器官，21 个在场器官**平均 Dice≈0.92**（肾/肺叶 0.97–0.99）。`models/organ_labels_candidate.json` 已改写为该确证映射。此举同时**验证了 GUI 推理管线正确**，并纠正旧推断的错标（`5`=肝非心脏；肺叶 `10,11`=左、`12,13,14`=右）。复现见 [`experiments/`](experiments/README.md) 的 `seg_validate.py`。
- ONNX 输入 `[1,1,D,H,W]`（每维 pad 到 32 倍数），输出 `[1,25,D,H,W]` logits，取 `argmax`。整卷推理约 **100 秒（CPU）**；如有 GPU 可为 `InferenceSession` 增加对应 ExecutionProvider 提速。

---

## 测试与质量

```bash
python tests/test_gui.py                     # 完整回归（80 项，需同目录 肺癌/ 真实数据）
SKIP_REAL_DATA=1 python tests/test_gui.py    # 仅数据无关子集（CI 用，无需真实数据/权重）
ruff check .                                 # 静态检查
coverage run tests/test_gui.py && coverage report   # 覆盖率
```

- 离屏 Qt 运行，覆盖 AI 引擎、历次修复、多器官分割/编辑、ROI 拖缩、双序列配准、Cine、合规等。退出码 0 = 全部通过。
- **CI**（`.github/workflows/ci.yml`）：每次 push/PR 在 Ubuntu 跑 `ruff` + 数据无关测试子集（离屏 Qt，无需真实数据或 119MB 权重）。
- **覆盖率**：完整套件 **≈66%**（`constants`/`ui_builder` 99–100%，`main`/`ai_engine` 81%；交互/图形/重建类含大量鼠标事件与算法路径，`recon.py` 的完整 FBP/DFR 由 [`experiments/`](experiments/README.md) 另行覆盖）。
- `recon.py`/`ai_engine.py` 为纯计算/无 Qt 模块，已加完整类型注解。

---

## 已知限制

- **非临床器械**：无监管认证、无算法验证文档、无审计/访问控制。
- **脱敏为显示层**：隐去屏幕与导出文件名中的 PHI，但**不清洗底层 DICOM 标签与烧录文字**——非完整去标识。
- **AI 标签已确证**：映射经 TotalSegmentator 真值实测（平均 Dice≈0.92），器官归属可信；但定量数值仍受单序列分辨率与分割边界误差影响。
- 矩阵重建（DMR/ART）受 `lstsq` 限制，实用上限 64×64（教学用途）。
- 单体数据模型：AI/重建/标注默认作用于当前主序列。
- **AI 蒙版叠加仅横断面**：冠状/矢状面不显示器官分割叠加。
- MPR 冠/矢状面**不保留** Ctrl+滚轮缩放（切片刷新会按解剖比例重适配；横断面正常保留）。

---

## 技术报告与量化研究

- **技术报告**（英文）：[docs/technical_report.md](docs/technical_report.md) —— 两组量化研究（低剂量重建剂量-质量权衡 / AI 分割 Dice 验证与出处确证）的方法、图表、结论。
- **预印本稿**（英文，研究一）：[docs/preprint_recon.md](docs/preprint_recon.md) —— 稀疏视角/低剂量 CT 重建权衡的可复现体模研究，学术格式（摘要/相关工作/方法/实验/讨论/复现/参考文献）。
- **可复现实验**：[experiments/](experiments/README.md) —— 脚本 + 图表 + CSV。

## 变更记录

历次缺陷排查与修复的审查小结见 [CHANGELOG.md](CHANGELOG.md)。

---

## 第三方组件 · Acknowledgements

本项目在自研代码之外集成了以下第三方成果，其著作权/许可归各自作者所有：

- **AI 分割模型**：`models/organs.onnx` 为 **TotalSegmentator v2**（Wasserthal et al., *Radiology: AI*, 2023；基于 **nnU-Net v2**，Isensee et al., *Nature Methods*, 2021）的 `class_map_part_organs` 导出图。**模型权重（`organs.onnx.data`）未随本仓库分发**，其许可以 TotalSegmentator 官方为准。
- **验证数据**：分割验证使用公开的 **TotalSegmentator-CT-Lite**（CC-BY-4.0）单例，**未随本仓库分发**。
- **框架/库**：PySide6 (Qt for Python, LGPL)、pydicom、NumPy、SciPy、scikit-image、ONNX Runtime、nibabel。

自研代码（`main.py`、`recon.py`、`graphics_view.py`、`ai_engine.py`、各 `*_lab.py`、`ui_builder.py`、`interaction.py`、`experiments/` 等）为本人独立编写。

## 版权 · Copyright

© 2026 盛超 (Sheng Chao)。**保留所有权利 / All rights reserved.**

本仓库为个人教学/科研与作品集用途，**未授予**任何开源复制、修改或再分发许可；如需使用请联系作者。第三方组件依其各自许可。
