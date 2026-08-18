<h1 align="center">医学影像工作站 + 重建实验室</h1>

<p align="center"><strong>CT 阅片 · AI 分割 · 可复现重建研究</strong></p>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

<p align="center">
  <a href="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white">
  <img alt="PySide6 / Qt6" src="https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&amp;logoColor=white">
  <a href="LICENSE"><img alt="专有许可" src="https://img.shields.io/badge/License-Proprietary-lightgrey"></a>
  <img alt="仅供教学科研" src="https://img.shields.io/badge/⚠️-教学%2F科研·非医疗器械-critical">
</p>

基于 **PySide6（Qt6）** 的桌面 **CT 影像工作站**，在一个应用内整合临床式 DICOM 阅片、AI 多器官分割与断层重建教学实验室。仓库另附**三项正式记录的量化研究、两项多例验证，以及一次促成产品改动的消融实验**，把产品代码转化为可测、可复现的证据。

> [!WARNING]
> **仅供教学与科研。** 本软件不是经认证的医疗器械，不得用于临床诊断。AI 分割和器官定量均为自动估计，不构成临床结论。

![AI 分割与逐器官置信度](docs/img/gui_confidence.png)

## 一览

| 产品 | 运行环境 | 证据 | 数据边界 |
|---|---|---|---|
| DICOM 阅片 + AI 分割 + 重建实验室 | Python 3.10 · PySide6/Qt6 · CPU-only | 研究 I–III · 57 例肺叶与 20 例多器官验证 · 一次修好真实缺陷的 spacing 消融 | 合成模体与公开去标识研究 CT；仓库不提交 PHI |

## 界面

| AI 多器官分割 | 三平面 MPR + 十字线联动 |
|:---:|:---:|
| ![轴位分割](docs/img/gui_axial_segmentation.png) | ![三平面 MPR](docs/img/gui_mpr_triplanar.png) |

**每一项定量都附带逐体素置信度。** 每个器官行给出模型的 softmax 最大类概率及其 5% 分位——低分位才是关键，因为误差集中在边界。低于 0.9 的条目会标出；下图这一次运行里，胆囊（`conf 0.88 / p5 0.54`）正是模型最没把握的那个，而它恰好也是 spacing 消融独立测出的最脆弱结构。

![AI 分割与逐体素置信度](docs/img/gui_confidence.png)

| 重建实验室（无需任何数据） | 模型说明卡：出处与适用边界 |
|:---:|:---:|
| ![内置模体重建](docs/img/gui_recon_phantom.png) | ![模型说明卡](docs/img/gui_model_card_zh.png) |

**内置 Shepp-Logan 模体**让整条重建链路在不导入任何数据时就能跑通——V3 是未滤波反投影（糊成一团），V4 是滤波后复现出同一个模体，连最小的病灶都在。模体的真值是解析已知的，因此误差图量的是与真相的距离，而不是与另一次重建的距离。**模型说明卡**写明模型身份如何由实测确证、验证到了什么程度、还有什么未被测量；卡上每个数字都从 `experiments/results/` 现读，重跑实验即自动更新。

> 截图使用 **TotalSegmentator-CT-Lite**（CC-BY-4.0）公开去标识研究数据；本仓库不包含 PHI。模体与说明卡两张则完全不需要数据。

## 核心能力

| 模块 | 能力 |
|---|---|
| **临床阅片** | 解剖排序 DICOM 加载 · 三平面 MPR + 十字线联动 · 6 套窗位预设 + 反色 · 三平面厚层 MIP / MinIP / AIP · 9 个测量和标注工具 · 椭圆 ROI 统计 · 四角 PACS 叠加 · Cine 播放 · 双序列随访对比 |
| **AI 分割** | 25 类后台滑窗 ONNX 推理（含 5 个肺叶）· 三平面彩色叠加与可点图例 · 光标 HUD · 器官统计与 CSV 导出 · marching cubes 三维表面预览、形状特征与 STL 导出 · 画笔/橡皮编辑和撤销 · 逐体素置信度 |
| **重建实验室** | 内置解析 Shepp-Logan 模体 · Radon 投影 · BP / 含 5 种滤波器的 FBP / DFR · 从零实现的 DMR、ART、SIRT · 误差图与 RMSE · 学习式 CNN 后处理，并在界面展示训练视角和输入滤波器限制 |
| **安全与审阅** | 显示层脱敏 · 常驻 AI 免责声明 · 模型说明卡（实测出处及未测边界）· 中英双语界面 |

## 快速开始

```bash
conda env create -f environment.yml     # 创建 Python 3.10 环境
conda activate dicom_gui
python main.py                           # 空载启动
python main.py --data /path/to/dicom_dir # 或启动时加载 DICOM 目录
```

- **CPU-only，无需 GPU。** 在参考机器上整卷 AI 推理约需 100 秒。
- 模型权重（`models/organs.onnx.data`，119 MB）**不随仓库分发**；缺失时自动降级为经典连通域算法。详见[架构说明 → 模型](docs/ARCHITECTURE.md#segmentation-model)。

## 量化证据

实验直接调用随软件发布的产品管线，而非另写一份替代实现。研究 I–III 与 spacing 消融见[技术报告](docs/technical_report.md)，脚本和已提交结果由 [`experiments/`](experiments/README.md) 统一索引。

| 证据线 | 实测结果 | 适用边界 |
|---|---|---|
| **研究 I —— 重建剂量-质量** | 误差在 ≈180 视角后饱和；最优 FBP 滤波器从稀疏角的平滑滤波切换为稠密角的锐利 Ram-Lak；Poisson 光子噪声下 ART 是已测方法中最鲁棒者。 | 解析二维 Shepp-Logan 模体；矩阵法限制在 ≈64×64。[预印本稿](docs/preprint_recon.md) |
| **研究 II —— 模型出处与 Dice** | 标签重叠混淆矩阵将未文档化 ONNX 模型确认为 TotalSegmentator v2 `class_map_part_organs`，并纠正两处标签错误。**20 例**患者级平均 Dice **0.909**（95% CI [0.889, 0.927]），单例 0.922 略偏乐观但落在区间内。 | 器官间可靠性差异远大于总体数字所示：肝 0.982、脾 0.976，而右肺上叶 0.773、前列腺 0.554（仅 7 例在场）。[`seg_multi.py`](experiments/seg_multi.py) |
| **研究 III —— 学习式稀疏角重建** | 自实现 1.9M 参数残差 U-Net 将 RMSE 降低 **3–6 倍**，病灶对比度保留率从 0.87 提升至 **0.96–1.00**；虚假结构率为 1.7%，分布外增益比为 0.81。 | 使用无噪声合成投影；幻觉率是有利条件下的下界，不能外推至光子饥饿的低剂量 CT。 |
| **消融 —— spacing 契约** | 引擎此前跳过了 nnU-Net 必需的「重采样到训练 spacing」。先测代价（spacing 偏离一倍时平均 Dice 由 0.9219 掉到 0.7995，小器官最先垮且非单调），再据此实现。**20 例配对**下同一份失配输入由 **0.684 回升到 0.840**，**20/20 例全部改善**（Wilcoxon *p* = 1.9×10⁻⁶）；随附序列的推理由 100s / 8.8GB 降至 **37s / 3.0GB**。 | 32GB 机器只测得到变粗方向，更细一侧是据「属降采样」推断而非实测。蒙版边界现按 1.5mm 网格量化——结构级准确度升、像素级边界精度降。[`seg_spacing.py`](experiments/seg_spacing.py) |
| **扩展验证 —— 肺叶** | 57 例公开 CT 的五肺叶平均 Dice 为 **0.8867**（95% CI **[0.859, 0.914]**）；右肺上叶为 0.727，而原单例为 0.967。 | 只验证五个肺叶。该结论被独立印证：另一次 20 例运行用不同脚本、不同抽样，把同一个右肺上叶测为 0.773。[`seg3d_eval.py`](experiments/seg3d_eval.py) · [`seg3d_report.py`](experiments/seg3d_report.py) |

## 工程与测试

- 原 God-object 已拆分为 **5 个 UI mixin + 9 个无 Qt 计算模块**。
- 全套 **487 项检查**（需本地研究数据）；CI 跑其中 **396 项数据无关检查**，交互层测试不在 CI 内。
- 重建算法测试断言数值正确性，而非只检查输出“有限”；DICOM 读取对畸形元数据作防御处理。

```bash
python tests/test_gui.py                     # 完整回归：487 项；需本地 RIDER 数据
SKIP_REAL_DATA=1 python tests/test_gui.py    # CI 使用的 396 项数据无关检查
ruff check .                                 # 静态检查
coverage run tests/test_gui.py && coverage report
```

<details>
<summary><strong>覆盖率详情</strong></summary>

离屏 Qt 覆盖率 **86%**（3266 条语句）。九个无 Qt 模块（`recon` 84%、`quantify` 100%、`segmentation` 86%、`mpr_geometry` 96%、`followup` 90%、`projection` 95%、`mesh3d` 96%、`registration` 98%、`model_card` 73%）均有独立单测；合成鼠标 press/move/release 序列会断言信号载荷（`graphics_view` 91%）。此前无人走过的几层提升明显：重建实验室 UI 调度 `recon_lab` 44% → **89%**，标注/分割编辑 `annotation_lab` 74% → **84%**，鼠标交互调度 `interaction.py` 64% → **79%**。写这些断言的过程挖出三个只读代码发现不了的缺陷：光标移出体积后探针仍显示上一次的读数、模型说明卡遇到截断的 CSV 会崩、数字 id 的标注永远渲染不出也删不掉。因此，CI 全绿只代表数据无关子集通过，不等于所有本地数据交互测试均已运行。

</details>

## 文档

| 文档 | 语言 | 用途 |
|---|---|---|
| [架构说明](docs/ARCHITECTURE.md) | 英文 | 模块图、God-object 分解、分割模型出处和 AI 管线契约 |
| [软件说明书](docs/manual_zh.md) · [English](docs/manual_en.md) · [PDF](docs/manual_zh.pdf) | 中文 · EN | 按界面截图讲解全部用户功能 |
| [技术报告](docs/technical_report.md) | 英文 | 研究 I–III 的方法、图表与结果 |
| [预印本稿 —— 研究 I](docs/preprint_recon.md) | 英文 | 学术格式的稀疏视角 / 低剂量重建研究 |
| [实验](experiments/README.md) | 英文 | 可复现实验脚本、图表与 CSV 输出 |
| [变更记录](CHANGELOG.md) | 英文 | 可审计的缺陷修复与审查记录 |
| [第三方许可](THIRD_PARTY_NOTICES.md) | 英文 | 已对上游核实的集成组件许可 |

## 安全边界与已知限制

- **非临床器械：**无监管认证、临床验证档案、审计追踪或访问控制。
- **仅做显示层脱敏：**屏幕与导出文件名会隐藏 PHI，但不会清洗底层 DICOM 标签和烧录文字。
- **AI 泛化仍有未测部分：**肺叶验证 57 例、21 器官验证 20 例，样本量仍小且全部来自同一个公开数据集（1.5mm 各向同性），未覆盖其他扫描协议与设备；器官间可靠性差异远大于总体数字（肝 0.98 vs 前列腺 0.55）。spacing 重采样已接入（见证据表），但更细一侧仍属推断而非实测，且扫描范围过大时会被跳过。输入 spacing **已按 nnU-Net 契约重采样**：不做时 spacing 翻倍会掉 13%（0.922→0.799），小器官最先垮（胆囊 0.82→0.10→0.55）；接入重采样后，**20 例配对验证**下 Dice 由 0.684 回升至 0.840（平均 +0.155，20/20 例全部改善，Wilcoxon p=1.9e-06），且推理由 100s/8.8GB 降至 37s/3.0GB。**代价**：蒙版边界在 1.5 mm 网格上决定，映射回更细的原分辨率后呈阶梯状（0.713 mm 数据实测约 2 像素平台）——结构级准确度升、像素级边界精度降。更细 spacing 一侧仍未直接验证（该侧属降采样，原理上更有利），扫描范围过大时会跳过重采样以免内存溢出。[`seg_spacing.py`](experiments/seg_spacing.py)
- **重建限于教学范围：**DMR / ART 矩阵重建受最小二乘成本限制，实用上限约 64×64。研究 III 使用无噪声合成投影，不能证明低剂量临床表现。
- **随访为刚性而非形变配准：**平面内配准在测试中将整体平移造成的 MAE 从 321 HU 降至 13 HU，但不会校正呼吸引起的器官形变；差异结果只能作定性参考，不能视为临床变化量。

## 许可与版权

© 2026 **盛超（Sheng Chao）、赖胜圣（Lai Shengsheng）**。保留所有权利。

本软件由上述两位著作权人共同享有权利，已向中国版权保护中心提交列明双方的计算机软件著作权登记申请。本仓库仅供教学、科研与作品集审阅，**未授予任何复制、修改或再分发许可**；如需使用，请联系著作权人。集成的第三方组件依其各自许可，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
