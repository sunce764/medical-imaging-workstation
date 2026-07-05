# 变更记录 · 代码审查小结

本文件汇总对**医学影像工作站 Pro + 重建实验室**的一轮系统性缺陷排查（2026-07）。

## 排查方法

每个问题都遵循同一条纪律闭环，不靠猜：

1. **假设 → 真实复现**：用真实数据或构造真实畸形 DICOM/工程文件，在离屏 Qt 下实跑，先证明问题确实触发（崩溃/错序/残留），再动代码。
2. **修复 → 复测**：改完用同一复现脚本验证问题消失。
3. **回归固化**：每个问题写进 `tests/test_gui.py`，防止回潮。
4. **一问题一提交**：提交前核对无 PHI / 大文件混入（`肺癌/`、`*.dcm`、`organs.onnx.data` 均 .gitignore）。

回归套件从最初约 10 项增长到 **64 项检查（20 个测试函数）**，`python tests/test_gui.py` 退出码 0 = 全过（后续工程化再增至 **102 项**，见下节）。

---

## 已修复缺陷

### 崩溃类

| 缺陷 | 触发场景 | 提交 |
|------|----------|------|
| 换病例撤销越界 | 换到更小病例后 `Ctrl+Z` 撤销分割，旧切片号越界 `IndexError` | `48cf8e0` |
| 系统矩阵构建异常卡死 UI | `build_system_matrix` 抛异常时模态进度框不关闭 → UI 冻死，异常还冒泡进按钮槽 | `22219b2` |
| 混合形状 DICOM 加载崩溃 | 同序列内切片矩阵尺寸不一致、或 `SeriesInstanceUID` 全缺失把多序列混为一组 → `np.array` 堆叠 `ValueError` | `7f1ff72` |
| 空值数值标签崩溃 | `getattr` 默认值只在标签缺失时生效；畸形 DICOM 把 RescaleSlope/PixelSpacing/SliceThickness 留空（`None`）→ `float(None)` 崩，序列打不开 | `6175a46` |
| 多帧 / 损坏 DICOM 崩溃 | 多帧单文件 `pixel_array` 为 3D 堆叠成 4D 解包崩；PixelData 截断/缺编解码器 → 一张坏片带崩整卷 | `654023b` |
| 畸形标注 JSON 卡死阅片 | 加载字段缺失/空 points/rect 长度错的工程 → `_render_annotations` 每次刷新都崩 = 阅片卡死 | `bfbab63` |

### 安全 / 隐私

| 缺陷 | 触发场景 | 提交 |
|------|----------|------|
| 导出文件名路径穿越 | `PatientID="../PWNED"` 直接拼进存盘路径 → 文件写到 `Exported_Lesions` 之外；含 `/` 则静默失败丢标注 | `a7c92d6` |
| 脱敏泄露既往检查日期 | 双序列对比 V2 标题在脱敏模式下仍显示既往 `StudyDate` | `48cf8e0` |

### 交互 / 状态一致性

| 缺陷 | 提交 |
|------|------|
| 重建模式切片滑条失效（`on_slice_changed` 只刷非重建态） | `4fddc11` |
| 换切片后链式源图 `_last_recon_img` 残留 | `4fddc11` |
| reset 后分割撤销栈残留 | `48cf8e0` |
| 换病例不停 Cine 播放 | `48cf8e0` |
| DICOM 排序键 float(z坐标)/int(序号)混排打乱解剖顺序 → 改序列级统一判定 | `dab0f44` |
| 关窗不取消后台 AI 推理（8.8GB/100s 滞留 + 完成后回调已拆除窗口 → RuntimeError） | `e715d57` |
| 语言切换后作用域勾选框不翻译（英文态残留中文） | `85bc022` |
| 图例与蒙版叠加显隐不一致（关 Anno 后图例仍列器官） | `bd33a9f` |

### 加固（对齐既有防御约定 / 一致性）

| 加固 | 提交 |
|------|------|
| DMR/ART/SIRT 重建输出经 `_finite_clip` 保证有限，对齐 DFR 的 `nan_to_num` 约定（病态弦图不再产出 NaN 黑图 + NaN RMSE） | `ecf390b` |
| `recon.build_system_matrix` 的 `_mp.cpu_count()` → `os.cpu_count() or 4`，防极端平台 `NotImplementedError` | `a7c92d6` |

### 沉淀的防御工具

后续所有 DICOM/文件名处理都应经过（`main.py` MedicalViewer）：

- **`_dcm_float(ds, tag, default, idx=None)`** —— 安全读 DICOM 数值标签（标签缺失/为空/非数值统一回退，杜绝 `float(None)`）。
- **`_safe_name(s, fallback)`** —— 患者标识净化为安全文件名片段（去路径分隔符与 `..`，杜绝路径穿越；存/取两侧统一，往返一致）。
- **`_valid_anno(a)`** —— 按类型校验标注结构，加载期过滤畸形/旧版本条目。

`_read_dicom_dir` 读盘三重加固：选切片最多的序列 → 按 `(Rows,Columns)` 保留多数形状 → 序列级排序键。

---

## 工程化与架构解耦（2026-07）

在缺陷排查之外做的一轮工程成熟度与架构提升。每步保持**零行为回归**（完整回归全过 + CI 数据无关子集），一步一提交，提交前核对无 PHI。

### 工程化（5 项）

| 项 | 内容 | 提交 |
|------|------|------|
| CI + 打包 | 新增 `pyproject.toml`（元数据/依赖/工具配置）+ GitHub Actions（push/PR 跑 ruff + 数据无关测试子集，离屏 Qt，无需真实数据或 119MB 权重）；测试拆出 `SKIP_REAL_DATA` 子集使 CI 可跑 | `887e2f2` |
| ruff + 类型注解 | 配置 ruff（忽略刻意的紧凑单行风格，专注真问题）+ 修全部真 lint；`recon.py`/`ai_engine.py` 加完整类型注解 | `029b572` |
| i18n 表驱动 | `update_language` 由 ~110 行 `setText` 三元墙改为 `(控件, 英文, 中文)` 表 + `_retranslate_combo` 辅助，根治漏译风险 | `6b0530b` |
| 入口去硬编码 | 删除启动硬编码自动加载 `肺癌/`（PHI 泄漏面）；改 `--data DIR` CLI 参数 + console 入口点，默认空载 | `9d4ff0b` |
| 覆盖率量化 | coverage 接入 pyproject + CI；完整套件覆盖率 **≈66%** | `1486efb` |

### 架构解耦（3 块，累计 4 个无 Qt 纯计算模块）

针对「计算核心缠在 God object 里、无法独立单测」的短板，按同一模式抽出：**纯逻辑 → 无 Qt 独立模块 → mixin/线程退化成薄包装 → 加合成数据的独立单测（进 CI 子集）**。

| 模块 | 从哪抽出 | 逻辑 | 独立单测 | 提交 |
|------|----------|------|----------|------|
| `quantify.py` | `AnnotationMixin` | 器官定量（体积 mL / 平均 HU） | `test_quantify`（覆盖率 100%） | `ed47ab6` |
| `segmentation.py` | `AutoAIEngineThread` | AI 数学降级（肺连通域分割） | `test_lung_fallback` | `e2a9857` |
| `mpr_geometry.py` | 收拢原散落三处的坐标约定 | MPR 坐标换算（hover↔voxel↔crosshair）+ 双序列 z 配准 | `test_mpr_geometry` | `33fec02` |

（`recon.py` 是最早的先例：重建算法本就无 Qt 依赖，实验室脚本可直接 `import`。）

回归套件 **64 → 102 项检查**；GitHub CI 连续 9 次全绿。

---

## 已知限制（如实入档，未修）

- **MPR 各向异性未校正**：冠状/矢状面按像素 1:1 显示，层厚 ≠ 面内像素间距时几何比例失真。修复需重做「scene 坐标 = 体素索引」这一贯穿 hover/测量/十字线的映射，属非手术式改动、风险高于收益，故记为限制。卡尺测量按真实 mm，**测量值正确**，仅显示比例不符解剖。
- **AI 蒙版叠加仅横断面**：冠状/矢状面不显示器官分割叠加。

详见 `README.md` 的「已知限制」。

---

## 定位声明

本软件是**影像教学 / 科研工具**，**非经认证的医疗器械，不得用于临床诊断**。以上修复提升的是软件健壮性与数据安全，不构成任何临床合规认证。
