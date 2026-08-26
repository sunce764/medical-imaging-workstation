# =============================================================================
# 医学影像工作站 Pro + 重建实验室
# Medical Imaging Workstation Pro + Recon Lab
#
# 技术栈：PySide6 (Qt6 Python绑定) + NumPy + pydicom + scikit-image
# 架构：多文件模块化
#   ai_engine.py    — AutoAIEngineThread（后台 AI 推理线程）
#   graphics_view.py — MedicalGraphicsView（影像交互视图组件）
#   recon.py        — 纯计算重建算法（无 Qt 依赖）
#   main.py         — MedicalViewer 主窗口 + 入口
# =============================================================================

import json
import math
import os
import re
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydicom  # 读取 DICOM 医学影像文件格式
from pydicom.uid import CTImageStorage
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

# 子模块导入
import mpr_geometry
import projection
from ai_engine import TARGET_SPACING, AutoAIEngineThread
from annotation_lab import AnnotationMixin
from compare_lab import CompareMixin
from constants import (
    AXIAL,
    CORONAL,
    LABEL_LUT,
    LABELS_JSON,
    LUNG_FALLBACK_LABEL,
    MANUAL_TRACK_LABEL,
    RECON_DL_VIEWS,
    SAGITTAL,
    TOOL_POINTER,
)
from dicom_geometry import SeriesGeometry, analyze_series, voxel_plane_edge_labels
from interaction import InteractionMixin
from recon_lab import ReconLabMixin
from ui_builder import UiBuilderMixin


def _int_tag(ds, name, default=0):
    """读 DICOM 的整数标签，空值与非法值一律回落到 default。

    与 MedicalViewer._dcm_float 同一职责，只是整数侧：getattr(ds, name, default) 的
    默认值【只在标签缺失时生效】，而畸形 DICOM 常见的是标签在、值为空，此时 pydicom
    回读 None，int(None) 抛的是 TypeError。排序键与形状分组都在 _read_dicom_dir 内，
    异常从那里冲出去会绕过 load_data 的回滚，留下 dicom_datasets 与 volume_hu 互不
    对应的半更新状态——比直接崩更糟，因为界面看起来还活着。
    """
    try:
        v = getattr(ds, name, default)
        return default if v is None else int(v)
    except (TypeError, ValueError):
        return default


def is_supported_classic_ct(ds) -> bool:
    """本轮明确支持的入口：classic single-frame CT Image Storage。"""
    try:
        frames = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    except (TypeError, ValueError):
        return False
    return (
        str(getattr(ds, 'Modality', '')).upper() == 'CT'
        and str(getattr(ds, 'SOPClassUID', '')) == str(CTImageStorage)
        and frames == 1
        and not hasattr(ds, 'SharedFunctionalGroupsSequence')
        and not hasattr(ds, 'PerFrameFunctionalGroupsSequence')
    )


# AutoAIEngineThread → 已移至 ai_engine.py
# =========================================================================
# 主窗口：医学影像工作站
# 负责：DICOM 加载、多平面临床阅片渲染、窗位/工具/布局/AI 调度、i18n 重译、键盘导航。
#       UI 构建见 UiBuilderMixin（ui_builder.py），重建实验室见 ReconLabMixin
#       （recon_lab.py），双序列对比见 CompareMixin（compare_lab.py），标注/分割/
#       器官定量见 AnnotationMixin（annotation_lab.py），Cine/MPR 交互见
#       InteractionMixin（interaction.py）。keyPressEvent 因 Qt 重写需留本体（MRO）。
# =========================================================================
class MedicalViewer(QMainWindow, ReconLabMixin, CompareMixin, AnnotationMixin,
                    UiBuilderMixin, InteractionMixin):
    # 临床标准窗宽/窗位预设值（中英文键名均支持，兼容语言切换后的下拉选项）
    # 提为类级常量避免在每次 update_display 中重新构造（MPR 多窗模式下每帧 4 次浪费）
    _WW_PRESETS = {"Lung": 1500, "Medi": 400, "Bone": 1500, "Vasc": 600, "Abdo": 150, "Brain": 80,
                   "肺窗": 1500, "纵隔": 400, "骨窗": 1500, "血管": 600, "腹部": 150, "脑窗": 80}
    _WL_PRESETS = {"Lung": -500, "Medi": 40, "Bone": 400, "Vasc": 150, "Abdo": 30, "Brain": 40,
                   "肺窗": -500, "纵隔": 40, "骨窗": 400, "血管": 150, "腹部": 30, "脑窗": 40}

    def __init__(self, data_dir=None):
        super().__init__()
        self.setWindowTitle("Medical Imaging Workstation Pro + Recon Lab")
        self.resize(1600, 950)

        # --- 影像数据 ---
        self.dicom_datasets = []          # 按 Z 轴位置排序的 pydicom Dataset 列表
        self.current_slice_idx = 0        # 当前显示的切片索引（冗余字段，实际以 current_3d_pos[0] 为准）
        self.views = {}                   # {vid: {'container', 'view', 'cb_plane', ...}} 视图字典

        # --- 工具与标注 ---
        self.active_tool = TOOL_POINTER
        # global_annotations 结构：{'all': [全局标注列表], slice_idx: [该切片标注列表]}
        # 'all' 键下的标注会穿透所有切片显示（由 chk_global_scope 控制新标注归属）
        self.global_annotations = {'all': []}

        # --- 3D 体数据 ---
        self.volume_hu = None             # 完整 HU 值体素数组 shape=(Z, H, W)，float32
        # 四项能力必须分别由 DICOM contract 证明；不能由“数组能堆叠”推断解剖/物理语义。
        self.hu_calibrated = False
        self.canonical_orientation = False
        self.inplane_spacing_valid = False
        self.uniform_z_geometry_valid = False
        self.series_geometry = SeriesGeometry(False, False, False, False, None, None, None)
        self.volume_mask = None           # AI 多器官标签图，shape=(Z,H,W) uint8：0=背景,1-24=器官,255=手动追踪
        self._ai_resampled = None         # 本次推理是否做过 spacing 重采样 (原shape, 送入shape)
        self.volume_conf = None           # 逐体素置信度 uint8（255=1.0）；仅 ONNX 路径产出，数学降级时为 None
        self.organ_names = self._load_organ_labels()  # {类别号: (中文名, 英文名)}，用于图例
        self._organ_stats = []            # 最近一次器官定量结果，供面板显示与 CSV 导出
        self._hidden_organs = set()       # 被用户在图例中点隐的器官类别，渲染时跳过
        self._mask_undo = []              # 分割编辑撤销栈：[(切片号, 编辑前蒙版切片)]，上限 20
        # 只有用户确认清空已有非零 mask 后才为 True；普通全零 placeholder 不能落成 cache hit。
        self._mask_cache_clear_requested = False
        self.is_english = False           # 界面语言，False=中文，True=英文
        self.anonymize = False            # 脱敏模式：显示层隐去患者身份信息（不改底层 DICOM）
        self._anon_session_nonce = ''     # 每次成功 load 后随机生成；匿名导出同 load 内稳定
        default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "Exported_Lesions")
        # 内部工程缓存与用户显式 export 分开建模；测试可分别重定向到临时目录。
        self.persistence_dir = default_output_dir
        self.export_dir = default_output_dir
        self.current_3d_pos = [0, 0, 0]  # [z, y, x]，MPR 联动的三维光标位置

        # --- 双序列随访对比状态（轻量版：独立模式，不改单体数据模型）---
        self.compare_mode_active = False  # 是否处于对比模式，影响 update_display 分支
        self.compare_volume = None        # 对比（既往）序列的 HU 体数据，shape=(Z2,H,W)
        self.compare_datasets = []        # 对比序列的 pydicom Dataset 列表
        self._pre_compare_layout = 0      # 进入对比模式前的布局，退出时还原
        self._primary_zpos = None         # 主序列各层的解剖 z 坐标(mm)，用于对比配准
        self._compare_zpos = None         # 对比序列各层的解剖 z 坐标(mm)

        # --- Cine 电影播放 ---
        self.cine_timer = QTimer(self)    # 自动连续翻片定时器
        self.cine_timer.timeout.connect(self._cine_step)
        self._cine_dir = 1                # 往返(bounce)播放方向：+1 向下 / -1 向上

        # --- 重建实验室状态 ---
        self.recon_mode_active = False    # 是否处于重建实验室 Tab，影响 update_display() 的行为分支
        self._pre_recon_layout = 0        # 进入重建实验室前的布局模式，退出时用于还原
        self._recon_ref_z = None          # V1 参考图上次渲染的切片号；仅当它改变才重置重建流水线
        self.current_sinogram = None      # 当前切片的弦图（Radon 变换结果），shape=(detectors, angles)
        self.current_theta = None         # 弦图对应的角度数组，单位为度
        self._last_recon_img = None       # 最近一次矩阵重建结果（n×n，未放大），作为下次生成弦图的输入源
        self._phantom_img = None          # 内置 Shepp-Logan 模体；非 None 时重建链路以它为源，无需导入任何数据

        # --- AI 引擎状态 ---
        self.ai_thread = None             # AutoAIEngineThread 实例
        # _ai_generation：每次加载新数据自增，回调中比对该值可丢弃旧数据的结果（竞态保护）
        self._ai_generation = 0
        self._ai_state = 'standby'        # 'standby' | 'running' | 'done' | 'failed'
        self._ai_time_ms = 0.0            # 最近一次 AI 推理耗时（毫秒）

        # --- 系统矩阵缓存 ---
        # DMR/ART 都需要构建 A 矩阵，计算代价极高（O(n²) 次 Radon 变换）
        # 当图像尺寸和角度配置不变时，直接复用缓存，避免重复等待
        self._cached_A = None             # 缓存的系统矩阵 A，shape=(n_rays, n*n)
        self._cached_A_key = None         # 缓存的 key=(n, len(theta), theta[0], theta[-1])

        # BP 结果缓存：FBP 需要先运行 BP，两者共享缓存避免重复计算
        self._cached_bp = None            # 缓存的 BP 重建结果
        self._cached_bp_sino = None       # 缓存对应的弦图对象引用（用 is 比较，避免 id() 回收复用风险）

        self.setup_stylesheet()
        self.init_ui()
        self.update_language()

        # 延迟 50ms 执行布局切换，确保 Qt 窗口几何完成初始化再设置 splitter 尺寸
        QTimer.singleShot(50, lambda: self.switch_layout(0))

        # 可选：启动时加载指定 DICOM 目录（由入口 --data 传入或测试显式指定）；默认不加载。
        # 不再硬编码自动加载本地"肺癌"目录——避免开发便利泄进产品入口，也避免误加载患者数据。
        if data_dir and os.path.isdir(data_dir):
            self.load_data(data_dir)

    def _load_organ_labels(self):
        """加载器官名表 organ_labels_candidate.json；缺失时回退到内置高置信名称。
        映射为实测所得=TotalSegmentator class_map_part_organs（见该 JSON 的 _meta 与 experiments/）。"""
        fallback = {5: ("肝", "Liver"),
                    10: ("左肺上叶", "Lung UL (L)"), 11: ("左肺下叶", "Lung LL (L)"),
                    12: ("右肺上叶", "Lung UL (R)"), 13: ("右肺中叶", "Lung ML (R)"),
                    14: ("右肺下叶", "Lung LL (R)"),
                    MANUAL_TRACK_LABEL: ("手动追踪", "Manual"),
                    LUNG_FALLBACK_LABEL: ("肺（降级算法）", "Lungs (fallback)")}
        try:
            with open(LABELS_JSON, encoding='utf-8') as f:
                data = json.load(f)
            names = {int(k): (v.get("name_zh", f"类{k}"), v.get("name_en", f"cls{k}"))
                     for k, v in data.get("labels", {}).items()}
            names[MANUAL_TRACK_LABEL] = ("手动追踪", "Manual")
            # 与 MANUAL_TRACK_LABEL 同理强制注入：它不属于模型类别表，JSON 里不会有，
            # 少了这条，降级结果在图例与定量面板里会显示成「类254」。
            names[LUNG_FALLBACK_LABEL] = ("肺（降级算法）", "Lungs (fallback)")
            return names
        except Exception:
            return fallback

    def toggle_language(self):
        """切换中英文界面，然后刷新所有控件文字。"""
        self.is_english = not self.is_english
        self.update_language()

    @staticmethod
    def _retranslate_combo(combo, en_items, cn_items, e, idx=None):
        """重译纯文本下拉框，保留选中项（blockSignals 防止触发回调）。
        idx=None 时沿用下拉框当前索引；显式传入用于按外部状态（如视图 plane）恢复。"""
        if idx is None:
            idx = combo.currentIndex()
        idx = max(0, idx)
        combo.blockSignals(True); combo.clear()
        combo.addItems(en_items if e else cn_items)
        combo.setCurrentIndex(idx); combo.blockSignals(False)

    def update_language(self):
        """按当前语言(self.is_english)重译所有常驻控件文字。
        绝大多数静态文案集中在下面的 (控件, 英文, 中文) 表里——增/改一条字符串只需
        在表中加一行，避免逐行 setText 那样容易漏译（历史上曾漏译 chk_global_scope）。
        含状态/索引逻辑的控件（下拉框、播放/对比/AI 状态等）在表后单列处理。"""
        e = self.is_english
        self.btn_lang.setText("中" if e else "EN")

        # 静态文案表：(控件, 英文, 中文) —— setText 类
        for w, en, cn in (
            (self.tool_btns['btn_ptr'], "Pan\nProbe", "探针\n拖拽"),
            (self.tool_btns['btn_rul'], "Ruler\nDist", "测距\n卡尺"),
            (self.tool_btns['btn_drw'], "Draw\nPath", "自由\n画笔"),
            (self.tool_btns['btn_rec'], "Rect\nCrop", "矩形\n截取"),
            (self.tool_btns['btn_las'], "Lasso\nMask", "套索\n抠图"),
            (self.tool_btns['btn_trk'], "3D\nTrack", "3D\n追踪"),
            (self.tool_btns['btn_brush'], "Seg\nBrush", "分割\n画笔"),
            (self.tool_btns['btn_erase'], "Seg\nErase", "分割\n橡皮"),
            (self.tool_btns['btn_roi'], "ROI\nStats", "ROI\n密度"),
            (self.btn_import, "Load DICOM Folder", "加载 DICOM 目录"),
            (self.btn_save_proj, "Save Project", "保存标注工程"),
            (self.btn_gen_sino, "Generate Sinogram", "发射射线生成弦图"),
            (self.lbl_oversample, "Sampling:", "采样密度:"),
            (self.btn_dfr, "Direct Fourier (DFR)", "直接傅里叶重建 (DFR)"),
            (self.btn_bp, "Back Projection (BP)", "反投影法 (BP - 未滤波)"),
            (self.lbl_filter_text, "Filter:", "选择滤波器:"),
            (self.btn_fbp, "Filtered BP (FBP)", "滤波反投影 (FBP) 对比"),
            (self.lbl_matrix_size, "Image Size:", "图像尺寸:"),
            (self.lbl_art_method, "Method:", "迭代方法:"),
            (self.lbl_art_iter, "Iterations:", "迭代次数:"),
            (self.btn_dmr, "Direct Matrix Recon (DMR)", "直接矩阵重建 (DMR)"),
            (self.btn_art, "Iterative (ART / SIRT / ASD-POCS)", "迭代重建 (ART / SIRT / ASD-POCS)"),
            (self.lbl_brush, "Brush R:", "画笔半径:"),
            (self.lbl_paint_target, "Paint as:", "画笔目标:"),
            (self.btn_export_stats, "Export Stats CSV", "导出定量 CSV"),
            (self.btn_mesh3d, "3D Surface Preview", "三维重建预览"),
            (self.lbl_disclaimer,
             "⚠ AI results & organ labels are auto-inferred — for reference only, not for diagnosis.",
             "⚠ AI 结果与器官标签为自动推断，仅供参考，非诊断依据。"),
            (self.btn_model_card, "Model Card: Provenance && Limits", "模型说明卡：出处与适用边界"),
            (self.btn_phantom, "Load Shepp-Logan Phantom", "载入 Shepp-Logan 模体"),
            # 英文原作 "Clear Mask" 漏了标注这一半，与中文不对等；此按钮两者都清，补齐
            (self.btn_clear_anno, "Clear Mask && Annotations", "清空蒙版与标注"),
            (self.btn_reset, "Reset Workspace", "重置工作区"),
            (self.lbl_ww_hint, "Right-drag on image to adjust WW/WL", "在图像上右键拖拽可快速调节窗宽/窗位"),
            (self.chk_overlay, "Overlay", "信息叠加"),
            (self.chk_invert, "Invert", "反色"),
            (self.chk_register, "Register", "配准"),
            (self.chk_anon, "De-ID", "脱敏"),
            (self.chk_global_scope, "New anno → all slices", "新标注穿透所有切片"),
        ):
            w.setText(en if e else cn)
        # 模体按钮文案随载入状态切换，上表登记的是"未载入"态，已载入时改写为卸下
        if getattr(self, '_phantom_img', None) is not None:
            self.btn_phantom.setText("Unload Phantom" if e else "卸下模体")

        # 分组框标题表：(分组框, 英文, 中文) —— setTitle 类
        for g, en, cn in (
            (self.grp_proj, "Projection Generation", "X射线投影生成"),
            (self.grp_algo, "Reconstruction Algorithms", "图像重建算法"),
            (self.grp_matrix, "Matrix Recon && Iterative", "直接矩阵重建 && 迭代重建"),
            (self.grp_mon, "Performance Monitor", "算法性能监控"),
            (self.grp_patient, "PATIENT INFO", "患者信息"),
            (self.grp_display, "READING", "阅片"),
            (self.grp_view, "VIEW", "视图"),
            (self.grp_followup, "FOLLOW-UP", "随访对比"),
            (self.grp_data, "DATA && PRIVACY", "数据与隐私"),
            (self.grp_ai, "Automated AI Engine", "自动化 AI 引擎"),
        ):
            g.setTitle(en if e else cn)

        # 工具按钮悬停提示（表驱动）
        _tips = {
            'btn_ptr': ("Pan & probe — drag to pan, click to measure HU, right-drag to adjust WW/WL",
                        "探针/拖拽 — 拖动平移 | 点击测量HU值 | 右键拖拽调节窗宽窗位"),
            'btn_rul': ("Ruler — drag to measure distance (mm)", "测距卡尺 — 拖出直线测量两点距离(mm)"),
            'btn_drw': ("Freehand draw — annotate freely", "自由画笔 — 在图像上自由绘制标注"),
            'btn_rec': ("Rect crop — select ROI to export stats", "矩形截取 — 框选区域导出ROI统计"),
            'btn_las': ("Lasso mask — polygon segmentation", "套索抠图 — 绘制多边形生成分割蒙版"),
            'btn_trk': ("3D track — track structure through slices", "3D追踪 — 框选区域执行三维连通域追踪"),
            'btn_brush': ("Seg brush — paint on the current Axial slice to add to the mask",
                          "分割画笔 — 在当前横断面涂画，补入分割蒙版（修正 AI 遗漏）"),
            'btn_erase': ("Seg erase — wipe mask (incl. AI errors) under the stroke",
                          "分割橡皮 — 擦除涂过处的蒙版（可清除 AI 误分割）"),
            'btn_roi': ("ROI density — drag an ellipse to read mean/SD/min/max HU & area",
                        "ROI 密度 — 拖出椭圆读取内部 均值/标准差/最值 HU 及面积"),
        }
        for key, (tip_en, tip_cn) in _tips.items():
            self.tool_btns[key].setToolTip(tip_en if e else tip_cn)
        self.chk_anon.setToolTip(
            "Display/export-filename de-identification only. DICOM tags, internal project "
            "cache identifiers, and burned-in pixel text are not removed."
            if e else
            "仅隐藏屏幕与显式导出文件名；不会清除 DICOM 标签、内部工程缓存标识或像素烧录文字。")

        self.tabs.setTabText(0, "Clinical Mode" if e else "临床阅片")
        self.tabs.setTabText(1, "Recon Lab" if e else "重建实验室")

        # 保留选中索引的纯文本下拉框
        self._retranslate_combo(self.combo_oversample,
                                ["Std 1×", "High 2×", "Ultra 4×"], ["标准 1×", "高 2×", "超高 4×"], e)
        self._retranslate_combo(self.combo_layout,
                                ["1x1 Single", "1x2 Dual", "2x2 Grid"],
                                ["单窗模式 (1x1)", "双窗对比 (1x2)", "四窗矩阵 (2x2)"], e)

        # 运行耗时占位（仅占位态需重译，已有结果不动）
        if "耗时: --" in self.lbl_time.text() or "Time: --" in self.lbl_time.text():
            self.lbl_time.setText("Run Time: -- ms" if e else "运行耗时: -- ms")
        self._update_organ_stats()  # 语言切换后按新语言重渲染定量面板

        # AI 状态文案随状态机
        if self._ai_state == 'standby':
            self.lbl_ai_status.setText("Status: Standby" if e else "状态: 待机中")
        elif self._ai_state == 'running':
            self.lbl_ai_status.setText("Processing AI Pipeline..." if e else "状态: AI 引擎自动运算中...")
        elif self._ai_state == 'done':
            self.lbl_ai_status.setText(self._ai_done_text())
        elif self._ai_state == 'failed':
            self.lbl_ai_status.setText(self._ai_failed_text())

        # 状态相关按钮
        mpr_on = self.btn_mpr.isChecked()
        self.btn_mpr.setText(("MPR Link: ON" if mpr_on else "MPR Link: OFF") if e
                             else ("MPR 联动: 开启" if mpr_on else "MPR 联动: 关"))
        for b, n in zip(self.preset_btns,
                        (["Lung", "Medi", "Bone", "Vasc", "Abdo", "Brain"] if e
                         else ["肺窗", "纵隔", "骨窗", "血管", "腹部", "脑窗"]), strict=False):
            b.setText(n)
        self.btn_compare.setText(("Exit Compare" if self.compare_mode_active else "Load Comparison") if e
                                 else ("退出对比" if self.compare_mode_active else "加载对比序列"))
        self.btn_cine.setText(("⏸ Pause" if self.cine_timer.isActive() else "▶ Play") if e
                              else ("⏸ 暂停" if self.cine_timer.isActive() else "▶ 播放"))
        # Cine 速度下拉（item 带 ms 数据，单列处理）
        cs = max(0, self.cb_cine_speed.currentIndex())
        self.cb_cine_speed.blockSignals(True); self.cb_cine_speed.clear()
        for _nm, _ms in ((("Slow" if e else "慢"), 250), (("Med" if e else "中"), 120), (("Fast" if e else "快"), 50)):
            self.cb_cine_speed.addItem(_nm, _ms)
        self.cb_cine_speed.setCurrentIndex(cs); self.cb_cine_speed.blockSignals(False)

        # 每视图的平面/窗位下拉 + 显示/锁定复选（cb_plane 按 vdata['plane'] 恢复，非下拉当前索引）
        v_en = ["Global", "Lung", "Medi", "Bone", "Vasc", "Abdo", "Brain"]
        v_cn = ["跟随", "肺窗", "纵隔", "骨窗", "血管", "腹部", "脑窗"]
        plane_en = ["Axial", "Coronal", "Sagittal"]
        plane_cn = ["横断面", "冠状面", "矢状面"]
        # 厚层投影模式：Slice=单层（默认，行为同原来）/ MIP 最大 / MinIP 最小 / AIP 平均
        proj_en = ["Slice", "MIP", "MinIP", "AIP"]
        proj_cn = ["单层", "最大密度", "最小密度", "平均密度"]
        for vdata in self.views.values():
            self._retranslate_combo(vdata['cb_plane'], plane_en, plane_cn, e, idx=vdata['plane'])
            self._retranslate_combo(vdata['preset'], v_en, v_cn, e)
            self._retranslate_combo(vdata['cb_proj'], proj_en, proj_cn, e)
            vdata['sp_thick'].setSuffix(" sl" if e else " 层")
            vdata['sp_thick'].setToolTip("Slab thickness in slices" if e else "投影层块厚度（层数）")
            vdata['chk_anno'].setText("Anno" if e else "显示")

        # 三个解析重建按钮的「为什么禁用」提示（DMR/ART 自造弦图故不在此列）
        for _b in (self.btn_dfr, self.btn_bp, self.btn_fbp):
            _b.setToolTip("Generate the sinogram first — this algorithm reconstructs from it" if e
                          else "需要先点「发射射线生成弦图」——本算法由弦图反解图像")
        # 学习式重建按钮：文案与「为什么禁用 / 有什么限制」的说明都要随语言切换
        self.btn_dl.setText("DL Recon (CNN post-processing)" if e else "深度学习重建 (CNN 后处理)")
        if getattr(self, '_dl_model_ready', False):
            self.btn_dl.setToolTip(
                (f"Generate the sinogram first.\nNote: the model was trained at "
                 f"{RECON_DL_VIEWS} views; other view counts degrade the result.\n"
                 f"Input is forced to Ram-Lak (ramp) FBP — smoothing filters have already "
                 f"discarded the detail, which the network cannot recover.") if e else
                (f"先点「发射射线生成弦图」。\n注意：模型在 {RECON_DL_VIEWS} 视角下训练，"
                 f"用于其他视角数时效果会打折。\n输入强制为 Ram-Lak(ramp) 的 FBP —— "
                 f"平滑滤波器已把细节滤掉，网络无从恢复。"))
        else:
            self.btn_dl.setToolTip(
                (f"Model models/recon_dl_v{RECON_DL_VIEWS}.onnx not found, or onnxruntime "
                 f"is missing. Train and export it via experiments/recon_dl.py.") if e else
                (f"未找到模型 models/recon_dl_v{RECON_DL_VIEWS}.onnx，或未安装 onnxruntime。\n"
                 f"可用 experiments/recon_dl.py 训练并导出。"))

        self._refresh_patient_info()   # 脱敏占位文字随语言刷新
        self.on_slice_changed(self.slider_slice.value())

    def on_tab_changed(self, index):
        """Tab 切换回调：在临床阅片 (index=0) 和重建实验室 (index=1) 之间切换。

        反闪烁设计：setUpdatesEnabled(False) 屏蔽所有绘制事件，整个切换过程只产生最终
        一帧；try/finally 确保异常时 UI 也能恢复正常刷新。具体进入 / 退出逻辑分别
        委托给 _enter_recon_mode / _exit_recon_mode。
        """
        self._stop_cine()   # 切 Tab 停止 Cine 播放
        # 切到重建实验室前先退出对比模式（对比是临床模式专属，还原布局避免冲突）
        if index == 1 and self.compare_mode_active:
            self._exit_compare_mode()
        self.recon_mode_active = (index == 1)
        self.setUpdatesEnabled(False)
        try:
            if self.recon_mode_active:
                self._enter_recon_mode()
            else:
                self._exit_recon_mode()
        finally:
            self.setUpdatesEnabled(True)

    def _apply_grid_visibility(self, mode):
        """根据布局模式（0=单窗，1=双窗，2=四窗）调整 V2/V3/V4 与 bottom_splitter 可见性。
        仅处理 show/hide，不设置 splitter 尺寸；由调用方按上下文决定同步 / 异步执行 setSizes，
        避免在闪烁控制（setUpdatesEnabled）外多调用一次 setSizes 引发额外重绘。
        """
        vs = [self.views[i]['container'] for i in range(1, 5)]
        if mode == 0:
            vs[1].hide(); vs[2].hide(); vs[3].hide(); self.bottom_splitter.hide()
        elif mode == 1:
            vs[1].show(); vs[2].hide(); vs[3].hide(); self.bottom_splitter.hide()
        else:
            vs[1].show(); vs[2].show(); vs[3].show(); self.bottom_splitter.show()

    def _apply_grid_sizes(self, mode):
        """根据布局模式设置 splitter 尺寸（1=均分上行，2=三个 splitter 全部均分；0=无）。"""
        if mode == 1:
            self.top_splitter.setSizes([1000, 1000])
        elif mode == 2:
            self.top_splitter.setSizes([1000, 1000])
            self.bottom_splitter.setSizes([1000, 1000])
            self.main_splitter.setSizes([1000, 1000])

    def set_view_title(self, vid, title):
        """更新指定视图工具栏中的标题标签文字。
        直接使用 create_independent_view 中缓存的 label 引用，避免每次 findChild 遍历视图树。
        """
        try:
            self.views[vid]['title_label'].setText(title)
        except Exception as e:
            print(f"Warning: set_view_title V{vid}: {e}")

    def on_window_changed_by_mouse(self, delta_ww, delta_wl):
        """右键拖拽调节窗宽/窗位。
        拖拽时若当前视图使用预设窗（非"跟随"），自动重置为全局跟随模式，
        防止预设窗覆盖手动调节的值（否则 update_display 会用预设覆盖 slider）。
        """
        if not self.dicom_datasets or self.recon_mode_active:
            return
        new_ww = max(self.slider_ww.minimum(), min(self.slider_ww.maximum(), self.slider_ww.value() + delta_ww))
        new_wl = max(self.slider_wl.minimum(), min(self.slider_wl.maximum(), self.slider_wl.value() + delta_wl))
        self.slider_ww.setValue(new_ww)
        self.slider_wl.setValue(new_wl)
        for vdata in self.views.values():
            if vdata['container'].isHidden():
                continue
            if vdata['preset'].currentText() not in ["Global", "跟随"]:
                # blockSignals 防止 setCurrentIndex 触发 update_display 重入
                vdata['preset'].blockSignals(True)
                vdata['preset'].setCurrentIndex(0)
                vdata['preset'].blockSignals(False)

    def change_active_tool(self, tid):
        """切换全局工具，并同步更新所有视图的 current_tool，确保各视图行为一致。"""
        self.active_tool = tid
        for v in self.views.values():
            v['view'].current_tool = tid

    def _set_brush_radius(self, r):
        """同步所有视图的分割修正画笔/橡皮半径。"""
        for v in self.views.values():
            v['view'].brush_radius = r

    def keyPressEvent(self, event):
        """键盘翻片：↓/PageDown 下一层，↑/PageUp 上一层，空格切换 Cine，Ctrl+Z 撤销分割编辑。"""
        if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self._undo_mask_edit(); return
        if self.volume_hu is not None and not self.recon_mode_active:
            k = event.key()
            if k in (Qt.Key_Down, Qt.Key_PageDown):
                self.slider_slice.setValue(min(self.slider_slice.value() + 1, self.slider_slice.maximum())); return
            if k in (Qt.Key_Up, Qt.Key_PageUp):
                self.slider_slice.setValue(max(self.slider_slice.value() - 1, 0)); return
            if k == Qt.Key_Space:
                self.toggle_cine(); return
        super().keyPressEvent(event)

    def reset_all_states(self):
        """重置工作区到初始状态：恢复单窗布局、默认窗宽窗位、清空所有标注和弦图缓存。
        注意：仅在临床阅片模式（非重建实验室）下调用 update_display，
        避免在重建实验室中意外清空正在查看的重建结果。
        """
        self._stop_cine()               # 重置时停止 Cine 播放
        if self.compare_mode_active:
            self._exit_compare_mode()   # 重置时退出对比模式
        # 布局复位仅在临床模式：recon 模式需保持 2x2，改 combo_layout 会隐藏 V2/3/4
        if not self.recon_mode_active:
            self.combo_layout.setCurrentIndex(0)
        self.slider_ww.setValue(1500); self.slider_wl.setValue(-500)
        self.tool_btns['btn_ptr'].setChecked(True); self.change_active_tool(0)
        self.global_annotations = {'all': []}
        if self.volume_mask is not None:
            self.volume_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
            self.volume_conf = None       # 置信度属于上一次推理，不可跨重置沿用
        self._hidden_organs.clear()
        self._mask_undo = []         # 重置清撤销栈，避免撤销回被清掉的编辑
        self.lbl_hud.setText("")     # 清除光标 HUD 残留文本
        self._update_organ_stats()  # 蒙版已清，定量面板同步清空
        self.btn_mpr.setChecked(False)
        self.current_sinogram = None; self.current_theta = None
        self._last_recon_img = None
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp, self.btn_dl]: b.setEnabled(False)
        # DMR/迭代重建不依赖弦图；判据统一由 _sync_matrix_buttons 给出
        # （模体也是合法源图，见该方法的说明）
        self._sync_matrix_buttons()
        for v in self.views.values():
            v['cb_plane'].setCurrentIndex(AXIAL)
            v['preset'].setCurrentIndex(0)
            v['chk_anno'].setChecked(True)
            v['view']._user_zoomed = False
            v['view'].fitInView(v['view'].scene.sceneRect(), Qt.KeepAspectRatio)
        if not self.recon_mode_active:
            self.update_display()

    def set_window(self, ww, wl):
        """快捷设置窗宽/窗位（供预设按钮调用），触发 slider.valueChanged → update_display。"""
        self.slider_ww.setValue(ww)
        self.slider_wl.setValue(wl)

    def switch_layout(self, m):
        self._apply_grid_visibility(m)
        # setSizes 和 fitInView 合并到同一帧执行，消除两步之间的闪烁间隙
        def _settle():
            self._apply_grid_sizes(m)
            for vd in self.views.values():
                v = vd['view']
                px = v.image_item.pixmap()
                # 只在 pixmap 真实存在时才 fitInView，避免对已清空的视图操作导致 m11 被改变
                if not vd['container'].isHidden() and px and not px.isNull():
                    v.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio)
        QTimer.singleShot(0, _settle)

    def load_data(self, path):
        """加载 DICOM 目录并构建 3D 体积——分四步：读盘 / 构 HU / 加载注解 / 启动 AI。"""
        self._stop_cine()               # 换病例停止 Cine 播放
        if self.compare_mode_active:
            self._exit_compare_mode()   # 新主序列作废旧的对比配对
        # 记住加载前状态：若新目录无法解码，恢复原序列而非留下 dicom_datasets 与 volume_hu
        # 不一致的半更新状态（否则后续按 idx 取切片会越界崩溃）。
        prev_datasets, prev_volume = self.dicom_datasets, self.volume_hu
        prev_geometry = self.series_geometry
        if not self._read_dicom_dir(path):
            # 【失败必须可见】此前这里直接 return：选了空文件夹或放错的目录时，界面
            # 一切不变、也没有任何提示，用户无从判断是加载失败还是加载了但没显示。
            # 而紧邻的另一条失败路径（_build_volume_hu 返回 None）是弹框的——同一个
            # 动作的两种失败，一种说话一种不说话，是更糟的不一致。
            QMessageBox.warning(self, "Load Failed" if self.is_english else "加载失败",
                                "No readable DICOM files were found in this folder."
                                if self.is_english else "该文件夹中没有可读取的 DICOM 文件。")
            return
        pid = self._build_volume_hu()
        if pid is None:
            self.dicom_datasets, self.volume_hu = prev_datasets, prev_volume
            self.series_geometry = prev_geometry
            QMessageBox.warning(self, "Load Failed" if self.is_english else "加载失败",
                                "No decodable image slices in this series."
                                if self.is_english else "该序列没有可解码的图像切片。")
            return
        # 只有 volume 已成功构建、确认新序列成为当前序列后才清旧 readout；读目录或
        # decode 失败的路径在上方返回，必须保留旧序列及其 probe/HUD，不能在 load 开始时清。
        self.lbl_hu_value.setText("")
        self.lbl_hud.setText("")
        # 匿名 token 只与本次成功 load session 绑定，不由 PatientID/UID/hash 推导。
        self._anon_session_nonce = secrets.token_hex(6)
        self._apply_series_capabilities()
        self._load_annotations_json(pid)
        ai_semantics_valid = all((self.hu_calibrated, self.canonical_orientation,
                                  self.inplane_spacing_valid, self.uniform_z_geometry_valid))
        mask_restored = self._load_saved_mask(pid) if ai_semantics_valid else False

        z = self.volume_hu.shape[0]
        self.on_slice_changed(z // 2)
        # 无需等待下一次 mouse move：用新序列中心体素和新 capability 立即重建 HUD；
        # probe 则保持空白，直到用户在新序列上真实探测。
        self._update_hud(*self.current_3d_pos)
        self._sync_matrix_buttons()
        for vd in self.views.values():
            vd['view']._user_zoomed = False   # 新病例回到适配状态
        # 延迟 100ms 做 fitInView，确保 Qt 已完成首次绘制布局再计算缩放
        QTimer.singleShot(100, lambda: [
            vd['view'].fitInView(vd['view'].scene.sceneRect(), Qt.KeepAspectRatio)
            for vd in self.views.values() if not vd['container'].isHidden()
        ])
        if mask_restored:
            # 已从磁盘恢复分割，跳过 ~100s 的 AI 重算。但仍必须作废上一序列可能还在跑的
            # 推理并推进代次——否则它完成时会盖掉这份刚恢复的蒙版（见 _invalidate_running_ai）。
            self._invalidate_running_ai()
            self._ai_state = 'done'
            self._ai_time_ms = 0.0
            self.lbl_ai_status.setStyleSheet("color: #00FF00; font-weight: bold;")
            self.lbl_ai_status.setText(self._ai_done_text())
            self._update_organ_stats()
        elif ai_semantics_valid:
            self._kickoff_ai()
        else:
            # viewer-only 仍显示可解码像素，但不得把未知/非 canonical 几何送入器官 AI。
            self._invalidate_running_ai()
            self._ai_state = 'standby'
            self.lbl_ai_status.setStyleSheet("color: #F1C40F; font-weight: bold;")
            if not self.hu_calibrated:
                msg = ("Viewer only — raw stored values; HU unavailable" if self.is_english
                       else "仅阅片 — 原始存储值；HU 不可用")
            else:
                msg = ("Viewer only — DICOM geometry is not AI-compatible" if self.is_english
                       else "仅阅片 — DICOM 几何不满足 AI 条件")
            self.lbl_ai_status.setText(msg)
            self.btn_export_stats.setEnabled(False)
            self.btn_mesh3d.setEnabled(False)

    def _apply_series_capabilities(self):
        """把纯 geometry contract 映射为本次序列可用的产品能力。"""
        geometry = self.series_geometry
        self.hu_calibrated = geometry.hu_calibrated
        self.canonical_orientation = geometry.canonical_orientation
        self.inplane_spacing_valid = geometry.inplane_spacing_valid
        self.uniform_z_geometry_valid = geometry.uniform_z_geometry_valid
        anatomical_mpr = (self.canonical_orientation and self.inplane_spacing_valid
                          and self.uniform_z_geometry_valid)
        self.btn_mpr.setEnabled(anatomical_mpr)
        for vdata in self.views.values():
            # 非 canonical 输入只显示 acquisition/source voxel plane；数组轴不能改名为
            # Axial/Coronal/Sagittal，也不能让下拉框进入伪解剖重切面。
            vdata['cb_plane'].setEnabled(anatomical_mpr)
            vdata['cb_plane'].setVisible(self.canonical_orientation)
            vdata['preset'].setEnabled(self.hu_calibrated)
            if not self.hu_calibrated:
                # disabled 只阻止新交互，不会清掉上一序列已选中的 Lung/Bone 等文本；
                # 必须主动回到 Global，避免 raw stored values 继续套用 CT-specific WW/WL。
                vdata['preset'].blockSignals(True)
                vdata['preset'].setCurrentIndex(0)
                vdata['preset'].blockSignals(False)
            if not anatomical_mpr:
                vdata['plane'] = AXIAL
                vdata['cb_plane'].setCurrentIndex(AXIAL)
        for button in self.preset_btns:
            button.setEnabled(self.hu_calibrated)
        if not anatomical_mpr:
            self.btn_mpr.setChecked(False)
        self.tool_btns['btn_rul'].setEnabled(self.inplane_spacing_valid)
        if not self.inplane_spacing_valid:
            # disabled 按钮不会撤销已经激活的工具；旧 Ruler 会继续让各 view 用显示用
            # (1,1) unitless spacing 计算并标成 mm，因此必须同步回到 Pointer。
            for vdata in self.views.values():
                vdata['view'].cancel_ruler_preview()
            self.tool_btns['btn_ptr'].setChecked(True)
            self.change_active_tool(TOOL_POINTER)
        for key in ('btn_rec', 'btn_roi'):
            self.tool_btns[key].setEnabled(self.hu_calibrated and self.inplane_spacing_valid)
        self.tool_btns['btn_trk'].setEnabled(all((self.hu_calibrated,
                                                 self.canonical_orientation,
                                                 self.inplane_spacing_valid,
                                                 self.uniform_z_geometry_valid)))
        self.btn_compare.setEnabled(all((self.hu_calibrated, self.canonical_orientation,
                                         self.inplane_spacing_valid,
                                         self.uniform_z_geometry_valid)))

    def _read_dicom_dir(self, path):
        """递归扫描目录并并行读取 DICOM，按 patient-space 投影排序。

        并行策略：用线程池 dcmread 各文件——pydicom 内部 IO + 大量 numpy 解码会释放 GIL，
        线程池在 SSD 上对千张切片可获 4–8× 加速。读盘失败的单个文件静默跳过，
        最终顺序与单线程版本严格一致（统一在所有线程完成后排序）。

        DICOM 排序策略：
          优先使用 dot(ImagePositionPatient, slice normal)（单位 mm）；
          若缺失该 tag，回退到 InstanceNumber（序列编号，精度较低但通用）。
        """
        # 第一阶段：列出所有候选文件（跳过 macOS 隐藏文件）
        file_paths = []
        for r, _d, fs in os.walk(path):
            for f in fs:
                if not f.startswith('.'):
                    file_paths.append(os.path.join(r, f))

        # 第二阶段：线程池并行 dcmread；max_workers 上限设为 16 避免过多线程导致上下文切换开销
        def _safe_read(fp):
            try:
                return pydicom.dcmread(fp)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
            results = list(ex.map(_safe_read, file_paths))

        # 只让明确支持的 classic single-frame CT 进入 pixel decode。Enhanced CT、
        # multi-frame 与非 CT 不能靠“有 PixelData”伪装成 HU/三维 CT 序列。
        datasets = [ds for ds in results
                    if ds is not None and 'PixelData' in ds and is_supported_classic_ct(ds)]
        if not datasets:
            return False
        if len(datasets) > 1 and any(not str(getattr(ds, 'SeriesInstanceUID', '')).strip()
                                     for ds in datasets):
            # 多文件没有 Series UID 时无法证明它们属于同一 acquisition；不发明按目录、
            # shape 或空字符串归组的 heuristic。单文件没有混序风险，可继续 viewer-only。
            return False

        # 多序列目录：按 SeriesInstanceUID 分组，只保留切片最多的序列，
        # 避免把不同序列（定位像、不同重建核等）混叠进同一个体积
        from collections import defaultdict
        groups = defaultdict(list)
        for ds in datasets:
            groups[str(getattr(ds, 'SeriesInstanceUID', ''))].append(ds)
        if len(groups) > 1:
            datasets = max(groups.values(), key=len)
            print(f"检测到 {len(groups)} 个序列，选用切片最多的（{len(datasets)} 张）")

        # 形状一致性过滤：即使同一 SeriesInstanceUID，个别切片的矩阵尺寸也可能不同
        # （扫描中途换重建矩阵）；更常见的是 SeriesInstanceUID 缺失把多个真实序列混成一组。
        # 混合形状会让后续 np.array 堆叠抛 ValueError（主路径 _build_volume_hu 崩溃、
        # 对比路径被 try/except 误判为"无法读取"）。按 (Rows, Columns) 保留数量最多的尺寸，
        # 与上面"选切片最多的序列"同一取舍思路。Rows/Columns 是含 PixelData 时的必填 tag。
        shape_groups = defaultdict(list)
        for ds in datasets:
            # 【getattr 的默认值挡不住空值】标签**缺失**时它给 0，但标签存在而值为空时
            # pydicom 返回 None，int(None) 抛 TypeError——正是 _dcm_float 存在的理由，
            # 而 int 这条路径一直没有对应防护。异常从 _read_dicom_dir 里冲出去，绕过了
            # load_data 的回滚（那只覆盖 _build_volume_hu 返回 None 的情形），于是留下
            # dicom_datasets=新坏序列、volume_hu=旧序列的半更新状态，此后每次切层都在
            # update_display 里越界崩，界面等同废掉。
            shape_groups[(_int_tag(ds, 'Rows'), _int_tag(ds, 'Columns'))].append(ds)
        if len(shape_groups) > 1:
            datasets = max(shape_groups.values(), key=len)
            r0, c0 = getattr(datasets[0], 'Rows', '?'), getattr(datasets[0], 'Columns', '?')
            print(f"检测到 {len(shape_groups)} 种切片尺寸，选用数量最多的（{len(datasets)} 张 {r0}×{c0}）")
        self.series_geometry = analyze_series(datasets)

        # patient-space 投影可用时沿 slice normal 排序；它对 axial/coronal/sagittal 都成立。
        # 几何无法证明时才整列统一回退 InstanceNumber，绝不逐切片混合不同量纲的键。
        if self.series_geometry.sort_indices is not None:
            self.dicom_datasets = [datasets[i] for i in self.series_geometry.sort_indices]
        else:
            self.dicom_datasets = sorted(datasets, key=lambda ds: _int_tag(ds, 'InstanceNumber'))
        return True

    def _build_volume_hu(self):
        """从 dicom_datasets 构建 3D 强度数组，初始化蒙版、3D 光标、切片滑动条。
        成功返回 PatientID；无任何可解码切片时返回 None（由 load_data 提示并中止）。

        只有逐片单位 contract 证明为标准 HU 时才应用 DICOM 线性变换：
          HU = pixel_value × RescaleSlope + RescaleIntercept
        否则整卷保留 raw stored values，且所有 HU consumer 保持关闭。
        """
        ds = self.dicom_datasets[0]
        pid = str(getattr(ds, 'PatientID', 'N/A'))

        # 逐片解码 raw stored values，防御式处理畸形数据（一张坏片不带崩整卷）：
        #   - pixel_array 解码失败（PixelData 截断 / 压缩语法缺编解码器 / group 0028 非法）→ 跳过该片；
        #   - classic CT contract 只接受 2-D single-frame；异常维度同样跳过。
        # decode 后再按实际保留切片重算 geometry，不能让一张已跳过的坏片继续证明 z spacing。
        frames, kept = [], []
        for d in self.dicom_datasets:
            try:
                arr = d.pixel_array
            except Exception as e:
                print(f"跳过无法解码的切片: {e}")
                continue
            raw = arr.astype(np.float32)
            if raw.ndim == 2:
                frames.append(raw); kept.append(d)
        if not frames:
            return None   # 无任何可解码切片
        # 兜底：万一帧尺寸仍不齐（多帧与单帧混合的极端情形），保留数量最多的尺寸
        shape_count = {}
        for f in frames:
            shape_count[f.shape] = shape_count.get(f.shape, 0) + 1
        dom = max(shape_count, key=shape_count.get)
        pairs = [(f, k) for f, k in zip(frames, kept, strict=False) if f.shape == dom]
        postdecode = analyze_series([k for _, k in pairs])
        if postdecode.sort_indices is not None:
            pairs = [pairs[i] for i in postdecode.sort_indices]
            postdecode = analyze_series([k for _, k in pairs])
        self.series_geometry = postdecode
        self.dicom_datasets = [k for _, k in pairs]
        self._refresh_patient_info()   # 按脱敏状态填患者面板
        # calibration 是 decode 后实际序列的单位合约：任一保留层无法证明时，整卷 raw；
        # 全部有效时才逐 slice 应用各自 slope/intercept，绝不构造混合单位 volume。
        self.volume_hu = np.array([
            raw * float(d.RescaleSlope) + float(d.RescaleIntercept)
            if postdecode.hu_calibrated else raw
            for raw, d in pairs
        ])
        self.volume_mask = np.zeros_like(self.volume_hu, dtype=np.uint8)
        self._mask_cache_clear_requested = False  # 新序列的全零 mask 只是 AI placeholder
        # 换序列必须一并作废置信度：两个序列 shape 常常相同（都是 512²），
        # quantify 的 shape 校验挡不住，旧序列的置信度会被安到新序列头上
        self.volume_conf = None
        self.global_annotations = {'all': []}
        self._hidden_organs.clear()   # 换病例时清除上一例的图例隐藏状态
        self._mask_undo = []          # 换病例清撤销栈，防止旧切片号越界访问新蒙版
        # 定量结果同样属于上一例：不清的话，走 _kickoff_ai 的那 ~100 秒里，患者面板与
        # 影像已是新病例、而定量面板与「导出定量 CSV」仍是上一例的体积/HU——导出的文件
        # 名用新 pid，内容却是旧病例，事后无从分辨。mask_restored 分支之所以看不出问题，
        # 是因为它紧接着调了 _update_organ_stats()；_kickoff_ai 分支要等推理回调才调。
        self._organ_stats = []
        z, y, x = self.volume_hu.shape
        # 默认将 3D 光标定位在体积中心（中间切片、中间行、中间列）
        self.current_3d_pos = [z // 2, y // 2, x // 2]
        self.slider_slice.setRange(0, z - 1)
        self.slider_slice.setValue(z // 2)
        return pid

    def _invalidate_running_ai(self):
        """作废仍在运行的推理并推进代次，返回新代次。

        【换数据时无论要不要启动新推理，都必须调用】这两件事原本只写在 _kickoff_ai 里，
        而 load_data 在磁盘已有缓存蒙版时会跳过 _kickoff_ai 直接用恢复的结果。后果有二：
        上一序列的推理继续跑到底（~8.8GB 不释放），而且代次没变——它完成时
        on_auto_ai_finished 的 generation 比对会【放行】，旧序列的蒙版覆盖掉新序列刚从
        磁盘恢复的那份，界面还照常显示绿色的「检出 N 个器官」。shape 比对也挡不住：
        两个序列常常都是 512²。触发条件是用户在那 ~100 秒里重新加载一次目录。
        """
        if self.ai_thread is not None and self.ai_thread.isRunning():
            self.ai_thread.cancel()
        self._ai_generation += 1
        return self._ai_generation

    def _kickoff_ai(self):
        """启动后台 AI 推理。
        每次加载新数据时自增 generation 计数器；旧 AI 线程回调时若 generation 不匹配则静默
        丢弃结果，防止旧数据覆盖新数据的蒙版（竞态条件保护）。
        并作废上一个仍在运行的推理线程，避免多个 ~8.8GB 推理并发叠加导致内存翻倍/OOM。
        """
        gen = self._invalidate_running_ai()
        self._mask_cache_clear_requested = False  # AI pending 的全零 mask 不是用户 explicit empty
        self._ai_state = 'running'
        self.lbl_ai_status.setStyleSheet("color: #F1C40F; font-weight: bold;")
        self.lbl_ai_status.setText("Processing AI Pipeline..." if self.is_english else "状态: AI 引擎自动运算中...")
        # lambda 中用 g=gen 捕获当前 generation 值（闭包变量，防止后续自增影响比对）
        # 体素物理间距 (z, y, x)：引擎据此重采样到模型的训练 spacing（nnU-Net 推理契约）。
        # 必经 _dcm_float——畸形 DICOM 的 None/非有限值会让 float() 直接崩，且此处一旦
        # 拿到坏值，重采样会按错误的物理尺寸缩放，比不重采样更糟。
        ds0 = self.dicom_datasets[0] if self.dicom_datasets else None
        spacing = None if ds0 is None else (
            self._slice_spacing(),
            self._dcm_float(ds0, 'PixelSpacing', 0.0, idx=0),
            self._dcm_float(ds0, 'PixelSpacing', 0.0, idx=1))
        if spacing is not None and not all(s > 0 for s in spacing):
            spacing = None      # 缺 tag 或值非法 → 视作未知，引擎自会跳过重采样
        self.ai_thread = AutoAIEngineThread(
            self.volume_hu,
            spacing=spacing,
            callback=lambda mask, t, g=gen: self.on_auto_ai_finished(mask, t, g),
            progress_callback=lambda d, t, g=gen: self._on_ai_progress(d, t, g),
            failed_callback=lambda why, g=gen: self._on_ai_failed(why, g)
        )
        self.ai_thread.start()

    def _slice_spacing(self):
        """返回由 patient-space 相邻位置证明的 uniform slice spacing；否则 0。

        ``SliceThickness`` 是准直厚度，``SpacingBetweenSlices`` 也不能替代本次有序
        栈的实测几何。未知、重复或不规则位置必须保持 unavailable，不能填默认单位。
        """
        geometry = getattr(self, 'series_geometry', None)
        if geometry is None or not geometry.uniform_z_geometry_valid:
            return 0.0
        return float(geometry.slice_spacing_mm or 0.0)

    def _on_ai_progress(self, done, total, generation):
        """AI 滑窗推理进度回调（经 Qt 信号 QueuedConnection 投递到主线程）。仅更新当前代的进度显示。"""
        if generation != self._ai_generation or self._ai_state != 'running':
            return
        pct = int(100 * done / total) if total else 0
        self.lbl_ai_status.setText(
            f"AI Segmenting... {pct}%" if self.is_english else f"状态: AI 分割中... {pct}%")

    def _ai_done_text(self):
        """AI 完成后的状态文案：检出的器官类别数，以及是否发生过 spacing 重采样。

        重采样这件事必须让用户看见：它提高了结构级准确度（模型回到训练工况），
        但蒙版的边界是在 1.5mm 网格上决定的，映射回原分辨率后呈阶梯状——
        实测 0.713mm 数据上边界平台约 2 像素。用户在原图上看到锯齿边缘时，
        应当知道原因，而不是以为分割质量出了问题。
        """
        n = 0
        if self.volume_mask is not None:
            ids = np.unique(self.volume_mask)
            n = int(((ids != 0) & (ids != MANUAL_TRACK_LABEL)).sum())
        if getattr(self, '_ai_fallback', False):
            # 降级时【不】沿用「检出 N 个器官」：那句话与 25 类模型跑通时完全一样，
            # 而实际拿到的是连通域算法分出的双肺，既不分左右也不是模型输出。
            return (f"AI unavailable — classical fallback: lungs only, not model output "
                    f"({self._ai_time_ms:.0f}ms)" if self.is_english
                    else f"AI 不可用 — 已降级为经典算法：仅双肺，非模型输出 "
                         f"({self._ai_time_ms:.0f}ms)")
        txt = (f"Ready: {n} organs ({self._ai_time_ms:.0f}ms)" if self.is_english
               else f"状态: 检出 {n} 个器官 ({self._ai_time_ms:.0f}ms)")
        rs = getattr(self, '_ai_resampled', None)
        if rs:
            txt += (f" · resampled to {TARGET_SPACING}mm for inference; mask edges are "
                    f"quantised to that grid" if self.is_english
                    else f" · 推理前已重采样至 {TARGET_SPACING}mm，蒙版边界按该网格量化")
        return txt

    def _ai_failed_text(self):
        """AI 失败文案。措辞刻意区别于「检出 0 个器官」——后者意味着跑成功了但没找到，
        与彻底失败是两回事，混为一谈会误导用户以为这张影像里真的没有器官。"""
        return ("AI segmentation failed — see console; manual tools still available"
                if self.is_english else "状态: AI 分割失败 —— 详见控制台；手动工具仍可用")

    def _on_ai_failed(self, why, generation=None):
        """AI 推理彻底失败（含兜底路径）的回调，经 Qt 信号投递到主线程。

        不弹模态框：分割失败不阻断阅片，测量/标注/重建实验室都还能用，弹窗只会打断。
        故用醒目的红色状态文字 + 控制台详情。"""
        if generation is not None and generation != self._ai_generation:
            return   # 过时的失败回调（用户已切到新数据），静默丢弃
        self._ai_state = 'failed'
        self.lbl_ai_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
        self.lbl_ai_status.setText(self._ai_failed_text())

    def on_auto_ai_finished(self, final_mask, time_ms, generation=None):
        """AI 推理完成的回调（由 Qt 信号 QueuedConnection 投递到主线程执行）。

        投递机制勿改为 QTimer.singleShot：它依附调用它的子线程，而子线程无 Qt 事件循环，
        回调根本不 fire（本项目踩过此坑，AI 蒙版曾从未真正显示）。见 ai_engine._AISignals。

        generation 比对：防止旧数据的 AI 结果在新数据加载后才回调，覆盖新数据的蒙版。
        shape 比对：防止数组维度不匹配导致后续 volume_mask 操作越界。
        recon_mode_active 检查：若用户已切换到重建实验室，不触发 update_display，
          避免破坏正在展示的重建结果（V2/V3/V4 的弦图和重建图像）。
        """
        if generation is not None and generation != self._ai_generation:
            return  # 过时的 AI 回调，静默丢弃
        if self.volume_hu is None or final_mask.shape != self.volume_hu.shape:
            return  # 数据已重置或维度不匹配，安全退出
        self._ai_state = 'done'
        self._ai_time_ms = time_ms
        self._mask_cache_clear_requested = False
        self.volume_mask = final_mask
        # 整卷换蒙版必须同时清撤销栈：栈里存的是【推理开始前】那一版的切片快照，
        # 推理期间用户完全可以画笔编辑（无任何守卫阻止）。不清的话，AI 回来后
        # 按一次 Ctrl+Z 就会把该层的 AI 分割整层覆盖回旧快照——器官体积静默变小
        # 而界面无任何提示。重置(_reset)与换病例(load_data)两处早已这么做，
        # 这条路径当时漏了。
        self._mask_undo = []
        # 逐体素置信度由引擎作为实例属性带出（见 ai_engine.confidence 的说明）。
        # 形状不符或走了数学降级路径时置 None——定量表据此决定是否显示置信度列。
        self._ai_resampled = getattr(self.ai_thread, 'resampled_from', None)
        self._ai_fallback = bool(getattr(self.ai_thread, 'used_fallback', False))
        cf = getattr(self.ai_thread, 'confidence', None)
        self.volume_conf = cf if (cf is not None and cf.shape == final_mask.shape) else None
        # 降级不是成功：绿色是「25 类模型跑通了」的语义，此处改用琥珀色，配合
        # _ai_done_text 里的文案，让「拿到的不是 AI 结果」在状态栏一眼可见。
        self.lbl_ai_status.setStyleSheet("color: #FFC107; font-weight: bold;" if self._ai_fallback
                                         else "color: #00FF00; font-weight: bold;")
        self.lbl_ai_status.setText(self._ai_done_text())
        self._update_organ_stats()
        if not self.recon_mode_active:
            self.update_display()

    def closeEvent(self, event):
        """关窗收尾：取消仍在运行的后台 AI 推理，停止 Cine 定时器。
        动机：AI 单次推理约 8.8GB / ~100s，关窗若不取消，线程会继续占内存，且完成后
        经 Qt 信号回调到已拆除的窗口（对已删除的 QLabel setText）→ RuntimeError。"""
        if self.ai_thread is not None:
            self.ai_thread.cancel()
        self._stop_cine()
        super().closeEvent(event)

    def on_slice_changed(self, idx):
        """切片滑动条 valueChanged 回调：更新 3D 光标 Z 轴坐标并刷新显示。
        无条件 update_display：临床/对比/重建三种模式各走自己的分支，确保切片滑条在
        重建实验室也能切换 V1 参考层（_render_recon_reference 靠 _recon_ref_z 判定是否重置流水线）。"""
        self.current_3d_pos[0] = idx
        self.lbl_slice.setText(f"{'Slice: ' if self.is_english else '层数: '}{idx + 1} / {len(self.dicom_datasets)}")
        self.update_display()

    def update_display(self):
        """核心显示刷新函数：根据当前模式选择重建实验室分支或临床阅片分支。

        重建实验室模式：仅更新 V1（参考切片），并清空 / 禁用 V2-V4 重建流水线。
        临床阅片模式：对每个可见视图按平面切取 2D 截面、做窗宽窗位映射，
        叠加 AI 蒙版、渲染标注、更新 MPR 十字线。
        """
        if self.volume_hu is None:
            return
        z, y, x = self.current_3d_pos

        if self.recon_mode_active:
            self._render_recon_reference(z)
            return

        if self.compare_mode_active and self.compare_volume is not None:
            self._render_compare()
            return

        ww_m, wl_m = self.slider_ww.value(), self.slider_wl.value()
        self.lbl_ww.setText(f"WW: {ww_m}"); self.lbl_wl.setText(f"WL: {wl_m}")
        ds = self.dicom_datasets[z]
        # 【行距与列距必须分开取】DICOM 的 PixelSpacing = [行间距, 列间距]，对应图像的
        # 垂直(Y)与水平(X)。此前全链路只取 idx=0 并把它同时当行距与列距用，面内各向
        # 异性时水平方向的 mm 换算、以及横断面的显示长宽比都会错。本项目现有数据
        # （RIDER 与 TotalSegmentator-Lite）面内恰好都是方形像素，所以这个错误在本地
        # 永远暴露不出来——与 _slice_spacing 里记下的那类坑同源。
        ps_row = self._dcm_float(ds, 'PixelSpacing', 1.0, idx=0)   # 行间距 → 垂直/Y
        ps_col = self._dcm_float(ds, 'PixelSpacing', ps_row, idx=1)  # 列间距 → 水平/X
        px_sp = ps_row      # 单值代表：仅用于层间距估算与四角叠加的 "Px" 一栏
        # 只有 uniform patient-space gaps 才是物理 z scale。viewer-only 的显示布局可以
        # 使用 1.0 像素比例，但 UI/测量不得把这个显示比例冒充成 mm。
        slice_thick = self._slice_spacing() or 1.0

        for vdata in self.views.values():
            if vdata['container'].isHidden():
                continue
            self._render_clinical_plane(vdata, z, y, x, ww_m, wl_m, px_sp, slice_thick,
                                        ps_row=ps_row, ps_col=ps_col)

        # 图例集中判定：仅当存在可见 Axial 视图开启 Anno 且蒙版有内容时才显示检出器官，
        # 否则清空——避免"关掉 Anno 后叠加已隐藏、图例却仍列着器官"的残留不一致。
        show_legend = self.volume_mask is not None and any(
            vd['plane'] == AXIAL and vd['chk_anno'].isChecked() and not vd['container'].isHidden()
            for vd in self.views.values())
        if show_legend:
            present = np.unique(self.volume_mask[z])
            self._update_legend(present[present != 0])
        else:
            self._update_legend([])

    def _patient_display(self):
        """返回用于显示的 (ID, 姓名, 年龄)；脱敏模式下隐去真实身份（仅显示层，不改 DICOM）。"""
        if not self.dicom_datasets:
            return ("N/A", "N/A", "")
        if self.anonymize:
            return ("ANON", "Anonymized" if self.is_english else "已脱敏", "")
        ds = self.dicom_datasets[0]
        pid = str(getattr(ds, 'PatientID', 'N/A'))
        name = str(getattr(ds, 'PatientName', '') or 'N/A').replace('^', ' ')
        age = str(getattr(ds, 'PatientAge', '') or '')
        return (pid, name, age)

    def _refresh_patient_info(self):
        """按当前脱敏状态刷新右侧患者信息面板。"""
        pid, name, age = self._patient_display()
        self.info_labels["ID"].setText(pid)
        self.info_labels["NAME"].setText(name)
        self.info_labels["AGE"].setText(age if age else "N/A")

    def _toggle_anonymize(self, on):
        """切换脱敏：刷新患者面板与四角叠加（overlay 经 update_display 重建）。"""
        self.anonymize = on
        self._refresh_patient_info()
        if not self.recon_mode_active:
            self.update_display()

    @staticmethod
    def _dcm_float(ds, tag, default, idx=None):
        """安全读取 DICOM 数值标签为 float：标签缺失、为空(None)、非有限值(NaN/±Inf)、
        或无法转 float 时返回 default。idx 非空时取序列第 idx 个元素（如 PixelSpacing[0]）。

        动机一（None）：getattr 的默认值只在属性【缺失】时生效；畸形 DICOM 常把数值标签
        留空（pydicom 读回 None），此时 float(None) / None[idx] 会抛 TypeError，导致
        加载/显示/定量全线崩溃。

        动机二（NaN/Inf）：NaN 是【合法的 float】，上面的 None 检查与 float() 都拦不住它，
        会一路静默流到下游——这比崩溃更糟：
          · RescaleSlope=NaN → HU 全 NaN → 弦图 100% 非有限，而 BP/FBP/DFR 照常「跑通」
            出图，界面无任何异常提示，用户看到的是从垃圾数据算出来的图；
          · PixelSpacing=NaN → 所有距离/面积/体积测量静默变成 nan。
        实测确认过上述两条链路，故在此一并兜住：非有限值同样退回 default。"""
        v = getattr(ds, tag, None)
        if v is None:
            return default
        try:
            if idx is not None:
                v = v[idx]
            f = float(v)
        except (TypeError, ValueError, IndexError):
            return default
        return f if math.isfinite(f) else default

    @staticmethod
    def _safe_name(s, fallback="Unknown"):
        """把患者标识净化为安全文件名片段。患者 DICOM 数据不可信：PatientID/Name
        可能含 '/'、'\\' 或 '..'，若直接拼进路径会导致存盘失败（子目录不存在）甚至
        路径穿越写到导出目录之外。仅保留字母数字/中日韩等词字符与 . _ -，其余替换为 _，
        再去掉首尾点（杜绝 '..' 与隐藏文件），并限长避免 ENAMETOOLONG。
        存/取两侧都经此函数，同一 PatientID 恒定映射到同一文件名，保证往返一致。"""
        s = re.sub(r'[^\w.\-]', '_', str(s), flags=re.UNICODE).strip('. ')
        return (s[:64] or fallback)

    def _export_tag(self):
        """导出文件名用的患者标识；脱敏时返回匿名前缀，防止文件名泄露姓名。"""
        if self.anonymize or not self.dicom_datasets:
            nonce = self._anon_session_nonce or secrets.token_hex(6)
            self._anon_session_nonce = nonce
            return f"ANON-{nonce}"
        name = str(getattr(self.dicom_datasets[0], 'PatientName', 'P')).replace('^', '_')
        return self._safe_name(name, fallback="P")

    @staticmethod
    def _unique_export_path(directory, filename):
        """返回不会静默覆盖已有文件的安全路径（name, name_2, ...）。"""
        stem, ext = os.path.splitext(filename)
        candidate = os.path.join(directory, filename)
        n = 2
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{stem}_{n}{ext}")
            n += 1
        return candidate

    def _toggle_overlay(self, on):
        """切换所有视图的 DICOM 信息叠加显隐。"""
        for vd in self.views.values():
            vd['view'].show_overlay = on
            vd['view'].viewport().update()

    def _apply_dicom_overlay(self, vdata, plane, z, y, x, ww, wl, px_sp, slice_thick,
                             ps_row=None, ps_col=None):
        """构建并下发 PACS 风格的四角信息叠加与解剖方位字母。"""
        e = self.is_english
        ds0 = self.dicom_datasets[0]
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape
        idx, tot = {AXIAL: (z, Z_MAX), CORONAL: (y, Y_MAX), SAGITTAL: (x, X_MAX)}[plane]
        if self.canonical_orientation:
            pname = ({AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"} if e else
                     {AXIAL: "横断面", CORONAL: "冠状面", SAGITTAL: "矢状面"})[plane]
        else:
            pname = "Source voxel plane" if e else "原始体素平面"
        zoom = vdata['view'].transform().m11() * 100
        pid, pt_name, age = self._patient_display()   # 脱敏时隐去真实身份
        tl = [f"ID: {pid}", pt_name] + ([f"Age: {age}"] if age else [])
        z_text = (f"Z spacing {slice_thick:.2f}mm" if e else f"层间距 {slice_thick:.2f}mm") \
            if self.uniform_z_geometry_valid else ("Z spacing unavailable" if e else "层间距不可用")
        if self.inplane_spacing_valid:
            row = px_sp if ps_row is None else ps_row
            column = row if ps_col is None else ps_col
            px_text = (f"Px {row:.2f}mm" if np.isclose(row, column)
                       else f"Px {row:.2f}×{column:.2f}mm")
        else:
            px_text = "Px unavailable" if e else "像素间距不可用"
        corners = {
            'tl': tl,
            'tr': [f"{getattr(ds0, 'Modality', 'CT')}  ·  V{vdata['view'].view_id}", pname],
            'bl': [f"W: {int(ww)}  L: {int(wl)}", f"Zoom: {zoom:.0f}%"],
            'br': [f"{'Slice' if e else '层'} {idx + 1}/{tot}", z_text, px_text],
        }
        # 解剖方位字母：Axial 图像左=解剖右(R)；冠/矢状面上=头(S)下=足(I)
        if self.canonical_orientation:
            orient = ({AXIAL: voxel_plane_edge_labels(ds0.ImageOrientationPatient),
                       CORONAL: {'top': 'S', 'bottom': 'I', 'left': 'R', 'right': 'L'},
                       SAGITTAL: {'top': 'S', 'bottom': 'I', 'left': 'A', 'right': 'P'}}[plane])
        else:
            orient = {}
        vdata['view'].set_overlay(corners, orient)

    def _render_clinical_plane(self, vdata, z, y, x, ww_m, wl_m, px_sp, slice_thick,
                               ps_row=None, ps_col=None):
        """临床阅片分支：渲染单个视图的 2D 截面 + 蒙版 + 标注 + 十字线。"""
        plane = vdata['plane']
        pre = vdata['preset'].currentText()

        # 窗宽/窗位来源：优先使用各视图独立预设，否则跟随全局滑动条
        if not self.hu_calibrated or pre in ["Global", "跟随"]:
            ww, wl = ww_m, wl_m
        else:
            ww, wl = self._WW_PRESETS.get(pre, ww_m), self._WL_PRESETS.get(pre, wl_m)

        # 根据平面切取对应的 2D 截面
        # 像素间距 sp=(行间距, 列间距)=(垂直/Y轴, 水平/X轴)，供 ruler 测距按真实 mm 换算
        # （graphics_view 里 d=√((dx·sp[1])²+(dy·sp[0])²)，故 sp[0] 配垂直、sp[1] 配水平）
        # 厚层投影：下拉为「单层」(index 0) 时走原路径，逐元素等价于直接切片；
        # 选到 MIP/MinIP/AIP 才按厚度取层块投影（projection.project 已保证 thickness=1 时一致）。
        pmode = ['slice', 'max', 'min', 'mean'][vdata['cb_proj'].currentIndex()]
        pthick = vdata['sp_thick'].value()
        idx_of = {AXIAL: z, CORONAL: y, SAGITTAL: x}
        if pmode != 'slice' and pthick > 1:
            hu = projection.project(self.volume_hu, plane, idx_of[plane], pthick, pmode)
        elif plane == AXIAL:
            hu = self.volume_hu[z, :, :]
        elif plane == CORONAL:
            hu = self.volume_hu[:, y, :]     # (Z, X)：垂直=Z→SliceThickness，水平=X→PixelSpacing
        else:                                # SAGITTAL
            hu = self.volume_hu[:, :, x]     # (Z, Y)：垂直=Z→SliceThickness，水平=Y→PixelSpacing
        if plane != AXIAL:
            # volume z 随 patient S 方向递增；显示需将 superior 放在 screen top，
            # 与 overlay 的上 S / 下 I 以及 hover/crosshair 坐标约定保持一致。
            hu = np.flipud(hu)
        # sp = (垂直/Y 的 mm 每像素, 水平/X 的 mm 每像素)，见 graphics_view 的卡尺换算。
        #   Axial    垂直=行→ps_row，水平=列→ps_col
        #   Coronal  垂直=Z→层间距，水平=X(列)→ps_col
        #   Sagittal 垂直=Z→层间距，水平=Y(行)→ps_row
        r = px_sp if ps_row is None else ps_row
        c = px_sp if ps_col is None else ps_col
        sp = (r, c) if plane == AXIAL else (slice_thick, c if plane == CORONAL else r)

        # 窗宽窗位映射：HU → [0, 255] 线性映射
        img = np.clip(hu, wl - ww / 2, wl + ww / 2)
        img = ((img - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        if self.chk_invert.isChecked():
            img = 255 - img  # 反色（黑白反转），观察骨/软组织边界常用
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()

        # AI 多器官蒙版叠加：三个平面均支持。volume_mask 与 volume_hu 同形状(Z,H,W)，
        # 故按与上方 hu 完全相同的索引取对应平面的蒙版切片，保证叠加与影像逐像素对齐。
        # 用调色板 LUT 一步向量化上色，每个类别号映射到 constants.LABEL_LUT 的 RGBA（0=背景全透明）。
        mq = None
        if vdata['chk_anno'].isChecked() and self.volume_mask is not None:
            if plane == AXIAL:
                sm = self.volume_mask[z, :, :]
            elif plane == CORONAL:
                sm = self.volume_mask[:, y, :]
            else:                                  # SAGITTAL
                sm = self.volume_mask[:, :, x]
            if plane != AXIAL:
                sm = np.flipud(sm)
            present = np.unique(sm)
            present = present[present != 0]  # 剔除背景，得到本切片出现的器官类别
            if present.size:  # 性能优化：无器官时跳过 QImage 构建
                lut = LABEL_LUT
                if self._hidden_organs:
                    lut = LABEL_LUT.copy()
                    for lid in self._hidden_organs:
                        lut[lid] = 0  # 被隐藏器官置为全透明
                ov = np.ascontiguousarray(lut[sm])  # (h,w,4) RGBA
                mq = QImage(ov.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            # 图例统一在 update_display 末尾按全局状态刷新（此处不再各视图分别刷，避免多视图相互覆盖）

        vdata['view'].set_image(QPixmap.fromImage(qimg), mq, sp)
        self._apply_dicom_overlay(vdata, plane, z, y, x, ww, wl, px_sp, slice_thick,
                                  ps_row=ps_row, ps_col=ps_col)
        vdata['view'].clear_annotations()  # 清除上一帧的标注图元，防止重影

        if plane == AXIAL and vdata['chk_anno'].isChecked():
            self._render_annotations(vdata, z, sp)

        # MPR 十字准线：联动开启时各平面投影不同的坐标轴对
        if self.btn_mpr.isChecked():
            cx, cy = mpr_geometry.voxel_to_crosshair(plane, z, y, x, self.volume_hu.shape)
            vdata['view'].draw_crosshair(cx, cy)
        else:
            vdata['view'].draw_crosshair(0, 0, show=False)

    def select_folder(self):
        """打开文件夹选择对话框，选择后触发 DICOM 加载。"""
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p:
            self.load_data(p)

    # =========================================================================
    # 矩阵重建共用工具
    # =========================================================================
def main():
    """程序入口。可选 --data DIR：启动即加载该 DICOM 目录（默认不加载任何数据）。"""
    import argparse
    import multiprocessing

    # freeze_support：多进程 'spawn' 模式在 macOS/Windows 打包环境中必须调用，
    # 防止子进程重入主程序逻辑导致无限递归启动
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="医学影像工作站 Pro + 重建实验室")
    parser.add_argument("--data", metavar="DIR", default=None,
                        help="启动时加载的 DICOM 目录（可选，默认不加载任何数据）")
    args, qt_args = parser.parse_known_args()
    app = QApplication(sys.argv[:1] + qt_args)
    window = MedicalViewer(data_dir=args.data)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
