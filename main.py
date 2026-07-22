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
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydicom  # 读取 DICOM 医学影像文件格式
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

# 子模块导入
import mpr_geometry
from ai_engine import AutoAIEngineThread
from annotation_lab import AnnotationMixin
from compare_lab import CompareMixin
from constants import (
    AXIAL,
    CORONAL,
    LABEL_LUT,
    LABELS_JSON,
    MANUAL_TRACK_LABEL,
    SAGITTAL,
    TOOL_POINTER,
)
from interaction import InteractionMixin
from recon_lab import ReconLabMixin
from ui_builder import UiBuilderMixin


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
        self.volume_mask = None           # AI 多器官标签图，shape=(Z,H,W) uint8：0=背景,1-24=器官,255=手动追踪
        self.organ_names = self._load_organ_labels()  # {类别号: (中文名, 英文名)}，用于图例
        self._organ_stats = []            # 最近一次器官定量结果，供面板显示与 CSV 导出
        self._hidden_organs = set()       # 被用户在图例中点隐的器官类别，渲染时跳过
        self._mask_undo = []              # 分割编辑撤销栈：[(切片号, 编辑前蒙版切片)]，上限 20
        self.is_english = False           # 界面语言，False=中文，True=英文
        self.anonymize = False            # 脱敏模式：显示层隐去患者身份信息（不改底层 DICOM）
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

        # --- AI 引擎状态 ---
        self.ai_thread = None             # AutoAIEngineThread 实例
        # _ai_generation：每次加载新数据自增，回调中比对该值可丢弃旧数据的结果（竞态保护）
        self._ai_generation = 0
        self._ai_state = 'standby'        # 'standby' | 'running' | 'done'
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
        映射已确证=TotalSegmentator class_map_part_organs（见该 JSON 的 _meta 与 experiments/）。"""
        fallback = {5: ("肝", "Liver"),
                    10: ("左肺上叶", "Lung UL (L)"), 11: ("左肺下叶", "Lung LL (L)"),
                    12: ("右肺上叶", "Lung UL (R)"), 13: ("右肺中叶", "Lung ML (R)"),
                    14: ("右肺下叶", "Lung LL (R)"),
                    MANUAL_TRACK_LABEL: ("手动追踪", "Manual")}
        try:
            with open(LABELS_JSON, encoding='utf-8') as f:
                data = json.load(f)
            names = {int(k): (v.get("name_zh", f"类{k}"), v.get("name_en", f"cls{k}"))
                     for k, v in data.get("labels", {}).items()}
            names[MANUAL_TRACK_LABEL] = ("手动追踪", "Manual")
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
            (self.btn_art, "ART / SIRT Iterative", "ART / SIRT 迭代重建"),
            (self.lbl_brush, "Brush R:", "画笔半径:"),
            (self.lbl_paint_target, "Paint as:", "画笔目标:"),
            (self.btn_export_stats, "Export Stats CSV", "导出定量 CSV"),
            (self.lbl_disclaimer,
             "⚠ AI results & organ labels are auto-inferred — for reference only, not for diagnosis.",
             "⚠ AI 结果与器官标签为自动推断，仅供参考，非诊断依据。"),
            (self.btn_clear_anno, "Clear Mask", "清空蒙版与标注"),
            (self.btn_reset, "Reset Workspace", "重置工作区"),
            (self.lbl_ww_hint, "Right-drag on image to adjust WW/WL", "在图像上右键拖拽可快速调节窗宽/窗位"),
            (self.chk_overlay, "Overlay", "信息叠加"),
            (self.chk_invert, "Invert", "反色"),
            (self.chk_anon, "De-ID", "脱敏"),
            (self.chk_global_scope, "New anno → all slices", "新标注穿透所有切片"),
        ):
            w.setText(en if e else cn)

        # 分组框标题表：(分组框, 英文, 中文) —— setTitle 类
        for g, en, cn in (
            (self.grp_proj, "Projection Generation", "X射线投影生成"),
            (self.grp_algo, "Reconstruction Algorithms", "图像重建算法"),
            (self.grp_matrix, "Matrix Recon & ART / SIRT", "直接矩阵重建 & ART / SIRT"),
            (self.grp_mon, "Performance Monitor", "算法性能监控"),
            (self.grp_patient, "PATIENT INFO", "患者信息"),
            (self.grp_display, "DISPLAY CONTROL", "显示控制"),
            (self.grp_measure, "MEASURE & CLEAN", "测量与清理"),
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
        for vdata in self.views.values():
            self._retranslate_combo(vdata['cb_plane'], plane_en, plane_cn, e, idx=vdata['plane'])
            self._retranslate_combo(vdata['preset'], v_en, v_cn, e)
            vdata['chk_anno'].setText("Anno" if e else "显示")
            vdata['lock'].setText("Lock" if e else "锁定")

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
        self._hidden_organs.clear()
        self._mask_undo = []         # 重置清撤销栈，避免撤销回被清掉的编辑
        self.lbl_hud.setText("")     # 清除光标 HUD 残留文本
        self._update_organ_stats()  # 蒙版已清，定量面板同步清空
        self.btn_mpr.setChecked(False)
        self.current_sinogram = None; self.current_theta = None
        self._last_recon_img = None
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]: b.setEnabled(False)
        # DMR/ART 只要有 DICOM 数据就可以运行（不依赖弦图）
        has_data = self.volume_hu is not None
        for b in [self.btn_dmr, self.btn_art]: b.setEnabled(has_data)
        for v in self.views.values():
            v['cb_plane'].setCurrentIndex(AXIAL)
            v['preset'].setCurrentIndex(0); v['lock'].setChecked(False)
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
        if not self._read_dicom_dir(path):
            return
        pid = self._build_volume_hu()
        if pid is None:
            self.dicom_datasets, self.volume_hu = prev_datasets, prev_volume
            QMessageBox.warning(self, "Load Failed" if self.is_english else "加载失败",
                                "No decodable image slices in this series."
                                if self.is_english else "该序列没有可解码的图像切片。")
            return
        self._load_annotations_json(pid)
        mask_restored = self._load_saved_mask(pid)  # 在首次显示前恢复上次的分割

        z = self.volume_hu.shape[0]
        self.on_slice_changed(z // 2)
        for b in [self.btn_dmr, self.btn_art]:
            b.setEnabled(True)
        for vd in self.views.values():
            vd['view']._user_zoomed = False   # 新病例回到适配状态
        # 延迟 100ms 做 fitInView，确保 Qt 已完成首次绘制布局再计算缩放
        QTimer.singleShot(100, lambda: [
            vd['view'].fitInView(vd['view'].scene.sceneRect(), Qt.KeepAspectRatio)
            for vd in self.views.values() if not vd['container'].isHidden()
        ])
        if mask_restored:
            # 已从磁盘恢复分割，跳过 ~100s 的 AI 重算
            self._ai_state = 'done'
            self._ai_time_ms = 0.0
            self.lbl_ai_status.setStyleSheet("color: #00FF00; font-weight: bold;")
            self.lbl_ai_status.setText(self._ai_done_text())
            self._update_organ_stats()
        else:
            self._kickoff_ai()

    def _read_dicom_dir(self, path):
        """递归扫描目录并并行读取所有 DICOM 文件，按 Z 物理位置排序。

        并行策略：用线程池 dcmread 各文件——pydicom 内部 IO + 大量 numpy 解码会释放 GIL，
        线程池在 SSD 上对千张切片可获 4–8× 加速。读盘失败的单个文件静默跳过，
        最终顺序与单线程版本严格一致（统一在所有线程完成后按 Z 物理位置排序）。

        DICOM 排序策略：
          优先使用 ImagePositionPatient[2]（床位 Z 坐标，单位 mm，物理精确）；
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

        # 过滤掉读取失败及不含像素数据的文件（DICOMDIR / RTSTRUCT 等）
        datasets = [ds for ds in results if ds is not None and 'PixelData' in ds]
        if not datasets:
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
            shape_groups[(int(getattr(ds, 'Rows', 0)), int(getattr(ds, 'Columns', 0)))].append(ds)
        if len(shape_groups) > 1:
            datasets = max(shape_groups.values(), key=len)
            r0, c0 = getattr(datasets[0], 'Rows', '?'), getattr(datasets[0], 'Columns', '?')
            print(f"检测到 {len(shape_groups)} 种切片尺寸，选用数量最多的（{len(datasets)} 张 {r0}×{c0}）")
        self.dicom_datasets = datasets

        # 排序键必须在整个序列内保持一致：z 物理坐标(mm)与 InstanceNumber(序号)是不同量纲，
        # 逐切片回退（部分切片缺 ImagePositionPatient）会把缺位置信息的层按序号插进有位置
        # 信息的层之间，打乱解剖顺序。因此先做序列级判定——所有切片都含 z 坐标才按 z 排序，
        # 否则整列统一回退 InstanceNumber（序列内单调的采集序号）。
        def _has_ipp(ds):
            try:
                float(ds.ImagePositionPatient[2])
                return True
            except Exception:
                return False

        if all(_has_ipp(ds) for ds in self.dicom_datasets):
            self.dicom_datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
        else:
            self.dicom_datasets.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
        return True

    def _build_volume_hu(self):
        """从 dicom_datasets 构建 3D HU 数组，初始化蒙版、3D 光标、切片滑动条。
        成功返回 PatientID；无任何可解码切片时返回 None（由 load_data 提示并中止）。

        HU 值转换公式（DICOM 标准）：
          HU = pixel_value × RescaleSlope + RescaleIntercept
          典型值：Slope=1, Intercept=-1024（GE 扫描仪常见），使得空气≈-1000 HU
        """
        ds = self.dicom_datasets[0]
        pid = str(getattr(ds, 'PatientID', 'N/A'))

        # 逐片解码并转 HU，防御式处理畸形数据（一张坏片不带崩整卷）：
        #   - pixel_array 解码失败（PixelData 截断 / 压缩语法缺编解码器 / group 0028 非法）→ 跳过该片；
        #   - 多帧文件（pixel_array 为 3D，NumberOfFrames>1）→ 展开为多个 2D 帧，共用该文件元数据。
        # dicom_datasets 随之同步为实际保留/展开后的切片，保证按 idx 取元数据与体积层一一对应。
        frames, kept = [], []
        for d in self.dicom_datasets:
            try:
                arr = d.pixel_array
            except Exception as e:
                print(f"跳过无法解码的切片: {e}")
                continue
            hu = (arr.astype(np.float32) * self._dcm_float(d, 'RescaleSlope', 1.0)
                  + self._dcm_float(d, 'RescaleIntercept', 0.0))
            if hu.ndim == 2:
                frames.append(hu); kept.append(d)
            elif hu.ndim == 3:              # 多帧：逐帧展开为切片
                for fr in hu:
                    frames.append(fr); kept.append(d)
            # 其余维度（异常数据）忽略
        if not frames:
            return None   # 无任何可解码切片
        # 兜底：万一帧尺寸仍不齐（多帧与单帧混合的极端情形），保留数量最多的尺寸
        shape_count = {}
        for f in frames:
            shape_count[f.shape] = shape_count.get(f.shape, 0) + 1
        dom = max(shape_count, key=shape_count.get)
        pairs = [(f, k) for f, k in zip(frames, kept, strict=False) if f.shape == dom]
        self.dicom_datasets = [k for _, k in pairs]
        self._refresh_patient_info()   # 按脱敏状态填患者面板
        self.volume_hu = np.array([f for f, _ in pairs])
        self.volume_mask = np.zeros_like(self.volume_hu, dtype=np.uint8)
        self.global_annotations = {'all': []}
        self._hidden_organs.clear()   # 换病例时清除上一例的图例隐藏状态
        self._mask_undo = []          # 换病例清撤销栈，防止旧切片号越界访问新蒙版
        z, y, x = self.volume_hu.shape
        # 默认将 3D 光标定位在体积中心（中间切片、中间行、中间列）
        self.current_3d_pos = [z // 2, y // 2, x // 2]
        self.slider_slice.setRange(0, z - 1)
        self.slider_slice.setValue(z // 2)
        return pid

    def _kickoff_ai(self):
        """启动后台 AI 推理。
        每次加载新数据时自增 generation 计数器；旧 AI 线程回调时若 generation 不匹配则静默
        丢弃结果，防止旧数据覆盖新数据的蒙版（竞态条件保护）。
        并作废上一个仍在运行的推理线程，避免多个 ~8.8GB 推理并发叠加导致内存翻倍/OOM。
        """
        if self.ai_thread is not None and self.ai_thread.isRunning():
            self.ai_thread.cancel()
        self._ai_generation += 1
        gen = self._ai_generation
        self._ai_state = 'running'
        self.lbl_ai_status.setStyleSheet("color: #F1C40F; font-weight: bold;")
        self.lbl_ai_status.setText("Processing AI Pipeline..." if self.is_english else "状态: AI 引擎自动运算中...")
        # lambda 中用 g=gen 捕获当前 generation 值（闭包变量，防止后续自增影响比对）
        self.ai_thread = AutoAIEngineThread(
            self.volume_hu,
            callback=lambda mask, t, g=gen: self.on_auto_ai_finished(mask, t, g),
            progress_callback=lambda d, t, g=gen: self._on_ai_progress(d, t, g)
        )
        self.ai_thread.start()

    def _on_ai_progress(self, done, total, generation):
        """AI 滑窗推理进度回调（经 Qt 信号 QueuedConnection 投递到主线程）。仅更新当前代的进度显示。"""
        if generation != self._ai_generation or self._ai_state != 'running':
            return
        pct = int(100 * done / total) if total else 0
        self.lbl_ai_status.setText(
            f"AI Segmenting... {pct}%" if self.is_english else f"状态: AI 分割中... {pct}%")

    def _ai_done_text(self):
        """AI 完成后的状态文案：显示检出的器官类别数（不含背景与手动追踪）。"""
        n = 0
        if self.volume_mask is not None:
            ids = np.unique(self.volume_mask)
            n = int(((ids != 0) & (ids != MANUAL_TRACK_LABEL)).sum())
        return (f"Ready: {n} organs ({self._ai_time_ms:.0f}ms)" if self.is_english
                else f"状态: 检出 {n} 个器官 ({self._ai_time_ms:.0f}ms)")

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
        self.volume_mask = final_mask
        self.lbl_ai_status.setStyleSheet("color: #00FF00; font-weight: bold;")
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
        px_sp = self._dcm_float(ds, 'PixelSpacing', 1.0, idx=0)
        # SliceThickness 用于冠/矢状面像素宽高比计算；若缺失/为空则估算为 px_sp×3（典型螺旋 CT 值）
        slice_thick = self._dcm_float(ds, 'SliceThickness', px_sp * 3)

        for vdata in self.views.values():
            if vdata['container'].isHidden():
                continue
            self._render_clinical_plane(vdata, z, y, x, ww_m, wl_m, px_sp, slice_thick)

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
        """安全读取 DICOM 数值标签为 float：标签缺失、为空(None)、或无法转 float 时返回 default。
        idx 非空时取序列第 idx 个元素（如 PixelSpacing[0]）。
        动机：getattr 的默认值只在属性【缺失】时生效；畸形 DICOM 常把数值标签留空
        （pydicom 读回 None），此时 float(None) / None[idx] 会抛 TypeError，导致
        加载/显示/定量全线崩溃。此处统一兜底。"""
        v = getattr(ds, tag, None)
        if v is None:
            return default
        try:
            if idx is not None:
                v = v[idx]
            return float(v)
        except (TypeError, ValueError, IndexError):
            return default

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
            return "ANON"
        name = str(getattr(self.dicom_datasets[0], 'PatientName', 'P')).replace('^', '_')
        return self._safe_name(name, fallback="P")

    def _toggle_overlay(self, on):
        """切换所有视图的 DICOM 信息叠加显隐。"""
        for vd in self.views.values():
            vd['view'].show_overlay = on
            vd['view'].viewport().update()

    def _apply_dicom_overlay(self, vdata, plane, z, y, x, ww, wl, px_sp, slice_thick):
        """构建并下发 PACS 风格的四角信息叠加与解剖方位字母。"""
        e = self.is_english
        ds0 = self.dicom_datasets[0]
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape
        idx, tot = {AXIAL: (z, Z_MAX), CORONAL: (y, Y_MAX), SAGITTAL: (x, X_MAX)}[plane]
        pname = ({AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"} if e else
                 {AXIAL: "横断面", CORONAL: "冠状面", SAGITTAL: "矢状面"})[plane]
        zoom = vdata['view'].transform().m11() * 100
        pid, pt_name, age = self._patient_display()   # 脱敏时隐去真实身份
        tl = [f"ID: {pid}", pt_name] + ([f"Age: {age}"] if age else [])
        corners = {
            'tl': tl,
            'tr': [f"{getattr(ds0, 'Modality', 'CT')}  ·  V{vdata['view'].view_id}", pname],
            'bl': [f"W: {int(ww)}  L: {int(wl)}", f"Zoom: {zoom:.0f}%"],
            'br': [f"{'Slice' if e else '层'} {idx + 1}/{tot}",
                   f"Thk {slice_thick:.1f}mm", f"Px {px_sp:.2f}mm"],
        }
        # 解剖方位字母：Axial 图像左=解剖右(R)；冠/矢状面上=头(S)下=足(I)
        orient = {AXIAL:    {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'},
                  CORONAL:  {'top': 'S', 'bottom': 'I', 'left': 'R', 'right': 'L'},
                  SAGITTAL: {'top': 'S', 'bottom': 'I', 'left': 'A', 'right': 'P'}}[plane]
        vdata['view'].set_overlay(corners, orient)

    def _render_clinical_plane(self, vdata, z, y, x, ww_m, wl_m, px_sp, slice_thick):
        """临床阅片分支：渲染单个视图的 2D 截面 + 蒙版 + 标注 + 十字线。"""
        plane = vdata['plane']
        pre = vdata['preset'].currentText()

        # 窗宽/窗位来源：优先使用各视图独立预设，否则跟随全局滑动条
        if pre in ["Global", "跟随"]:
            ww, wl = ww_m, wl_m
        else:
            ww, wl = self._WW_PRESETS.get(pre, ww_m), self._WL_PRESETS.get(pre, wl_m)

        # 根据平面切取对应的 2D 截面
        # 像素间距 sp=(行间距, 列间距)=(垂直/Y轴, 水平/X轴)，供 ruler 测距按真实 mm 换算
        # （graphics_view 里 d=√((dx·sp[1])²+(dy·sp[0])²)，故 sp[0] 配垂直、sp[1] 配水平）
        if plane == AXIAL:
            hu = self.volume_hu[z, :, :]
            sp = (px_sp, px_sp)              # 横断面：行/列均为 PixelSpacing
        elif plane == CORONAL:
            hu = self.volume_hu[:, y, :]     # (Z, X)：垂直=Z→SliceThickness，水平=X→PixelSpacing
            sp = (slice_thick, px_sp)
        elif plane == SAGITTAL:
            hu = self.volume_hu[:, :, x]     # (Z, Y)：垂直=Z→SliceThickness，水平=Y→PixelSpacing
            sp = (slice_thick, px_sp)

        # 窗宽窗位映射：HU → [0, 255] 线性映射
        img = np.clip(hu, wl - ww / 2, wl + ww / 2)
        img = ((img - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        if self.chk_invert.isChecked():
            img = 255 - img  # 反色（黑白反转），观察骨/软组织边界常用
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()

        # AI 多器官蒙版叠加：仅 Axial 平面支持。用调色板 LUT 一步向量化上色，
        # 每个类别号映射到 constants.LABEL_LUT 中的 RGBA（0=背景为全透明）。
        mq = None
        if plane == AXIAL and vdata['chk_anno'].isChecked() and self.volume_mask is not None:
            sm = self.volume_mask[z]
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
        self._apply_dicom_overlay(vdata, plane, z, y, x, ww, wl, px_sp, slice_thick)
        vdata['view'].clear_annotations()  # 清除上一帧的标注图元，防止重影

        if plane == AXIAL and vdata['chk_anno'].isChecked():
            self._render_annotations(vdata, z, sp)

        # MPR 十字准线：联动开启时各平面投影不同的坐标轴对
        if self.btn_mpr.isChecked():
            cx, cy = mpr_geometry.voxel_to_crosshair(plane, z, y, x)
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
