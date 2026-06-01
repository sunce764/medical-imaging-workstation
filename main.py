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

import sys
import os
import math
import csv
import json
import time
import pydicom           # 读取 DICOM 医学影像文件格式
import numpy as np
import scipy.ndimage as ndimage          # 用于 3D 连通域标记 (label)、缩放 (zoom)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QFileDialog, QSlider,
                               QLabel, QGroupBox, QFormLayout, QSplitter, QCheckBox,
                               QComboBox, QFrame, QGridLayout,
                               QGraphicsLineItem, QGraphicsTextItem, QGraphicsPathItem,
                               QButtonGroup, QMessageBox, QProgressDialog, QTabWidget, QRadioButton, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QLineF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPolygonF, QPainterPath

# 子模块导入
from ai_engine import AutoAIEngineThread
from graphics_view import MedicalGraphicsView
from constants import TOOL_POINTER, AXIAL, CORONAL, SAGITTAL
import recon as recon_lib

# AutoAIEngineThread → 已移至 ai_engine.py
# =========================================================================
# 主窗口：医学影像工作站 + 重建实验室
# 负责：UI 构建、DICOM 加载、多平面重建显示、标注管理、重建算法调度
# =========================================================================
class MedicalViewer(QMainWindow):
    # 临床标准窗宽/窗位预设值（中英文键名均支持，兼容语言切换后的下拉选项）
    # 提为类级常量避免在每次 update_display 中重新构造（MPR 多窗模式下每帧 4 次浪费）
    _WW_PRESETS = {"Lung": 1500, "Medi": 400, "Bone": 1500, "Vasc": 600, "Abdo": 150, "Brain": 80,
                   "肺窗": 1500, "纵隔": 400, "骨窗": 1500, "血管": 600, "腹部": 150, "脑窗": 80}
    _WL_PRESETS = {"Lung": -500, "Medi": 40, "Bone": 400, "Vasc": 150, "Abdo": 30, "Brain": 40,
                   "肺窗": -500, "纵隔": 40, "骨窗": 400, "血管": 150, "腹部": 30, "脑窗": 40}

    def __init__(self):
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
        self.volume_mask = None           # AI 分割蒙版，shape=(Z, H, W)，uint8 (0/1)
        self.is_english = False           # 界面语言，False=中文，True=英文
        self.current_3d_pos = [0, 0, 0]  # [z, y, x]，MPR 联动的三维光标位置

        # --- 重建实验室状态 ---
        self.recon_mode_active = False    # 是否处于重建实验室 Tab，影响 update_display() 的行为分支
        self._pre_recon_layout = 0        # 进入重建实验室前的布局模式，退出时用于还原
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

        # 启动时自动加载同目录下的"肺癌"文件夹（开发调试用，生产环境可删除）
        dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "肺癌")
        if os.path.exists(dp):
            self.load_data(dp)

    def setup_stylesheet(self):
        """从 style.qss 加载暗色主题样式表；文件缺失时静默跳过，UI 仍可用 Qt 默认样式渲染。"""
        qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
        try:
            with open(qss_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except IOError as e:
            print(f"Warning: failed to load style.qss: {e}")

    def init_ui(self):
        """构建主窗口三栏布局：左工具栏 | 中视图栅格 | 右控制面板。"""
        mw = QWidget()
        self.setCentralWidget(mw)
        l = QHBoxLayout(mw); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(0)

        self._build_left_toolbar()
        self._build_view_grid()
        self._build_right_panel()

        l.addWidget(self.left_toolbar)
        l.addWidget(self.main_splitter, 1)
        l.addWidget(self.right_panel)

    def _build_left_toolbar(self):
        """左侧 70px 宽工具按钮列：指针/卡尺/画笔/矩形/套索/3D 追踪，互斥选中。"""
        self.left_toolbar = QFrame()
        self.left_toolbar.setObjectName("LeftToolbar")
        self.left_toolbar.setFixedWidth(70)
        ll = QVBoxLayout(self.left_toolbar); ll.setContentsMargins(5, 20, 5, 20); ll.setSpacing(15)
        self.tool_btn_group = QButtonGroup(self)
        self.tool_btns = {}
        tool_data = [(0, 'btn_ptr'), (1, 'btn_rul'), (2, 'btn_drw'),
                     (4, 'btn_rec'), (3, 'btn_las'), (5, 'btn_trk')]
        for tid, key in tool_data:
            b = QPushButton(); b.setProperty("class", "ToolBtn"); b.setCheckable(True); b.setChecked(tid == 0)
            self.tool_btn_group.addButton(b, tid); ll.addWidget(b); self.tool_btns[key] = b
        self.tool_btn_group.idClicked.connect(self.change_active_tool)
        ll.addStretch()

    def _build_view_grid(self):
        """中央 4 视图栅格：QSplitter 嵌套结构（main_splitter 含 top/bottom 两个横向 splitter）。"""
        self.main_splitter = QSplitter(Qt.Vertical)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        for vid in (1, 2, 3, 4):
            self.create_independent_view(vid, AXIAL)
        self.top_splitter.addWidget(self.views[1]['container'])
        self.top_splitter.addWidget(self.views[2]['container'])
        self.bottom_splitter.addWidget(self.views[3]['container'])
        self.bottom_splitter.addWidget(self.views[4]['container'])
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)

    def _build_right_panel(self):
        """右侧 320px 宽控制面板：语言切换 / 加载 / 保存 + 两个 Tab（临床阅片 / 重建实验室）。"""
        self.right_panel = QFrame()
        self.right_panel.setObjectName("RightPanel")
        self.right_panel.setFixedWidth(320)
        rl = QVBoxLayout(self.right_panel); rl.setContentsMargins(12, 12, 12, 12); rl.setSpacing(5)

        # 顶部：语言切换按钮（靠右）
        th = QHBoxLayout()
        self.btn_lang = QPushButton("EN"); self.btn_lang.setFixedWidth(40)
        self.btn_lang.setStyleSheet("font-size: 10px; color: #5C677D; border: 1px solid #373E4D;")
        self.btn_lang.clicked.connect(self.toggle_language)
        th.addStretch(); th.addWidget(self.btn_lang); rl.addLayout(th)

        self.btn_import = QPushButton("加载 DICOM 目录"); self.btn_import.setObjectName("PrimaryBtn")
        self.btn_import.clicked.connect(self.select_folder); rl.addWidget(self.btn_import)
        self.btn_save_proj = QPushButton("保存标注工程"); self.btn_save_proj.setProperty("class", "ActionBtn")
        self.btn_save_proj.clicked.connect(self.save_project); rl.addWidget(self.btn_save_proj)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tab_clinical = QWidget()
        self.tab_recon = QWidget()
        self.tabs.addTab(self.tab_clinical, "临床阅片")
        self.tabs.addTab(self.tab_recon, "重建实验室")
        self._build_clinical_tab()
        self._build_recon_tab()
        rl.addWidget(self.tabs)

    def _build_clinical_tab(self):
        """临床阅片 Tab：患者信息 / 显示控制（布局+MPR+三滑条+预设）/ AI 状态 / 测量与清理。"""
        t1_lay = QVBoxLayout(self.tab_clinical)
        t1_lay.setContentsMargins(0, 0, 0, 0)

        # 患者信息分组
        self.grp_patient = QGroupBox("患者信息")
        info_lay = QFormLayout(); info_lay.setContentsMargins(10, 15, 10, 10)
        self.info_labels = {"ID": QLabel("N/A"), "NAME": QLabel("N/A"), "AGE": QLabel("N/A")}
        for k, v in self.info_labels.items():
            v.setObjectName("ValueText"); info_lay.addRow(QLabel(k), v)
        self.grp_patient.setLayout(info_lay)
        t1_lay.addWidget(self.grp_patient)

        # 显示控制分组（布局下拉、MPR 按钮、三滑条、预设窗口栅格）
        self.grp_display = QGroupBox("显示控制")
        dl = QVBoxLayout(); dl.setContentsMargins(10, 15, 10, 10)
        top_dl = QHBoxLayout()
        self.combo_layout = QComboBox()
        self.combo_layout.currentIndexChanged.connect(self.switch_layout)
        self.combo_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_dl.addWidget(self.combo_layout)
        self.btn_mpr = QPushButton("MPR 联动: 关"); self.btn_mpr.setObjectName("MprBtn"); self.btn_mpr.setCheckable(True)
        self.btn_mpr.clicked.connect(self.on_mpr_toggled)
        self.btn_mpr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_dl.addWidget(self.btn_mpr)
        top_dl.setStretch(0, 3); top_dl.setStretch(1, 2)
        dl.addLayout(top_dl)

        self.lbl_slice = QLabel(); self.slider_slice = QSlider(Qt.Horizontal)
        self.slider_slice.valueChanged.connect(self.on_slice_changed)
        self.lbl_ww = QLabel(); self.slider_ww = QSlider(Qt.Horizontal)
        self.slider_ww.setRange(1, 4000); self.slider_ww.setValue(1500); self.slider_ww.valueChanged.connect(self.update_display)
        self.lbl_wl = QLabel(); self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(-1200, 1200); self.slider_wl.setValue(-500); self.slider_wl.valueChanged.connect(self.update_display)
        for lbl, slider in [(self.lbl_slice, self.slider_slice), (self.lbl_ww, self.slider_ww), (self.lbl_wl, self.slider_wl)]:
            lbl.setFixedWidth(76); row = QHBoxLayout(); row.setSpacing(6); row.addWidget(lbl); row.addWidget(slider); dl.addLayout(row)
        self.lbl_ww_hint = QLabel(); self.lbl_ww_hint.setStyleSheet("color: #5C677D; font-size: 10px;")
        dl.addWidget(self.lbl_ww_hint)

        # 6 个临床预设窗口按钮（3 列栅格）
        pl = QGridLayout(); self.preset_btns = []
        for i, (n, ww, wl) in enumerate([("Lung", 1500, -500), ("Medi", 400, 40), ("Bone", 1500, 400),
                                          ("Vasc", 600, 150), ("Abdo", 150, 30), ("Brain", 80, 40)]):
            b = QPushButton(n); b.setProperty("class", "ActionBtn"); b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda c, w=ww, l=wl: self.set_window(w, l))
            pl.addWidget(b, i // 3, i % 3); self.preset_btns.append(b)
        pl.setColumnStretch(0, 1); pl.setColumnStretch(1, 1); pl.setColumnStretch(2, 1)
        dl.addLayout(pl)
        self.grp_display.setLayout(dl)
        t1_lay.addWidget(self.grp_display)

        # AI 状态分组（原 AI 按钮改为状态显示，因为已全自动）
        self.grp_ai = QGroupBox("自动化 AI 引擎")
        ai_lay = QVBoxLayout(); ai_lay.setContentsMargins(10, 15, 10, 10)
        self.lbl_ai_status = QLabel("状态: 待机中")
        self.lbl_ai_status.setStyleSheet("color: #8B949E; font-weight: bold;")
        ai_lay.addWidget(self.lbl_ai_status)
        self.grp_ai.setLayout(ai_lay)
        t1_lay.addWidget(self.grp_ai)

        # 测量与清理分组
        self.grp_measure = QGroupBox("测量与清理")
        ml = QVBoxLayout(); ml.setContentsMargins(10, 15, 10, 10)
        self.lbl_hu_value = QLabel()
        self.lbl_hu_value.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 13px; min-height: 18px; max-height: 18px;")
        self.lbl_hu_value.setAlignment(Qt.AlignCenter); ml.addWidget(self.lbl_hu_value)
        self.chk_global_scope = QCheckBox("新标注穿透所有切片"); ml.addWidget(self.chk_global_scope)
        self.btn_clear_anno = QPushButton("清空蒙版与标注"); self.btn_clear_anno.setProperty("class", "ActionBtn")
        self.btn_clear_anno.clicked.connect(self.clear_current_slice_annotations); ml.addWidget(self.btn_clear_anno)
        self.btn_reset = QPushButton("重置工作区"); self.btn_reset.setObjectName("DangerBtn")
        self.btn_reset.clicked.connect(self.reset_all_states); ml.addWidget(self.btn_reset)
        self.grp_measure.setLayout(ml)
        t1_lay.addWidget(self.grp_measure)
        t1_lay.addStretch()

    def _build_recon_tab(self):
        """重建实验室 Tab：投影生成 / BP-FBP-DFR / DMR-ART-SIRT / 性能监控。"""
        t2_lay = QVBoxLayout(self.tab_recon)
        t2_lay.setContentsMargins(0, 0, 0, 0)

        # 投影生成分组：角度单选 + 生成按钮
        self.grp_proj = QGroupBox("X射线投影生成")
        play = QVBoxLayout(); play.setSpacing(10)
        self.rad_60 = QRadioButton("60°"); self.rad_120 = QRadioButton("120°")
        self.rad_180 = QRadioButton("180°"); self.rad_360 = QRadioButton("360°")
        self.rad_180.setChecked(True)
        h_rad = QHBoxLayout()
        for r in [self.rad_60, self.rad_120, self.rad_180, self.rad_360]:
            r.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); h_rad.addWidget(r)
        h_rad.setStretch(0, 1); h_rad.setStretch(1, 1); h_rad.setStretch(2, 1); h_rad.setStretch(3, 1)
        play.addLayout(h_rad)
        self.btn_gen_sino = QPushButton("发射射线生成弦图"); self.btn_gen_sino.setProperty("class", "ActionBtn")
        self.btn_gen_sino.setStyleSheet("background-color: #D35400; color: white;")
        self.btn_gen_sino.clicked.connect(self.generate_sinogram)
        play.addWidget(self.btn_gen_sino)
        self.grp_proj.setLayout(play)
        t2_lay.addWidget(self.grp_proj)

        # 图像重建算法分组：DFR / BP / 滤波器选择 / FBP
        self.grp_algo = QGroupBox("图像重建算法")
        alay = QVBoxLayout(); alay.setSpacing(10)
        self.btn_dfr = QPushButton("直接傅里叶重建 (DFR)"); self.btn_dfr.setProperty("class", "ActionBtn")
        self.btn_dfr.clicked.connect(self.run_dfr)
        self.btn_bp = QPushButton("反投影法 (BP - 未滤波)"); self.btn_bp.setProperty("class", "ActionBtn")
        self.btn_bp.clicked.connect(self.run_bp)

        h_fbp = QHBoxLayout()
        self.lbl_filter_text = QLabel("选择滤波器:"); self.lbl_filter_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_fbp.addWidget(self.lbl_filter_text)
        self.cb_filter = QComboBox()
        self.cb_filter.addItems(["Ram-Lak", "Shepp-Logan", "Cosine", "Hamming", "Hann"])
        self.cb_filter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_fbp.addWidget(self.cb_filter)
        h_fbp.setStretch(0, 1); h_fbp.setStretch(1, 2)
        alay.addLayout(h_fbp)

        self.btn_fbp = QPushButton("滤波反投影 (FBP) 对比"); self.btn_fbp.setProperty("class", "ActionBtn")
        self.btn_fbp.setStyleSheet("background-color: #27AE60; color: white;")
        self.btn_fbp.clicked.connect(self.run_fbp)

        alay.addWidget(self.btn_dfr); alay.addWidget(self.btn_bp); alay.addWidget(self.btn_fbp)
        self.grp_algo.setLayout(alay)
        t2_lay.addWidget(self.grp_algo)
        # DFR/BP/FBP 三个重建按钮在生成弦图前保持禁用，强制工作流顺序：先投影再重建
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]:
            b.setEnabled(False)

        # 矩阵重建分组：尺寸 / 方法 / 迭代次数 / DMR / ART 按钮
        self.grp_matrix = QGroupBox("直接矩阵重建 & ART / SIRT")
        mxlay = QVBoxLayout(); mxlay.setSpacing(8)
        h_ms = QHBoxLayout()
        self.lbl_matrix_size = QLabel("图像尺寸:"); self.lbl_matrix_size.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_matrix_size = QComboBox(); self.cb_matrix_size.addItems(["16×16", "32×32", "64×64"])
        self.cb_matrix_size.setCurrentIndex(1); self.cb_matrix_size.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_ms.addWidget(self.lbl_matrix_size); h_ms.addWidget(self.cb_matrix_size)
        h_ms.setStretch(0, 1); h_ms.setStretch(1, 2); mxlay.addLayout(h_ms)
        h_mm = QHBoxLayout()
        self.lbl_art_method = QLabel("迭代方法:"); self.lbl_art_method.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_art_method = QComboBox(); self.cb_art_method.addItems(["ART", "SIRT"])
        self.cb_art_method.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_mm.addWidget(self.lbl_art_method); h_mm.addWidget(self.cb_art_method)
        h_mm.setStretch(0, 1); h_mm.setStretch(1, 2); mxlay.addLayout(h_mm)
        h_mi = QHBoxLayout()
        self.lbl_art_iter = QLabel("迭代次数:"); self.lbl_art_iter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cb_art_iter = QComboBox(); self.cb_art_iter.addItems(["10", "20", "50"])
        self.cb_art_iter.setCurrentIndex(1); self.cb_art_iter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_mi.addWidget(self.lbl_art_iter); h_mi.addWidget(self.cb_art_iter)
        h_mi.setStretch(0, 1); h_mi.setStretch(1, 2); mxlay.addLayout(h_mi)
        self.btn_dmr = QPushButton("直接矩阵重建 (DMR)"); self.btn_dmr.setProperty("class", "ActionBtn")
        self.btn_dmr.setStyleSheet("background-color: #1A5276; color: white;"); self.btn_dmr.clicked.connect(self.run_dmr)
        self.btn_art = QPushButton("ART / SIRT 迭代重建"); self.btn_art.setProperty("class", "ActionBtn")
        self.btn_art.setStyleSheet("background-color: #145A32; color: white;"); self.btn_art.clicked.connect(self.run_art_sirt)
        mxlay.addWidget(self.btn_dmr); mxlay.addWidget(self.btn_art)
        self.grp_matrix.setLayout(mxlay)
        t2_lay.addWidget(self.grp_matrix)
        # DMR/ART 不依赖弦图（自行生成小图并计算），但需要有 DICOM 数据才能运行
        for b in [self.btn_dmr, self.btn_art]:
            b.setEnabled(False)

        # 性能监控分组：耗时显示
        self.grp_mon = QGroupBox("算法性能监控")
        mlay = QVBoxLayout()
        self.lbl_time = QLabel("运行耗时: -- ms")
        self.lbl_time.setStyleSheet("color: #00FF00; font-family: monospace; font-size: 14px; font-weight: bold; background-color: #000000; padding: 6px; border-radius: 4px; border: 1px solid #333; min-height: 20px; max-height: 20px;")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        mlay.addWidget(self.lbl_time)
        self.grp_mon.setLayout(mlay)
        t2_lay.addWidget(self.grp_mon)
        t2_lay.addStretch()

    def toggle_language(self):
        """切换中英文界面，然后刷新所有控件文字。"""
        self.is_english = not self.is_english
        self.update_language()

    def update_language(self):
        e = self.is_english; self.btn_lang.setText("中" if e else "EN")
        self.tool_btns['btn_ptr'].setText("Pan\nProbe" if e else "探针\n拖拽")
        self.tool_btns['btn_rul'].setText("Ruler\nDist" if e else "测距\n卡尺")
        self.tool_btns['btn_drw'].setText("Draw\nPath" if e else "自由\n画笔")
        self.tool_btns['btn_rec'].setText("Rect\nCrop" if e else "矩形\n截取")
        self.tool_btns['btn_las'].setText("Lasso\nMask" if e else "套索\n抠图")
        self.tool_btns['btn_trk'].setText("3D\nTrack" if e else "3D\n追踪")
        _tips = {
            'btn_ptr': ("Pan & probe — drag to pan, click to measure HU, right-drag to adjust WW/WL",
                        "探针/拖拽 — 拖动平移 | 点击测量HU值 | 右键拖拽调节窗宽窗位"),
            'btn_rul': ("Ruler — drag to measure distance (mm)", "测距卡尺 — 拖出直线测量两点距离(mm)"),
            'btn_drw': ("Freehand draw — annotate freely", "自由画笔 — 在图像上自由绘制标注"),
            'btn_rec': ("Rect crop — select ROI to export stats", "矩形截取 — 框选区域导出ROI统计"),
            'btn_las': ("Lasso mask — polygon segmentation", "套索抠图 — 绘制多边形生成分割蒙版"),
            'btn_trk': ("3D track — track structure through slices", "3D追踪 — 框选区域执行三维连通域追踪"),
        }
        for key, (tip_en, tip_cn) in _tips.items():
            self.tool_btns[key].setToolTip(tip_en if e else tip_cn)
        self.btn_import.setText("Load DICOM Folder" if e else "加载 DICOM 目录")
        self.btn_save_proj.setText("Save Project" if e else "保存标注工程")
        
        self.tabs.setTabText(0, "Clinical Mode" if e else "临床阅片")
        self.tabs.setTabText(1, "Recon Lab" if e else "重建实验室")

        self.grp_proj.setTitle("Projection Generation" if e else "X射线投影生成")
        self.btn_gen_sino.setText("Generate Sinogram" if e else "发射射线生成弦图")
        self.grp_algo.setTitle("Reconstruction Algorithms" if e else "图像重建算法")
        self.btn_dfr.setText("Direct Fourier (DFR)" if e else "直接傅里叶重建 (DFR)")
        self.btn_bp.setText("Back Projection (BP)" if e else "反投影法 (BP - 未滤波)")
        self.lbl_filter_text.setText("Filter:" if e else "选择滤波器:")
        self.btn_fbp.setText("Filtered BP (FBP)" if e else "滤波反投影 (FBP) 对比")
        self.grp_matrix.setTitle("Matrix Recon & ART / SIRT" if e else "直接矩阵重建 & ART / SIRT")
        self.lbl_matrix_size.setText("Image Size:" if e else "图像尺寸:")
        self.lbl_art_method.setText("Method:" if e else "迭代方法:")
        self.lbl_art_iter.setText("Iterations:" if e else "迭代次数:")
        self.btn_dmr.setText("Direct Matrix Recon (DMR)" if e else "直接矩阵重建 (DMR)")
        self.btn_art.setText("ART / SIRT Iterative" if e else "ART / SIRT 迭代重建")
        self.grp_mon.setTitle("Performance Monitor" if e else "算法性能监控")
        
        if "耗时: --" in self.lbl_time.text() or "Time: --" in self.lbl_time.text():
            self.lbl_time.setText("Run Time: -- ms" if e else "运行耗时: -- ms")

        self.grp_patient.setTitle("PATIENT INFO" if e else "患者信息")
        self.grp_display.setTitle("DISPLAY CONTROL" if e else "显示控制")
        self.grp_measure.setTitle("MEASURE & CLEAN" if e else "测量与清理")
        self.grp_ai.setTitle("Automated AI Engine" if e else "自动化 AI 引擎")
        
        if self._ai_state == 'standby':
            self.lbl_ai_status.setText("Status: Standby" if e else "状态: 待机中")
        elif self._ai_state == 'running':
            self.lbl_ai_status.setText("Processing AI Pipeline..." if e else "状态: AI 引擎自动运算中...")
        elif self._ai_state == 'done':
            self.lbl_ai_status.setText(f"Ready ({self._ai_time_ms:.1f}ms)" if e else f"状态: 自动分割完成 ({self._ai_time_ms:.1f}ms)")
            
        mpr_on = self.btn_mpr.isChecked()
        self.btn_mpr.setText(("MPR Link: ON" if mpr_on else "MPR Link: OFF") if e else ("MPR 联动: 开启" if mpr_on else "MPR 联动: 关"))
        opts = ["1x1 Single", "1x2 Dual", "2x2 Grid"] if e else ["单窗模式 (1x1)", "双窗对比 (1x2)", "四窗矩阵 (2x2)"]
        ci = max(0, self.combo_layout.currentIndex()); self.combo_layout.blockSignals(True); self.combo_layout.clear(); self.combo_layout.addItems(opts); self.combo_layout.setCurrentIndex(ci); self.combo_layout.blockSignals(False)
        p_en, p_cn = ["Lung","Medi","Bone","Vasc","Abdo","Brain"], ["肺窗","纵隔","骨窗","血管","腹部","脑窗"]
        for b, n in zip(self.preset_btns, p_en if e else p_cn): b.setText(n)
        self.btn_clear_anno.setText("Clear Mask" if e else "清空蒙版与标注")
        self.btn_reset.setText("Reset Workspace" if e else "重置工作区")
        
        v_en = ["Global", "Lung", "Medi", "Bone", "Vasc", "Abdo", "Brain"]
        v_cn = ["跟随", "肺窗", "纵隔", "骨窗", "血管", "腹部", "脑窗"]
        plane_en = ["Axial", "Coronal", "Sagittal"]
        plane_cn = ["横断面", "冠状面", "矢状面"]
        
        for vdata in self.views.values():
            curr_p = max(0, vdata['plane'])
            vdata['cb_plane'].blockSignals(True); vdata['cb_plane'].clear(); vdata['cb_plane'].addItems(plane_en if e else plane_cn); vdata['cb_plane'].setCurrentIndex(curr_p); vdata['cb_plane'].blockSignals(False)
            curr_preset = max(0, vdata['preset'].currentIndex())
            vdata['preset'].blockSignals(True); vdata['preset'].clear(); vdata['preset'].addItems(v_en if e else v_cn); vdata['preset'].setCurrentIndex(curr_preset); vdata['preset'].blockSignals(False)
            vdata['chk_anno'].setText("Anno" if e else "显示")
            vdata['lock'].setText("Lock" if e else "锁定")
        self.lbl_ww_hint.setText("Right-drag on image to adjust WW/WL" if e else "在图像上右键拖拽可快速调节窗宽/窗位")
        self.on_slice_changed(self.slider_slice.value())

    def on_tab_changed(self, index):
        """Tab 切换回调：在临床阅片 (index=0) 和重建实验室 (index=1) 之间切换。

        反闪烁设计：setUpdatesEnabled(False) 屏蔽所有绘制事件，整个切换过程只产生最终
        一帧；try/finally 确保异常时 UI 也能恢复正常刷新。具体进入 / 退出逻辑分别
        委托给 _enter_recon_mode / _exit_recon_mode。
        """
        self.recon_mode_active = (index == 1)
        self.setUpdatesEnabled(False)
        try:
            if self.recon_mode_active:
                self._enter_recon_mode()
            else:
                self._exit_recon_mode()
        finally:
            self.setUpdatesEnabled(True)

    def _enter_recon_mode(self):
        """进入重建实验室：记忆原布局、清空视图、切到 2x2、隐藏每视图工具栏控件。"""
        self._pre_recon_layout = self.combo_layout.currentIndex()
        for vid in range(1, 5):
            v = self.views[vid]['view']
            v.image_item.setPixmap(QPixmap())
            v.mask_item.setPixmap(QPixmap())
            v.resetTransform()
        # 切到 2x2，setSizes 在 setUpdatesEnabled(False) 下同步生效
        self._apply_grid_visibility(2)
        self._apply_grid_sizes(2)
        self.set_view_title(1, "V1 [Ground Truth]" if self.is_english else "V1 [真实切片]")
        self._set_recon_pending_titles()
        for v in self.views.values():
            v['cb_plane'].hide(); v['preset'].hide(); v['chk_anno'].hide(); v['lock'].hide()
        self.update_display()

    def _exit_recon_mode(self):
        """退出重建实验室：清空弦图缓存与按钮、恢复每视图工具栏控件、还原原布局。"""
        self.current_sinogram = None
        self._cached_bp = None; self._cached_bp_sino = None
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]:
            b.setEnabled(False)
        for vid, v in self.views.items():
            v['view'].image_item.setPixmap(QPixmap())
            v['view'].mask_item.setPixmap(QPixmap())
            v['view'].resetTransform()
            v['view'].setRenderHint(QPainter.SmoothPixmapTransform, True)
            v['cb_plane'].show(); v['preset'].show(); v['chk_anno'].show(); v['lock'].show()
            self.set_view_title(vid, f"V{vid}")
        prev = self._pre_recon_layout
        self._apply_grid_visibility(prev)
        self._apply_grid_sizes(prev)
        self.update_display()

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

    def _set_recon_pending_titles(self):
        """将 V2/V3/V4 标题统一设置为"请先生成弦图"的等待提示。
        进入重建实验室、切换切片导致旧弦图失效时调用，集中处理避免散落多份字符串。
        """
        txt = "[— run projection —]" if self.is_english else "[— 请先生成弦图 —]"
        for vid in (2, 3, 4):
            self.set_view_title(vid, f"V{vid} {txt}")

    def _get_n_angles(self):
        """读取当前 UI 选中的投影角度数（60/120/180/360），180° 为默认值。"""
        if self.rad_60.isChecked():    return 60
        if self.rad_120.isChecked():   return 120
        if self.rad_360.isChecked():   return 360
        return 180

    def set_view_title(self, vid, title):
        """更新指定视图工具栏中的标题标签文字。
        直接使用 create_independent_view 中缓存的 label 引用，避免每次 findChild 遍历视图树。
        """
        try:
            self.views[vid]['title_label'].setText(title)
        except Exception as e:
            print(f"Warning: set_view_title V{vid}: {e}")

    def display_numpy_image(self, vid, img_array, is_freq=False):
        """将 NumPy 2D 数组归一化为灰度图并显示到指定视图。

        is_freq=True：频域模式，使用对数压缩（log1p）展示大动态范围的频谱；
                       绝对值取对数是频谱可视化的标准做法，防止高频分量因量级太小而不可见。
        is_freq=False：空间域模式，使用百分位数鲁棒归一化，
                        忽略 1%~99% 之外的极端值，避免孤立噪点将整体对比度压缩到极小范围。
        """
        if img_array is None:
            return
        h, w = img_array.shape

        if is_freq:
            # log1p(|F|)：对复数取模，再取对数（+1 防止 log(0)）
            img_norm = np.log1p(np.abs(img_array))
            ptp = img_norm.max() - img_norm.min()
            denom = ptp if ptp > 0 else 1.0
            img_norm = ((img_norm - img_norm.min()) / denom * 255).astype(np.uint8)
        else:
            # 百分位数截断：排除顶部 1% 和底部 1% 极端值
            # 好处：DMR/ART 重建图像可能存在边缘溢出，截断后主体对比度不受影响
            pmin = np.percentile(img_array, 1)
            pmax = np.percentile(img_array, 99)
            img_clipped = np.clip(img_array, pmin, pmax)
            denom = pmax - pmin if pmax > pmin else 1.0
            img_norm = ((img_clipped - pmin) / denom * 255).astype(np.uint8)
            # ascontiguousarray 确保内存布局为 C 连续，
            # 防止 Qt C++ 底层读取跨步数组时发生内存访问错误
            img_norm = np.ascontiguousarray(img_norm)

        # QImage 直接引用 img_norm.data 的内存（零拷贝），.copy() 使 Qt 持有独立副本，
        # 防止 NumPy 数组离开作用域后 Qt 访问已释放内存
        qimg = QImage(img_norm.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.views[vid]['view'].set_image(QPixmap.fromImage(qimg), pixel_spacing=(1.0, 1.0))
        self.views[vid]['view'].clear_annotations()

    def generate_sinogram(self):
        """对当前 Axial 切片执行 Radon 变换，生成弦图（Sinogram）。

        弦图的物理含义：
          X 射线从不同角度穿过人体，探测器在每个角度测量透射强度（即线积分）。
          弦图的横轴为角度（°），纵轴为探测器位置（像素），
          每一列是该角度下所有探测器的一次测量——即一个"投影"。
          将所有角度的投影并排排列，得到的 2D 图像就是弦图。

        角度选择影响重建质量：
          - 180°：覆盖完整，重建质量最高（临床 CT 标准）
          - 120°：欠采样，重建出现条状伪影
          - 60°：严重欠采样，重建质量很差（教学演示稀疏投影问题）

        归一化处理：radon 对线性值求积分，HU 值可能为负（空气=-1000），
        需先归一化到 [0,1] 保证弦图数值范围一致，便于后续显示和重建。

        生成后：
          - V2 显示弦图（.T 转置使角度在横轴）
          - V3/V4 清空并提示需要重建
          - 启用 DFR/BP/FBP 三个重建按钮
        """
        if not self.dicom_datasets or self.volume_hu is None:
            return
        self.current_theta = recon_lib.make_theta(self._get_n_angles())

        # 来源选择：有重建结果时对重建图做 Radon，用完清空（下次回到原图）
        if self._last_recon_img is not None:
            img_src = self._last_recon_img
            src_label = "重建图" if not self.is_english else "Recon"
            self._last_recon_img = None   # 消费后清空，下次按钮回到原图路径
        else:
            z = self.current_3d_pos[0]
            img_gt = self.volume_hu[z]
            denom = img_gt.max() - img_gt.min()
            img_src = (img_gt - img_gt.min()) / (denom if denom > 0 else 1.0)
            src_label = "原图" if not self.is_english else "Origin"

        start_t = time.perf_counter()
        self.current_sinogram = recon_lib.compute_sinogram(img_src, self.current_theta)
        elapsed = (time.perf_counter() - start_t) * 1000
        self.lbl_time.setText(f"Radon [{src_label}]: {elapsed:.1f} ms")
        self.display_numpy_image(2, self.current_sinogram.T)
        self.set_view_title(2, f"V2 [Sinogram - {src_label}]")
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]:
            b.setEnabled(True)
        # 生成新弦图后，V3/V4 的旧重建结果已作废，清空并更新提示标题
        self.views[3]['view'].image_item.setPixmap(QPixmap())
        self.views[4]['view'].image_item.setPixmap(QPixmap())
        self.set_view_title(3, "V3 [— run reconstruction —]" if self.is_english else "V3 [— 请选择算法重建 —]")
        self.set_view_title(4, "V4 [— run reconstruction —]" if self.is_english else "V4 [— 请选择算法重建 —]")


    def run_bp(self):
        """反投影法 (Back Projection, BP) 重建——不加任何滤波器的原始反投影。

        原理：将弦图中每个角度的投影值"抹回"到图像空间的对应路径上，
        所有角度的贡献叠加得到重建图像。
        缺陷：低频分量被过度叠加，导致重建图像边缘极度模糊（星形/放射状伪影）。
        对比目的：展示滤波（FBP）对图像质量的改善效果。
        """
        if self.current_sinogram is None:
            return
        self._fit_recon_views(smooth=True)
        start_t = time.perf_counter()
        recon_bp = recon_lib.compute_bp(self.current_sinogram, self.current_theta)
        elapsed = (time.perf_counter() - start_t) * 1000
        self.lbl_time.setText(f"BP Time: {elapsed:.1f} ms" if self.is_english else f"纯反投影(BP)耗时: {elapsed:.1f} ms")
        self.display_numpy_image(4, recon_bp)
        self.set_view_title(4, "V4 [BP Unfiltered]" if self.is_english else "V4 [反投影 BP - 边缘模糊]")

    def run_fbp(self):
        """滤波反投影法 (Filtered Back Projection, FBP)——CT 扫描仪最核心的重建算法。

        FBP = 先对每个投影做频域高通滤波（加强高频/边缘），再做反投影。
        常用滤波器：
          - Ram-Lak (Ramp)：理想高通，噪声放大最大但分辨率最高
          - Shepp-Logan：Ram-Lak 乘以 sinc 窗，减少振铃伪影
          - Cosine/Hamming/Hann：更强的低通特性，噪声小但分辨率略低

        注意：skimage 内部将 Ram-Lak 称为 'ramp'，UI 显示为 'Ram-Lak'，
        需要在此处手动映射，否则 skimage 会抛出 ValueError。

        同时显示 BP（V3）和 FBP（V4）方便直观对比滤波效果。
        BP 结果有缓存：同一弦图切换不同滤波器时无需重新计算 BP。
        """
        if self.current_sinogram is None:
            return
        self._fit_recon_views(smooth=True)
        filter_name = self.cb_filter.currentText().lower()
        # 用对象身份（is 比较）作缓存键：弦图对象替换时 self._cached_bp_sino 不再 is 新对象，自动失效
        # 改用 is 而非 id()：id() 在对象被 GC 后会回收复用，新对象可能巧合命中旧 id 造成错误缓存命中
        start_t = time.perf_counter()
        if self._cached_bp is None or self._cached_bp_sino is not self.current_sinogram:
            self._cached_bp = recon_lib.compute_bp(self.current_sinogram, self.current_theta)
            self._cached_bp_sino = self.current_sinogram
        recon_bp = self._cached_bp
        # compute_fbp 内部处理 'ram-lak' → 'ramp' 的名称映射
        _, recon_fbp = recon_lib.compute_fbp(self.current_sinogram, self.current_theta, filter_name)
        elapsed = (time.perf_counter() - start_t) * 1000
        self.lbl_time.setText(f"FBP ({filter_name}) Time: {elapsed:.1f} ms" if self.is_english else f"FBP ({filter_name})耗时: {elapsed:.1f} ms")
        self.display_numpy_image(3, recon_bp)
        self.set_view_title(3, "V3 [BP Comparison]" if self.is_english else "V3 [未滤波反投影对比]")
        self.display_numpy_image(4, recon_fbp)
        self.set_view_title(4, f"V4 [FBP - {filter_name}]" if self.is_english else f"V4 [滤波反投影 FBP - {filter_name}]")

    def run_dfr(self):
        """直接傅里叶重建法 (Direct Fourier Reconstruction, DFR)。

        理论基础——傅里叶中心切片定理 (Fourier Slice Theorem)：
          对投影数据在探测器方向做 1D FFT，得到的结果等于图像 2D FFT 在
          对应角度方向穿过原点的一条"切片"。
          因此，收集所有角度的 1D FFT，就等于在极坐标系中填充了 2D 频域，
          再做 2D 逆 FFT 即可还原图像——这正是 DFR 的核心思路。

        实现步骤：
          1. 对弦图每列做 1D FFT（沿探测器方向），得到极坐标频域样本
          2. 将极坐标 (r, θ) 样本插值到直角坐标网格（griddata）
          3. 对插值后的 2D 频域做 2D 逆 FFT，得到重建图像

        关键坑点：
          - FFT 前后必须做 fftshift/ifftshift，使频域零频在中心，
            否则极坐标映射的角度与 FFT 轴不对齐
          - 插值必须用 'linear' 或 'cubic'，'nearest' 会产生放射状锯齿伪影，
            因为极坐标在低频（r≈0）区域样本密集，高频区稀疏，
            最近邻在稀疏区产生大块相同值的伪影
        """
        if self.current_sinogram is None:
            return
        self._fit_recon_views(smooth=True)
        p = QProgressDialog("Computing 2D FFT & Gridding..." if self.is_english else "正在计算 2D 傅里叶极坐标插值...", None, 0, 0, self)
        p.setWindowModality(Qt.WindowModal); p.show(); QApplication.processEvents()

        start_t = time.perf_counter()
        freq_domain_2d, fft_1d_display, recon_dfr = recon_lib.compute_dfr(
            self.current_sinogram, self.current_theta
        )
        elapsed = (time.perf_counter() - start_t) * 1000
        p.close()

        self.lbl_time.setText(f"DFR Time: {elapsed:.1f} ms" if self.is_english else f"傅里叶重建(DFR)耗时: {elapsed:.1f} ms")
        # V2 临时征用：展示"二维频域分布图"
        self.display_numpy_image(2, freq_domain_2d, is_freq=True)
        self.set_view_title(2, "V2 [2D Freq Spectrum]" if self.is_english else "V2 [映射后的二维频域分布]")
        # V3 显示：投影的一维傅里叶谱
        self.display_numpy_image(3, fft_1d_display, is_freq=False)
        self.set_view_title(3, "V3 [1D FFT Spectrum]" if self.is_english else "V3 [投影的一维傅里叶谱]")
        # V4 显示：重建图像
        self.display_numpy_image(4, np.rot90(np.abs(recon_dfr)))
        self.set_view_title(4, "V4 [Direct Fourier DFR]" if self.is_english else "V4 [直接傅里叶重建 DFR]")

    def on_mpr_toggled(self, checked):
        self.update_language() 
        if checked:
            default_planes = [0, AXIAL, CORONAL, SAGITTAL, AXIAL]
            for vid, v in self.views.items(): v['cb_plane'].setCurrentIndex(default_planes[vid])
            self.update_display()
        else:
            for vdata in self.views.values(): vdata['view'].draw_crosshair(0, 0, show=False)

    def create_independent_view(self, vid, plane=AXIAL):
        c = QFrame(); c.setObjectName("ViewContainer"); lay = QVBoxLayout(c); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        t = QFrame(); t.setObjectName("ViewToolbar"); t.setFixedHeight(32) 
        tl = QHBoxLayout(t); tl.setContentsMargins(8,2,8,2); tl.setSpacing(6)
        lt = QLabel(f"V{vid}"); lt.setStyleSheet("color: #C9D1D9; font-weight: bold; min-width: 20px;")
        cb_plane = QComboBox(); cb_plane.setFixedWidth(80)
        ps = QComboBox(); ps.setFixedWidth(85); ps.currentIndexChanged.connect(self.update_display)
        an = QCheckBox(); an.setObjectName("ViewOption"); an.setChecked(True); an.stateChanged.connect(self.update_display)
        lk = QCheckBox(); lk.setObjectName("ViewOption"); lk.stateChanged.connect(self.update_display)
        tl.addWidget(lt); tl.addWidget(cb_plane); tl.addWidget(ps); tl.addStretch(); tl.addWidget(an); tl.addWidget(lk)
        v = MedicalGraphicsView(vid)
        v.clicked_pos.connect(lambda p, id=vid: self.measure_hu(p, id))
        v.wheel_scrolled.connect(lambda d, id=vid: self.on_wheel_mpr(d, id)) 
        v.annotation_added.connect(self.handle_annotation_added)
        v.annotation_deleted.connect(self.handle_annotation_deleted)
        v.crop_requested.connect(lambda pts, id=vid: self.handle_crop_requested(id, pts))
        v.track_requested.connect(lambda r, id=vid: self.handle_3d_track_requested(id, r))
        v.window_changed.connect(self.on_window_changed_by_mouse)
        v.mouse_hovered.connect(lambda pos, id=vid: self.sync_crosshair(pos, id))
        lay.addWidget(t); lay.addWidget(v); t.raise_()
        self.views[vid] = {'container':c, 'cb_plane': cb_plane, 'preset':ps, 'lock':lk, 'chk_anno':an, 'view':v, 'plane': plane, 'title_label': lt}
        cb_plane.currentIndexChanged.connect(lambda idx, v_id=vid: self.change_view_plane(v_id, idx))

    def change_view_plane(self, vid, plane_idx):
        """切换某个视图的成像平面（横断/冠状/矢状）。
        切换后延迟 20ms 做 fitInView，等待新图像渲染完成后再适配缩放，
        避免基于旧图像尺寸计算比例。
        """
        if plane_idx < 0:
            return  # 下拉框清空重填时会触发 index=-1，需要过滤
        self.views[vid]['plane'] = plane_idx
        if not self.recon_mode_active:
            self.update_display()
        v = self.views[vid]['view']
        QTimer.singleShot(20, lambda: v.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio))

    def sync_crosshair(self, scene_pos, vid):
        """MPR 联动：当用户在任意视图中移动鼠标时，同步更新所有视图的十字准线位置。

        坐标映射规则（三平面共用同一个 3D 光标 [z, y, x]）：
          - Axial 视图   → 鼠标 (px, py) 对应 3D 的 (x=px, y=py)，z 不变
          - Coronal 视图 → 鼠标 (px, py) 对应 3D 的 (x=px, z=py)，y 不变
          - Sagittal 视图 → 鼠标 (px, py) 对应 3D 的 (y=px, z=py)，x 不变
        """
        if self.volume_hu is None or self.recon_mode_active:
            return
        if not self.btn_mpr.isChecked():
            return
        source_plane = self.views[vid]['plane']
        z, y, x = self.current_3d_pos
        pos_x, pos_y = int(scene_pos.x()), int(scene_pos.y())
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape
        if source_plane == AXIAL:
            x, y = pos_x, pos_y
        elif source_plane == CORONAL:
            x, z = pos_x, pos_y
        elif source_plane == SAGITTAL:
            y, z = pos_x, pos_y
        # 限制在体积范围内，防止越界
        x = max(0, min(x, X_MAX - 1))
        y = max(0, min(y, Y_MAX - 1))
        z = max(0, min(z, Z_MAX - 1))
        self.current_3d_pos = [z, y, x]
        # 将十字线投影到每个可见平面的 2D 坐标
        for v_id, vdata in self.views.items():
            if vdata['container'].isHidden():
                continue
            p = vdata['plane']
            if p == AXIAL:
                vdata['view'].draw_crosshair(x, y)
            elif p == CORONAL:
                vdata['view'].draw_crosshair(x, z)
            elif p == SAGITTAL:
                vdata['view'].draw_crosshair(y, z)

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

    def handle_3d_track_requested(self, vid, rect):
        """3D 连通域追踪：在当前 Axial 切片上框选 ROI，提取该区域的 HU 统计特征，
        然后在整个 3D 体积中找出 HU 分布相似的连通域，生成 3D 分割蒙版。

        算法原理：
          1. 计算 ROI 的 HU 中位数和标准差（中位数比均值更抗离群值）
          2. 在全体积中找出 HU 在 [med-1.5σ, med+1.5σ] 范围内的体素（类似区域增长）
          3. 对该 HU 范围内的体素做 3D 连通域标记
          4. 选取在 ROI 框内体素最多的连通域标签，即为目标结构
        """
        if self.volume_hu is None or self.recon_mode_active:
            return
        if self.views[vid]['plane'] != AXIAL:
            QMessageBox.information(self, "提示", "目前智能追踪仅支持在 Axial 进行。")
            return
        idx = self.current_3d_pos[0]
        x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
        h, w = self.volume_hu.shape[1], self.volume_hu.shape[2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        p = QProgressDialog("Computing 3D..." if self.is_english else "正在计算 3D...", None, 0, 0, self)
        p.setWindowModality(Qt.WindowModal); p.show(); QApplication.processEvents()
        try:
            roi = self.volume_hu[idx, y1:y2, x1:x2]
            med, std = np.median(roi), np.std(roi)
            bv = (self.volume_hu >= med - 1.5 * std) & (self.volume_hu <= med + 1.5 * std)
            lab, _ = ndimage.label(bv)
            rl = lab[idx, y1:y2, x1:x2]
            rl = rl[rl > 0]  # 过滤背景标签 0
            if len(rl) > 0:
                # bincount 统计 ROI 区域内各标签出现次数，取最多的那个为目标
                self.volume_mask = (lab == np.bincount(rl.flatten()).argmax()).astype(np.uint8)
        except Exception:
            pass
        p.close()
        self.update_display()

    def handle_crop_requested(self, vid, pts):
        """截取工具：对多边形 ROI 区域统计 HU 值，可选保存裁剪图像和 CSV 报告。

        步骤：
          1. 用 QPainter 将多边形栅格化为白色掩码图（白=ROI内，黑=ROI外）
          2. 将掩码转换为 NumPy 数组，提取 ROI 内的 HU 值
          3. 计算面积（像素数 × 像素间距²）和平均 HU
          4. 弹框确认，用户选择是否保存裁剪图像和 CSV 记录
        """
        if self.recon_mode_active or self.views[vid]['plane'] != AXIAL:
            return
        idx = self.current_3d_pos[0]
        ds = self.dicom_datasets[idx]
        hu = self.volume_hu[idx]
        sp = (float(getattr(ds, 'PixelSpacing', [1, 1])[0]), float(getattr(ds, 'PixelSpacing', [1, 1])[1]))
        h, w = hu.shape
        # 用 QPainter 将多边形光栅化为掩码图像
        mq = QImage(w, h, QImage.Format_Grayscale8); mq.fill(Qt.black)
        painter = QPainter(mq)
        painter.setBrush(Qt.white)
        painter.drawPolygon(QPolygonF([QPointF(p[0], p[1]) for p in pts]))
        painter.end()
        # 将 QImage 转换为 NumPy 掩码，bytesPerLine 可能因对齐而大于 w，需要裁剪
        ma = np.array(mq.constBits(), dtype=np.uint8).reshape((h, mq.bytesPerLine()))[:, :w].copy()
        bm = (ma > 0).astype(np.uint8)
        rh = hu[bm == 1]
        if len(rh) > 0:
            area = len(rh) * sp[0] * sp[1]
            if QMessageBox.question(self, "Stats",
                                    f"Area: {area:.2f} mm2\nMean: {np.mean(rh):.1f} HU\nSave?") == QMessageBox.Yes:
                # 软组织窗归一化：-1250~250 HU 映射到 0~255（保存为 PNG）
                img = np.clip(hu, -1250, 250)
                img = ((img + 1250) / 1500 * 255).astype(np.uint8)
                fn = f"{str(getattr(ds, 'PatientName', 'P')).replace('^', '_')}_S{idx+1}_{datetime.now().strftime('%H%M%S')}.png"
                ed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions")
                os.makedirs(ed, exist_ok=True)
                s_p, _ = QFileDialog.getSaveFileName(self, "Save", os.path.join(ed, fn), "PNG (*.png)")
                if s_p:
                    # img*bm：将 ROI 外的像素清零，保留病灶区域
                    QImage((img * bm).data, w, h, w, QImage.Format_Grayscale8).copy().save(s_p)
                    try:
                        with open(os.path.join(os.path.dirname(s_p), "export_log.csv"), 'a',
                                  newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow([os.path.basename(s_p), idx + 1, round(area, 2), round(np.mean(rh), 2)])
                    except IOError as e:
                        QMessageBox.warning(self, "Export Warning", f"Image saved but log write failed:\n{e}")

    def handle_annotation_added(self, data):
        """将新增标注持久化到内存数据结构，并刷新显示。
        根据 chk_global_scope 决定标注归属：
          - 勾选"穿透所有切片"→ 存入 global_annotations['all']，所有切片可见
          - 未勾选 → 存入 global_annotations[当前切片索引]，仅该切片可见
        """
        if self.recon_mode_active:
            return
        tk = 'all' if self.chk_global_scope.isChecked() else self.current_3d_pos[0]
        if tk not in self.global_annotations:
            self.global_annotations[tk] = []
        self.global_annotations[tk].append(data)
        self.update_display()

    def handle_annotation_deleted(self, aid):
        """按 UUID 从所有切片的标注列表中删除指定标注。
        遍历所有键是因为用户可能在不知情的情况下删除了一个全局标注。
        """
        if self.recon_mode_active:
            return
        for k in self.global_annotations:
            self.global_annotations[k] = [a for a in self.global_annotations[k] if a['id'] != aid]
        self.update_display()

    def clear_current_slice_annotations(self):
        """清空当前切片的标注，并将整个 3D 蒙版重置为全零。
        注意：蒙版是 3D 的，清空操作影响全部切片（AI 分割结果一并清除）。
        """
        idx = self.current_3d_pos[0]
        if idx in self.global_annotations:
            self.global_annotations[idx] = []
        if self.volume_mask is not None:
            self.volume_mask = np.zeros_like(self.volume_hu)
        if not self.recon_mode_active:
            self.update_display()

    def change_active_tool(self, tid):
        """切换全局工具，并同步更新所有视图的 current_tool，确保各视图行为一致。"""
        self.active_tool = tid
        for v in self.views.values():
            v['view'].current_tool = tid

    def on_wheel_mpr(self, d, vid):
        if self.volume_hu is None or self.recon_mode_active: return
        plane = self.views[vid]['plane']; increment = -1 if d > 0 else 1
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape; z, y, x = self.current_3d_pos
        if plane == AXIAL: self.slider_slice.setValue(max(0, min(z + increment, Z_MAX - 1)))
        elif plane == CORONAL: self.current_3d_pos[1] = max(0, min(y + increment, Y_MAX - 1)); self.update_display()
        elif plane == SAGITTAL: self.current_3d_pos[2] = max(0, min(x + increment, X_MAX - 1)); self.update_display()

    def measure_hu(self, p, vid):
        if self.active_tool == TOOL_POINTER and self.volume_hu is not None and not self.recon_mode_active:
            vd = self.views.get(vid); c = vd['view'].get_real_coordinates(p); plane = vd['plane']
            if c: 
                try:
                    if plane == AXIAL: val = self.volume_hu[self.current_3d_pos[0], c[1], c[0]]
                    elif plane == CORONAL: val = self.volume_hu[c[1], self.current_3d_pos[1], c[0]]
                    elif plane == SAGITTAL: val = self.volume_hu[c[1], c[0], self.current_3d_pos[2]]
                    plane_str = {"Axial": "Axial", "Coronal": "Coronal", "Sagittal": "Sagittal"} if self.is_english else {"Axial": "横断面", "Coronal": "冠状面", "Sagittal": "矢状面"}
                    p_name = plane_str.get({AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[plane])
                    self.lbl_hu_value.setText(f"V{vid} [{p_name}] ({c[0]}, {c[1]}) : {val:.1f} HU")
                except Exception: pass

    def reset_all_states(self):
        """重置工作区到初始状态：恢复单窗布局、默认窗宽窗位、清空所有标注和弦图缓存。
        注意：仅在临床阅片模式（非重建实验室）下调用 update_display，
        避免在重建实验室中意外清空正在查看的重建结果。
        """
        self.combo_layout.setCurrentIndex(0)
        self.slider_ww.setValue(1500); self.slider_wl.setValue(-500)
        self.tool_btns['btn_ptr'].setChecked(True); self.change_active_tool(0)
        self.global_annotations = {'all': []}
        if self.volume_mask is not None:
            self.volume_mask = np.zeros_like(self.volume_hu)
        self.btn_mpr.setChecked(False)
        self.current_sinogram = None; self.current_theta = None
        self._last_recon_img = None
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]: b.setEnabled(False)
        # DMR/ART 只要有 DICOM 数据就可以运行（不依赖弦图）
        has_data = self.volume_hu is not None
        for b in [self.btn_dmr, self.btn_art]: b.setEnabled(has_data)
        for vid, v in self.views.items():
            v['cb_plane'].setCurrentIndex(AXIAL)
            v['preset'].setCurrentIndex(0); v['lock'].setChecked(False)
            v['chk_anno'].setChecked(True)
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
        if not self._read_dicom_dir(path):
            return
        pid = self._build_volume_hu()
        self._load_annotations_json(pid)

        z = self.volume_hu.shape[0]
        self.on_slice_changed(z // 2)
        for b in [self.btn_dmr, self.btn_art]:
            b.setEnabled(True)
        # 延迟 100ms 做 fitInView，确保 Qt 已完成首次绘制布局再计算缩放
        QTimer.singleShot(100, lambda: [
            vd['view'].fitInView(vd['view'].scene.sceneRect(), Qt.KeepAspectRatio)
            for vd in self.views.values() if not vd['container'].isHidden()
        ])
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

        self.dicom_datasets = [ds for ds in results if ds is not None]
        if not self.dicom_datasets:
            return False

        def _sort_key(ds):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                return int(getattr(ds, 'InstanceNumber', 0))

        self.dicom_datasets.sort(key=_sort_key)
        return True

    def _build_volume_hu(self):
        """从 dicom_datasets 构建 3D HU 数组，初始化蒙版、3D 光标、切片滑动条。返回 PatientID。

        HU 值转换公式（DICOM 标准）：
          HU = pixel_value × RescaleSlope + RescaleIntercept
          典型值：Slope=1, Intercept=-1024（GE 扫描仪常见），使得空气≈-1000 HU
        """
        ds = self.dicom_datasets[0]
        pid = str(getattr(ds, 'PatientID', 'N/A'))
        self.info_labels["ID"].setText(pid)
        self.info_labels["NAME"].setText(str(getattr(ds, 'PatientName', 'Unknown')).replace('^', ' '))
        self.info_labels["AGE"].setText(str(getattr(ds, 'PatientAge', 'N/A')))

        # 批量转换 HU 值：列表推导式遍历所有切片，堆叠为 3D float32 数组
        # getattr 提供默认值兼容不规范 DICOM（缺少 RescaleSlope/Intercept 的旧设备）
        self.volume_hu = np.array([
            d.pixel_array.astype(np.float32) * float(getattr(d, 'RescaleSlope', 1)) +
            float(getattr(d, 'RescaleIntercept', 0))
            for d in self.dicom_datasets
        ])
        self.volume_mask = np.zeros_like(self.volume_hu, dtype=np.uint8)
        self.global_annotations = {'all': []}
        z, y, x = self.volume_hu.shape
        # 默认将 3D 光标定位在体积中心（中间切片、中间行、中间列）
        self.current_3d_pos = [z // 2, y // 2, x // 2]
        self.slider_slice.setRange(0, z - 1)
        self.slider_slice.setValue(z // 2)
        return pid

    def _load_annotations_json(self, pid):
        """尝试加载同 PatientID 命名的注解 JSON 文件，恢复历史标注。文件不存在或损坏均静默跳过。"""
        af = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Exported_Lesions", f"{pid}_annotations.json")
        if not os.path.exists(af):
            return
        try:
            with open(af, 'r', encoding='utf-8') as f:
                for k, v in json.load(f).items():
                    # JSON 键只能是字符串，数字键需要转回 int
                    self.global_annotations[int(k) if k.isdigit() else k] = v
        except Exception as e:
            print(f"Warning: failed to load annotations from {af}: {e}")

    def _kickoff_ai(self):
        """启动后台 AI 推理。
        每次加载新数据时自增 generation 计数器；旧 AI 线程回调时若 generation 不匹配则静默
        丢弃结果，防止旧数据覆盖新数据的蒙版（竞态条件保护）。
        """
        self._ai_generation += 1
        gen = self._ai_generation
        self._ai_state = 'running'
        self.lbl_ai_status.setStyleSheet("color: #F1C40F; font-weight: bold;")
        self.lbl_ai_status.setText("Processing AI Pipeline..." if self.is_english else "状态: AI 引擎自动运算中...")
        # lambda 中用 g=gen 捕获当前 generation 值（闭包变量，防止后续自增影响比对）
        self.ai_thread = AutoAIEngineThread(
            self.volume_hu,
            callback=lambda mask, t, g=gen: self.on_auto_ai_finished(mask, t, g)
        )
        self.ai_thread.start()

    def on_auto_ai_finished(self, final_mask, time_ms, generation=None):
        """AI 推理完成的回调（由 QTimer.singleShot 投递到主线程执行）。

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
        self.lbl_ai_status.setText(f"Ready ({time_ms:.1f}ms)" if self.is_english else f"状态: 自动分割完成 ({time_ms:.1f}ms)")
        if not self.recon_mode_active:
            self.update_display()

    def on_slice_changed(self, idx):
        """切片滑动条 valueChanged 回调：更新 3D 光标 Z 轴坐标并刷新显示。"""
        self.current_3d_pos[0] = idx
        self.lbl_slice.setText(f"{'Slice: ' if self.is_english else '层数: '}{idx + 1} / {len(self.dicom_datasets)}")
        if not self.recon_mode_active:
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

        ww_m, wl_m = self.slider_ww.value(), self.slider_wl.value()
        self.lbl_ww.setText(f"WW: {ww_m}"); self.lbl_wl.setText(f"WL: {wl_m}")
        ds = self.dicom_datasets[z]
        px_sp = float(getattr(ds, 'PixelSpacing', [1, 1])[0])
        # SliceThickness 用于冠/矢状面像素宽高比计算；若缺失则估算为 px_sp×3（典型螺旋 CT 值）
        slice_thick = float(getattr(ds, 'SliceThickness', px_sp * 3))

        for vid, vdata in self.views.items():
            if vdata['container'].isHidden():
                continue
            self._render_clinical_plane(vdata, z, y, x, ww_m, wl_m, px_sp, slice_thick)

    def _render_recon_reference(self, z):
        """重建实验室分支：仅刷新 V1 的"真实切片"参考图，并重置 V2-V4 重建流水线状态。"""
        img_gt = self.volume_hu[z]
        ww, wl = self.slider_ww.value(), self.slider_wl.value()
        # 窗宽/窗位映射：将 HU 值线性映射到 [0, 255]
        img_windowed = np.clip(img_gt, wl - ww / 2, wl + ww / 2)
        img_windowed = ((img_windowed - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        img_windowed = np.ascontiguousarray(img_windowed)
        h, w = img_windowed.shape
        qimg = QImage(img_windowed.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.views[1]['view'].set_image(QPixmap.fromImage(qimg))
        self.views[1]['view'].clear_annotations()
        self.set_view_title(1, "V1 [Ground Truth]" if self.is_english else "V1 [真实切片]")
        # 切片改变后，弦图不再对应当前切片，必须重置重建流水线
        for vid in [2, 3, 4]:
            self.views[vid]['view'].image_item.setPixmap(QPixmap())
        self.current_sinogram = None
        self._cached_bp = None; self._cached_bp_sino = None
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]:
            b.setEnabled(False)
        self._set_recon_pending_titles()

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
        # 像素间距 sp=(行间距, 列间距)，用于标注测量时的实际尺寸换算
        if plane == AXIAL:
            hu = self.volume_hu[z, :, :]
            sp = (px_sp, px_sp)              # 横断面：行/列均为 PixelSpacing
        elif plane == CORONAL:
            hu = self.volume_hu[:, y, :]
            sp = (px_sp, slice_thick)        # 冠状面：行=PixelSpacing，列=SliceThickness
        elif plane == SAGITTAL:
            hu = self.volume_hu[:, :, x]
            sp = (px_sp, slice_thick)        # 矢状面：同冠状面

        # 窗宽窗位映射：HU → [0, 255] 线性映射
        img = np.clip(hu, wl - ww / 2, wl + ww / 2)
        img = ((img - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()

        # AI 蒙版叠加：仅 Axial 平面支持，颜色 #00ADB5（青色），alpha=100（约 40% 透明）
        mq = None
        if plane == AXIAL and vdata['chk_anno'].isChecked() and self.volume_mask is not None:
            sm = self.volume_mask[z]
            if np.any(sm):  # 性能优化：无蒙版时跳过 QImage 构建
                ov = np.zeros((h, w, 4), dtype=np.uint8)
                ov[sm == 1] = [0, 173, 181, 100]  # RGBA：青色半透明
                mq = QImage(ov.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

        vdata['view'].set_image(QPixmap.fromImage(qimg), mq, sp)
        vdata['view'].clear_annotations()  # 清除上一帧的标注图元，防止重影

        if plane == AXIAL and vdata['chk_anno'].isChecked():
            self._render_annotations(vdata, z, sp)

        # MPR 十字准线：联动开启时各平面投影不同的坐标轴对
        if self.btn_mpr.isChecked():
            if plane == AXIAL:      vdata['view'].draw_crosshair(x, y)
            elif plane == CORONAL:  vdata['view'].draw_crosshair(x, z)
            elif plane == SAGITTAL: vdata['view'].draw_crosshair(y, z)
        else:
            vdata['view'].draw_crosshair(0, 0, show=False)

    def _render_annotations(self, vdata, z, sp):
        """在视图场景中渲染当前切片的标注图元（仅 Axial 平面调用）。
        颜色区分：切片专属标注用青色，全局穿透标注用黄色；分组遍历避免 O(n²) 成员检查。
        """
        col_slice = QColor("#00ADB5")
        col_global = QColor("#F1C40F")
        slice_annos = self.global_annotations.get(z, [])
        global_annos = self.global_annotations.get('all', [])
        for annos, col in ((slice_annos, col_slice), (global_annos, col_global)):
            for anno in annos:
                if anno['type'] == 'ruler':
                    line = QGraphicsLineItem(QLineF(anno['p1'][0], anno['p1'][1], anno['p2'][0], anno['p2'][1]))
                    line.setPen(QPen(col, 2))
                    line.setToolTip(anno['id'])          # toolTip 存 UUID，Delete 键删除时用
                    line.setFlag(QGraphicsLineItem.ItemIsSelectable)
                    vdata['view'].scene.addItem(line)
                    # 距离计算：勾股定理，分别乘以 X/Y 方向像素间距换算为毫米
                    dist = math.sqrt(
                        ((anno['p2'][0] - anno['p1'][0]) * sp[1]) ** 2 +
                        ((anno['p2'][1] - anno['p1'][1]) * sp[0]) ** 2
                    )
                    txt = QGraphicsTextItem(f"{dist:.1f} mm")
                    txt.setDefaultTextColor(col)
                    txt.setFont(QFont("Arial", 11, QFont.Bold))
                    txt.setPos(anno['p2'][0] + 10, anno['p2'][1] + 10)
                    vdata['view'].scene.addItem(txt)
                elif anno['type'] == 'path':
                    pts = anno['points']
                    path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
                    for p in pts[1:]:
                        path.lineTo(QPointF(p[0], p[1]))
                    pen = QPen(col, 2)
                    pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setFlag(QGraphicsPathItem.ItemIsSelectable)
                    item.setToolTip(anno['id'])
                    vdata['view'].scene.addItem(item)

    def save_project(self):
        """将当前所有标注保存为 JSON 文件（以 PatientID 命名），方便下次加载时自动恢复。
        JSON 键必须为字符串（JSON 规范），整数切片索引在此序列化为字符串，加载时再转回 int。
        """
        if not self.dicom_datasets:
            return
        pid = str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown'))
        ed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions")
        os.makedirs(ed, exist_ok=True)
        try:
            with open(os.path.join(ed, f"{pid}_annotations.json"), 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.global_annotations.items()}, f, indent=4)
            QMessageBox.information(self, "Success", "Project Saved.")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Failed to save project:\n{e}")

    def select_folder(self):
        """打开文件夹选择对话框，选择后触发 DICOM 加载。"""
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p:
            self.load_data(p)

    # =========================================================================
    # 矩阵重建共用工具
    # =========================================================================
    def _prepare_small_image_and_sinogram(self):
        """为 DMR/ART 准备小尺寸图像及其弦图。

        步骤：
          1. 取当前切片的 HU 值，归一化到 [0, 1]
          2. 用双三次插值（ndimage.zoom）缩小到 n×n（n 由 UI 下拉框选择：16/32/64）
          3. 施加圆形掩码（circle=True 的 radon 只处理内切圆区域，圆外置0）
             关键：若不施加此掩码，V1（原图）角落有值而 V4（重建）角落为0，
             误差图会在角落显示虚假的大误差，迷惑用户误判算法质量
          4. 对小图做 Radon 变换生成弦图，角度数量与 UI 的 60°/120°/180° 选项对应

        返回：(img_small, sinogram, theta, n)
        """
        if not self.dicom_datasets or self.volume_hu is None:
            return None, None, None, None
        z = self.current_3d_pos[0]
        img_gt = self.volume_hu[z]
        denom = img_gt.max() - img_gt.min()
        img_norm = (img_gt - img_gt.min()) / (denom if denom > 0 else 1.0)
        n = int(self.cb_matrix_size.currentText().split('×')[0])
        img_small, sinogram, theta = recon_lib.prepare_small_image(img_norm, n, self._get_n_angles())
        return img_small, sinogram, theta, n

    def _build_system_matrix(self, n, theta):
        """逐像素构建系统矩阵 A，用于 DMR（最小二乘）和 ART/SIRT（迭代）。

        系统矩阵 A 的物理含义：
          A[i, j] 表示"第 j 个像素对第 i 条射线的贡献量"（即射线 i 穿过像素 j 的路径长度）。
          用于线性方程组 A·x = p，其中：
            x = 展平的图像（n×n 个未知像素值）
            p = 展平的弦图（所有射线的测量值）

        构建方法：
          将图像逐像素置1（单位冲激），对每个像素单独做 Radon 变换，
          其结果即为矩阵 A 的对应列（该像素对所有射线的贡献）。
          这是最直观的构建方式，缺点是时间复杂度 O(n²) 次 Radon 变换。

        缓存策略：
          key = (n, 角度数, 起始角, 终止角)；图像尺寸和角度配置不变时直接复用，
          64×64 × 180角的 A 矩阵约需数分钟构建，缓存节省大量等待时间。
        """
        n_pixels = n * n
        step = max(1, n_pixels // 50)
        prog = QProgressDialog(
            f"Building {n}x{n} system matrix..." if self.is_english else f"构建 {n}x{n} 系统矩阵...",
            None, 0, n_pixels, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        def _progress(j, _total):
            if j % step == 0:
                prog.setValue(j)
                QApplication.processEvents()

        A, key = recon_lib.build_system_matrix(
            n, theta, self._cached_A, self._cached_A_key, progress_cb=_progress
        )
        prog.setValue(n_pixels)
        prog.close()
        self._cached_A = A
        self._cached_A_key = key
        return A

    def _fit_recon_views(self, smooth=True):
        """刷新所有重建视图的渲染质量设置并重新适配缩放。

        smooth 参数控制 SmoothPixmapTransform（双线性插值）：
          - True（BP/FBP/DFR）：连续灰度图像应使用平滑插值，缩放后不出现锯齿
          - False（DMR/ART）：像素块图像应关闭平滑，保留色块边界清晰度

        延迟 0ms（singleShot(0)）的原因：
          display_numpy_image 中的 set_image 调用 fitInView 时图像可能还未完成布局，
          defer 到下一个事件循环 tick 保证几何计算基于最终尺寸进行。
        """
        for vid in [1, 2, 3, 4]:
            v = self.views[vid]['view']
            v.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
            QTimer.singleShot(0, lambda vv=v: vv.fitInView(vv.scene.sceneRect(), Qt.KeepAspectRatio))

    # =========================================================================
    # 直接矩阵重建法 (Direct Matrix Reconstruction, DMR)
    # =========================================================================
    def run_dmr(self):
        """DMR：将 CT 重建问题建模为线性方程组 A·x = p，用最小二乘法直接求解。

        数学原理：
          A·x = p
            A: 系统矩阵 (n_rays × n²)，描述每个像素对每条射线的贡献
            x: 未知图像（展平为向量，长度 n²）
            p: 测量的弦图（展平为向量，长度 n_rays）

          np.linalg.lstsq 求最小二乘解 x* = argmin ||A·x - p||₂²
          等价于求伪逆：x* = A⁺·p = (AᵀA)⁻¹Aᵀ·p

        优点：精确的代数解，无迭代误差
        缺点：
          1. A 矩阵构建耗时（O(n²) 次 Radon 变换）
          2. lstsq 求解内存消耗大（对 64×64 约需 ~GB 级中间矩阵）
          3. 实际 CT 系统 n 通常为 512 甚至更大，DMR 不可扩展

        视图分配：V1=原图, V2=弦图, V3=误差图, V4=重建结果
        渲染：smooth=False 保留像素块（与 kron 上采样配合）
        """
        if self.volume_hu is None:
            return
        img_small, sinogram, theta, n = self._prepare_small_image_and_sinogram()
        if img_small is None:
            return
        A = self._build_system_matrix(n, theta)
        p_vec = sinogram.flatten().astype(np.float32)
        img_recon, t_ms = recon_lib.compute_dmr(A, p_vec, n)
        self._last_recon_img = img_recon   # 供"生成弦图"按钮对重建结果做正向投影
        error_map = np.abs(img_small - img_recon)

        self.display_numpy_image(1, recon_lib.upscale_recon(img_small, n))
        self.display_numpy_image(2, sinogram.T)
        self.display_numpy_image(3, recon_lib.upscale_recon(error_map, n))
        self.display_numpy_image(4, recon_lib.upscale_recon(img_recon, n))
        self._fit_recon_views(smooth=False)

        rmse = float(np.sqrt(np.mean(error_map ** 2)))
        self.set_view_title(1, f"V1 [Orig {n}x{n}]" if self.is_english else f"V1 [原始 {n}x{n}]")
        self.set_view_title(2, "V2 [Sinogram]" if self.is_english else "V2 [投影弦图]")
        self.set_view_title(3, f"V3 [Error RMSE={rmse:.4f}]")
        self.set_view_title(4, f"V4 [DMR {n}x{n}]")
        self.lbl_time.setText(f"DMR lstsq: {t_ms:.1f} ms" if self.is_english else f"直接矩阵重建耗时: {t_ms:.1f} ms")

    # =========================================================================
    # ART / SIRT 迭代重建
    # =========================================================================
    def run_art_sirt(self):
        """ART 和 SIRT 迭代重建——通过逐步修正逼近方程组的解。

        ART（代数重建技术，Algebraic Reconstruction Technique）：
          逐射线更新，每次用一条射线的残差修正整个图像：
            x ← x + (p_i - A_i·x) / ||A_i||² · A_i
          其中 A_i 是矩阵第 i 行（该射线对所有像素的权重）。
          特点：每次迭代顺序处理所有射线（串行），收敛快但对噪声敏感。

        SIRT（同步迭代重建技术，Simultaneous Iterative Reconstruction Technique）：
          一次性用所有射线的残差做加权平均更新：
            x ← x + C · Aᵀ · (R · (p - A·x))
          其中 C = diag(1/列和)，R = diag(1/行和) 是归一化矩阵。
          特点：每次迭代计算量更大（矩阵乘法），但噪声鲁棒性更好，收敛更平滑。

        两种方法共同特点：
          - x = clip(x, 0) 每轮强制非负约束（HU值不存在负像素强度）
          - 支持中途取消（wasCanceled），取消后显示当前迭代的中间结果
          - 视图分配与 DMR 相同：V1=原图, V2=弦图, V3=误差, V4=重建
        """
        if self.volume_hu is None:
            return
        img_small, sinogram, theta, n = self._prepare_small_image_and_sinogram()
        if img_small is None:
            return
        A = self._build_system_matrix(n, theta)
        p_vec = sinogram.flatten().astype(np.float32)
        method = self.cb_art_method.currentText()
        n_iter = int(self.cb_art_iter.currentText())
        prog_iter = QProgressDialog(
            f"Running {method} ({n_iter} iterations)..." if self.is_english else f"正在运行 {method}（共 {n_iter} 次迭代）...",
            "Cancel" if self.is_english else "取消", 0, n_iter, self)
        prog_iter.setWindowModality(Qt.WindowModal)
        prog_iter.show()

        def _cancel():
            QApplication.processEvents()
            return prog_iter.wasCanceled()

        def _progress(it):
            prog_iter.setValue(it + 1)
            QApplication.processEvents()

        if method == 'ART':
            img_recon, t_ms = recon_lib.compute_art(
                A, p_vec, n, n_iter, cancel_check=_cancel, progress_cb=_progress)
        else:
            img_recon, t_ms = recon_lib.compute_sirt(
                A, p_vec, n, n_iter, cancel_check=_cancel, progress_cb=_progress)

        prog_iter.close()

        self._last_recon_img = img_recon   # 供"生成弦图"按钮对重建结果做正向投影
        error_map = np.abs(img_small - img_recon)
        self.display_numpy_image(1, recon_lib.upscale_recon(img_small, n))
        self.display_numpy_image(2, sinogram.T)
        self.display_numpy_image(3, recon_lib.upscale_recon(error_map, n))
        self.display_numpy_image(4, recon_lib.upscale_recon(img_recon, n))
        self._fit_recon_views(smooth=False)
        rmse = float(np.sqrt(np.mean(error_map ** 2)))
        self.set_view_title(1, f"V1 [Orig {n}x{n}]" if self.is_english else f"V1 [原始 {n}x{n}]")
        self.set_view_title(2, "V2 [Sinogram]" if self.is_english else "V2 [投影弦图]")
        self.set_view_title(3, f"V3 [Error RMSE={rmse:.4f}]")
        self.set_view_title(4, f"V4 [{method} {n_iter}it {n}x{n}]")
        self.lbl_time.setText(f"{method} ({n_iter}it): {t_ms:.1f} ms" if self.is_english else f"{method} ({n_iter}次迭代)耗时: {t_ms:.1f} ms")


if __name__ == "__main__":
    # freeze_support：多进程 'spawn' 模式在 macOS/Windows 打包环境中必须调用，
    # 防止子进程重入主程序逻辑导致无限递归启动
    import multiprocessing
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MedicalViewer()
    window.show()
    sys.exit(app.exec())
    