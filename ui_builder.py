# =============================================================================
# UI 构建 Mixin
# 负责：主窗口三栏布局与全部控件的构建（左工具栏 / 中视图栅格 / 右控制面板 /
#       临床阅片 Tab / 重建实验室 Tab / 单个联动视图 / 暗色主题）。
#
# 设计：以 Mixin 形式并入 MedicalViewer，在 __init__ 中调用 setup_stylesheet()
#       与 init_ui() 完成装配。这些方法创建 self.xxx 控件并把信号连到留在
#       main / 各 Mixin 的槽方法上，全部经 self 在合并实例上解析。仅负责"搭建"，
#       运行期逻辑（update_language 重译、change_view_plane、switch_layout 等）仍在 main。
# =============================================================================

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from constants import AXIAL, MANUAL_TRACK_LABEL
from graphics_view import MedicalGraphicsView


class UiBuilderMixin:
    """主窗口 UI 构建方法集合，混入 MedicalViewer。"""

    def setup_stylesheet(self):
        """从 style.qss 加载暗色主题样式表；文件缺失时静默跳过，UI 仍可用 Qt 默认样式渲染。"""
        qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
        try:
            with open(qss_path, encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except OSError as e:
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
                     (4, 'btn_rec'), (3, 'btn_las'), (5, 'btn_trk'),
                     (6, 'btn_brush'), (7, 'btn_erase'),  # 6/7=分割修正：画笔补画 / 橡皮擦除
                     (8, 'btn_roi')]                       # 8=椭圆 ROI 密度测量
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
        self.tab_clinical = QWidget()
        self.tab_recon = QWidget()
        self.tabs.addTab(self.tab_clinical, "临床阅片")
        self.tabs.addTab(self.tab_recon, "重建实验室")
        self._build_clinical_tab()
        self._build_recon_tab()
        # 信号连接必须在两个 tab 构建完成后：否则 addTab 触发的 currentChanged 会
        # 在 btn_dfr 等重建控件尚未创建时调用 on_tab_changed，抛 AttributeError
        self.tabs.currentChanged.connect(self.on_tab_changed)
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
        # 标签宽度按字体实算，不写死：曾硬编码 76px，结果中文「层数: 117 / 233」被裁成
        # 「117 / 23」、英文「Slice: 117 / 233」更短一截——用户看到的总层数是**错的**，
        # 比排版难看严重得多。取中英文所有标签的最长文本之最大值，故切换语言时
        # 宽度不变、三个滑条始终左对齐，也不会因语种不同而重新裁切。
        w_need = max(QFontMetrics(lbl.font()).horizontalAdvance(s) for lbl in
                     (self.lbl_slice, self.lbl_ww, self.lbl_wl)
                     for s in ("Slice: 9999 / 9999", "层数: 9999 / 9999",
                               "WW: 4000", "WL: -1200")) + 6
        for lbl, slider in [(self.lbl_slice, self.slider_slice), (self.lbl_ww, self.slider_ww), (self.lbl_wl, self.slider_wl)]:
            lbl.setFixedWidth(w_need); row = QHBoxLayout(); row.setSpacing(6); row.addWidget(lbl); row.addWidget(slider); dl.addLayout(row)
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
        # 信息叠加显隐 + 反色（医学影像软件标配）
        h_disp = QHBoxLayout()
        self.chk_overlay = QCheckBox("信息叠加"); self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self._toggle_overlay)
        self.chk_invert = QCheckBox("反色"); self.chk_invert.toggled.connect(self.update_display)
        self.chk_anon = QCheckBox("脱敏"); self.chk_anon.toggled.connect(self._toggle_anonymize)
        h_disp.addWidget(self.chk_overlay); h_disp.addWidget(self.chk_invert); h_disp.addWidget(self.chk_anon)
        dl.addLayout(h_disp)
        # Cine 电影播放（往返连播，速度可调） + 双序列随访对比
        h_nav = QHBoxLayout()
        self.btn_cine = QPushButton("▶ 播放"); self.btn_cine.setProperty("class", "ActionBtn")
        self.btn_cine.clicked.connect(self.toggle_cine)
        self.cb_cine_speed = QComboBox()   # 播放速度：itemData 为定时器间隔(ms)
        self.cb_cine_speed.addItem("慢", 250); self.cb_cine_speed.addItem("中", 120); self.cb_cine_speed.addItem("快", 50)
        self.cb_cine_speed.setCurrentIndex(1)
        self.cb_cine_speed.currentIndexChanged.connect(self._on_cine_speed_changed)
        self.btn_compare = QPushButton("加载对比序列"); self.btn_compare.setProperty("class", "ActionBtn")
        self.btn_compare.clicked.connect(self.toggle_compare)
        # 对比模式的平面内刚性配准开关：默认关闭，因为配准会改变 V2 显示的像素
        # （重采样），用户应当明确知道自己看的是配准后的图而不是原始层面。
        # 刻意不设 objectName="ViewOption"：qss 里那条规则是 `min-width/max-width: 60px`，
        # 为视图顶部「显示 / 锁定」两个短标签复选框而写。本控件英文作 "Register"（需 72px），
        # 套上会被样式表钉死在 60px 裁成「Regist」——qss 的 max-width 优先级高于
        # setMinimumWidth，光调布局 stretch 是修不动的。外观仍继承通用 QCheckBox 样式。
        self.chk_register = QCheckBox("配准")
        # 只在对比模式下有意义：未加载对比序列时它无事可做，可勾但毫无效果，
        # 属于会误导用户的控件，故默认禁用，由 _enter/_exit_compare_mode 启停。
        self.chk_register.setEnabled(False)
        self.chk_register.setMinimumWidth(max(
            QFontMetrics(self.chk_register.font()).horizontalAdvance(s)
            for s in ("Register", "配准")) + 30)
        self.chk_register.setToolTip("对比模式：平面内刚性配准（平移+旋转）后再算差异")
        self.chk_register.stateChanged.connect(self.update_display)
        # 拆成两行：原先四个控件挤一行，英文下「Load Comparison」被裁成「d Compari」、
        # 「Register」裁成「Regist」、速度下拉只剩一个字符。且这一行本就混了两个功能域
        # ——Cine 是单序列内的播放导航，对比/配准属于双序列随访——分行后语义也更清楚。
        h_nav.addWidget(self.btn_cine, 2); h_nav.addWidget(self.cb_cine_speed, 1)
        dl.addLayout(h_nav)
        h_cmp = QHBoxLayout()
        h_cmp.addWidget(self.btn_compare, 3); h_cmp.addWidget(self.chk_register, 1)
        dl.addLayout(h_cmp)
        self.grp_display.setLayout(dl)
        t1_lay.addWidget(self.grp_display)

        # AI 状态分组（原 AI 按钮改为状态显示，因为已全自动）
        self.grp_ai = QGroupBox("自动化 AI 引擎")
        ai_lay = QVBoxLayout(); ai_lay.setContentsMargins(10, 15, 10, 10)
        self.lbl_ai_status = QLabel("状态: 待机中")
        self.lbl_ai_status.setStyleSheet("color: #8B949E; font-weight: bold;")
        ai_lay.addWidget(self.lbl_ai_status)
        # 器官图例：显示当前切片检测到的器官及其颜色（随切片刷新）
        self.lbl_ai_legend = QLabel("")
        self.lbl_ai_legend.setWordWrap(True)
        self.lbl_ai_legend.setTextFormat(Qt.RichText)
        self.lbl_ai_legend.setStyleSheet("font-size: 11px;")
        # 图例条目做成可点击链接：单击切换该器官在蒙版叠加中的显隐
        self.lbl_ai_legend.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.lbl_ai_legend.linkActivated.connect(self._toggle_organ)
        ai_lay.addWidget(self.lbl_ai_legend)
        # 器官定量：分割完成后列出各器官体积/平均 HU，并支持导出 CSV
        self.lbl_ai_stats = QLabel("")
        self.lbl_ai_stats.setWordWrap(True)
        self.lbl_ai_stats.setTextFormat(Qt.RichText)
        self.lbl_ai_stats.setStyleSheet("font-size: 11px; color: #B0B8C4;")
        ai_lay.addWidget(self.lbl_ai_stats)
        self.btn_export_stats = QPushButton("导出定量 CSV")
        self.btn_export_stats.setProperty("class", "ActionBtn")
        self.btn_export_stats.setEnabled(False)
        self.btn_export_stats.clicked.connect(self.export_organ_stats)
        ai_lay.addWidget(self.btn_export_stats)
        # 三维表面重建：对选中器官做 marching cubes，多视角预览 + 形状特征 + STL 导出
        self.btn_mesh3d = QPushButton("三维重建预览")
        self.btn_mesh3d.setProperty("class", "ActionBtn")
        self.btn_mesh3d.setEnabled(False)
        self.btn_mesh3d.clicked.connect(self.show_mesh3d)
        ai_lay.addWidget(self.btn_mesh3d)
        # 合规免责声明：AI 结果非诊断依据，常驻显示
        self.lbl_disclaimer = QLabel("⚠ AI 结果与器官标签为自动推断，仅供参考，非诊断依据。")
        self.lbl_disclaimer.setWordWrap(True)
        self.lbl_disclaimer.setStyleSheet("color: #C0392B; font-size: 10px;")
        ai_lay.addWidget(self.lbl_disclaimer)
        self.grp_ai.setLayout(ai_lay)
        t1_lay.addWidget(self.grp_ai)

        # 测量与清理分组
        self.grp_measure = QGroupBox("测量与清理")
        ml = QVBoxLayout(); ml.setContentsMargins(10, 15, 10, 10)
        # 光标 HUD：鼠标悬停时实时显示 (x,y,z) 坐标 / HU 值 / 所在器官
        self.lbl_hud = QLabel("")
        self.lbl_hud.setStyleSheet("color: #8B949E; font-family: monospace; font-size: 11px; min-height: 16px; max-height: 16px;")
        self.lbl_hud.setAlignment(Qt.AlignCenter); ml.addWidget(self.lbl_hud)
        self.lbl_hu_value = QLabel()
        self.lbl_hu_value.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 13px; min-height: 18px; max-height: 18px;")
        self.lbl_hu_value.setAlignment(Qt.AlignCenter); ml.addWidget(self.lbl_hu_value)
        self.chk_global_scope = QCheckBox("新标注穿透所有切片"); ml.addWidget(self.chk_global_scope)
        # 分割修正画笔半径（画笔/橡皮工具共用）
        h_brush = QHBoxLayout()
        self.lbl_brush = QLabel("画笔半径:")
        self.spin_brush = QSpinBox(); self.spin_brush.setRange(1, 40); self.spin_brush.setValue(6)
        self.spin_brush.valueChanged.connect(self._set_brush_radius)
        h_brush.addWidget(self.lbl_brush); h_brush.addWidget(self.spin_brush)
        ml.addLayout(h_brush)
        # 画笔目标：补画写入哪个标签（手动标注 or 当前检出的某个器官，使修正计入该器官定量）
        h_target = QHBoxLayout()
        self.lbl_paint_target = QLabel("画笔目标:")
        self.cb_paint_target = QComboBox()
        self.cb_paint_target.addItem("手动标注", MANUAL_TRACK_LABEL)
        h_target.addWidget(self.lbl_paint_target); h_target.addWidget(self.cb_paint_target)
        ml.addLayout(h_target)
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
        # 采样密度：在选定角度范围内的投影数量倍率。越高角度间隔越细、重建质量越好（越慢）。
        h_dens = QHBoxLayout()
        self.lbl_oversample = QLabel("采样密度:")
        self.combo_oversample = QComboBox()
        self.combo_oversample.addItems(["标准 1×", "高 2×", "超高 4×"])
        self.combo_oversample.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_dens.addWidget(self.lbl_oversample); h_dens.addWidget(self.combo_oversample)
        play.addLayout(h_dens)
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
        self.grp_matrix = QGroupBox("直接矩阵重建 && ART / SIRT")
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


    def create_independent_view(self, vid, plane=AXIAL):
        c = QFrame(); c.setObjectName("ViewContainer"); lay = QVBoxLayout(c); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        t = QFrame(); t.setObjectName("ViewToolbar"); t.setFixedHeight(32)
        tl = QHBoxLayout(t); tl.setContentsMargins(8,2,8,2); tl.setSpacing(6)
        lt = QLabel(f"V{vid}"); lt.setStyleSheet("color: #C9D1D9; font-weight: bold; min-width: 20px;")
        cb_plane = QComboBox(); cb_plane.setFixedWidth(80)
        ps = QComboBox(); ps.setFixedWidth(85); ps.currentIndexChanged.connect(self.update_display)
        # 厚层投影：模式 + 厚度（层数）。默认 Slice/1 时行为与原单层切片完全一致，
        # 故不改变既有默认体验；MIP 用于肺结节/血管，MinIP 用于气道，AIP 用于降噪。
        cb_proj = QComboBox(); cb_proj.setFixedWidth(72)
        sp_thick = QSpinBox(); sp_thick.setRange(1, 200); sp_thick.setValue(1); sp_thick.setFixedWidth(52)
        sp_thick.setEnabled(False)          # 单层模式下厚度无意义，选到投影模式才启用
        cb_proj.currentIndexChanged.connect(lambda i, s=sp_thick: (s.setEnabled(i > 0), self.update_display()))
        sp_thick.valueChanged.connect(self.update_display)
        an = QCheckBox(); an.setObjectName("ViewOption"); an.setChecked(True); an.stateChanged.connect(self.update_display)
        lk = QCheckBox(); lk.setObjectName("ViewOption"); lk.stateChanged.connect(self.update_display)
        tl.addWidget(lt); tl.addWidget(cb_plane); tl.addWidget(ps); tl.addWidget(cb_proj); tl.addWidget(sp_thick)
        tl.addStretch(); tl.addWidget(an); tl.addWidget(lk)
        v = MedicalGraphicsView(vid)
        v.clicked_pos.connect(lambda p, id=vid: self.measure_hu(p, id))
        v.wheel_scrolled.connect(lambda d, id=vid: self.on_wheel_mpr(d, id))
        v.annotation_added.connect(self.handle_annotation_added)
        v.annotation_deleted.connect(self.handle_annotation_deleted)
        v.crop_requested.connect(lambda pts, id=vid: self.handle_crop_requested(id, pts))
        v.track_requested.connect(lambda r, id=vid: self.handle_3d_track_requested(id, r))
        v.window_changed.connect(self.on_window_changed_by_mouse)
        v.mouse_hovered.connect(lambda pos, id=vid: self.sync_crosshair(pos, id))
        v.seg_paint_requested.connect(lambda pts, er, id=vid: self.handle_seg_paint(id, pts, er))
        lay.addWidget(t); lay.addWidget(v); t.raise_()
        self.views[vid] = {'container':c, 'cb_plane': cb_plane, 'preset':ps, 'lock':lk, 'chk_anno':an, 'view':v, 'plane': plane, 'title_label': lt,
                           'cb_proj': cb_proj, 'sp_thick': sp_thick}
        cb_plane.currentIndexChanged.connect(lambda idx, v_id=vid: self.change_view_plane(v_id, idx))

