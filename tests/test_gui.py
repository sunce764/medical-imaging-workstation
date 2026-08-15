#!/usr/bin/env python
# =============================================================================
# 医学影像工作站 —— 回归测试套件
#
# 覆盖：启动/工具栏、AI 推理引擎（取消/进度/信号回调）、历次修复项、
#      多器官分割渲染与定量、分割手动编辑（画笔/橡皮/目标/撤销）、
#      椭圆 ROI（渲染/拖动/缩放/命中）、采样密度、双序列对比（配准/守卫）、
#      Cine（往返/调速/键盘）、合规（脱敏/免责）、重建算法数值正确性（解析模体验算）。
#
# 运行：conda activate dicom_gui && python tests/test_gui.py
#      （离屏 Qt，无需真实显示；依赖同目录 ../肺癌 真实数据自动加载）
# 退出码 0 = 全部通过；非 0 = 有失败。
# =============================================================================
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QGraphicsTextItem, QGraphicsView, QMessageBox

import ai_engine
import main as m
from constants import AXIAL, CORONAL, MANUAL_TRACK_LABEL, TOOL_POINTER
from graphics_view import ROIGraphicsItem

# 静音弹窗，避免离屏阻塞。三个都必须 stub：question 曾被遗漏，导致触发
# 「矩形截取 → 是否保存?」的测试挂死（模态框弹出后无人应答，进程永久阻塞）。
# question 返回 No：测试默认不触发保存副作用，需要保存路径的用例自行临时改写。
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

_FAILS = []


def check(cond, label):
    print(("  PASS " if cond else "  FAIL ") + label)
    if not cond:
        _FAILS.append(label)


def drain(engine, app, extra=40):
    while engine.isRunning():
        app.processEvents()
    for _ in range(extra):
        app.processEvents()


def test_ai_engine(app):
    print("[AI 引擎] 取消 / 进度 / 信号回调")
    vol = np.full((40, 64, 64), -500, dtype=np.float32)
    # 取消：start 前作废 -> 不回调
    c = {"n": 0}
    e = ai_engine.AutoAIEngineThread(vol, callback=lambda mk, t: c.__setitem__("n", c["n"] + 1),
                                     model_path="/nonexistent.onnx")
    e.cancel(); e.start(); drain(e, app)
    check(c["n"] == 0, "取消后不触发回调")
    # 正常（数学降级）-> 信号回调一次
    c = {"n": 0, "mask": None}
    e = ai_engine.AutoAIEngineThread(vol, callback=lambda mk, t: (c.__setitem__("n", c["n"] + 1),
                                                                  c.__setitem__("mask", mk)),
                                     model_path="/nonexistent.onnx")
    e.start(); drain(e, app)
    check(c["n"] == 1 and c["mask"] is not None and c["mask"].dtype == np.uint8,
          "跨线程信号回调触发一次，返回 uint8 标签图")
    check(hasattr(ai_engine, "_SESSION_CACHE"), "InferenceSession 缓存接口存在")


def test_startup(v):
    print("[启动/工具栏]")
    check(v.volume_hu is not None and v.volume_hu.shape[0] == 233, "自动加载主序列 (233 层)")
    check(len(v.tool_btns) == 9, "工具栏含 9 个工具 (指针/卡尺/画笔/矩形/套索/追踪/分割画笔/橡皮/ROI)")
    # 结构：重建/对比逻辑经 Mixin 混入（拆分 main.py 后的架构约束）
    from compare_lab import CompareMixin
    from recon_lab import ReconLabMixin
    check(isinstance(v, ReconLabMixin), "MedicalViewer 混入 ReconLabMixin")
    check(all(hasattr(v, mth) for mth in
              ("generate_sinogram", "run_bp", "run_fbp", "run_dfr", "run_dmr", "run_art_sirt",
               "display_numpy_image", "_render_recon_reference", "_enter_recon_mode")),
          "重建方法经 mixin 全部就位")
    check(isinstance(v, CompareMixin), "MedicalViewer 混入 CompareMixin")
    check(all(hasattr(v, mth) for mth in
              ("toggle_compare", "_read_compare_dir", "_enter_compare_mode",
               "_exit_compare_mode", "_render_compare", "_show_windowed")),
          "对比方法经 mixin 全部就位")
    from annotation_lab import AnnotationMixin
    check(isinstance(v, AnnotationMixin), "MedicalViewer 混入 AnnotationMixin")
    check(all(hasattr(v, mth) for mth in
              ("handle_seg_paint", "_undo_mask_edit", "handle_annotation_added",
               "_render_annotations", "_compute_organ_stats", "_update_legend",
               "save_project", "_load_annotations_json", "_load_saved_mask")),
          "标注/分割方法经 mixin 全部就位")
    from ui_builder import UiBuilderMixin
    check(isinstance(v, UiBuilderMixin), "MedicalViewer 混入 UiBuilderMixin")
    check(all(hasattr(v, w) for w in
              ("left_toolbar", "main_splitter", "right_panel", "tabs", "tool_btns",
               "slider_ww", "btn_dfr", "combo_layout")) and len(v.views) == 4,
          "UI 经 mixin 完整构建（4 视图 + 关键控件就位）")
    from interaction import InteractionMixin
    check(isinstance(v, InteractionMixin), "MedicalViewer 混入 InteractionMixin")
    check(all(hasattr(v, mth) for mth in
              ("on_mpr_toggled", "sync_crosshair", "on_wheel_mpr", "measure_hu",
               "toggle_cine", "_cine_step", "_stop_cine")),
          "Cine/MPR 交互方法经 mixin 全部就位")
    # keyPressEvent 是 Qt 重写，必须在本体（MRO 中 QMainWindow 先于 Mixin，否则被遮蔽）
    check("keyPressEvent" in m.MedicalViewer.__dict__, "keyPressEvent 保留在 MedicalViewer 本体")


def test_prior_fixes(v, app):
    print("[历次修复项]")
    # recon 调窗不清弦图；切片才重置
    v.tabs.setCurrentIndex(1); app.processEvents()
    for i in range(1, 5):
        check(v.views[i]['view'].overlay_lines == {}, "进入重建实验室清空各视图 overlay") if i == 1 else None
    v.slider_slice.setValue(100); app.processEvents()   # 经正常路径同步 _recon_ref_z
    v.generate_sinogram(); app.processEvents()
    sino = v.current_sinogram
    v.slider_ww.setValue(2200); app.processEvents()
    check(v.current_sinogram is sino, "重建模式调窗不清弦图")
    # 切片滑条在重建模式生效（V1 跟随）+ 换切片重置流水线并清链式源图
    v._last_recon_img = np.zeros((64, 64), np.float32)
    v.slider_slice.setValue(50); app.processEvents()
    check(v._recon_ref_z == 50, "重建模式切片滑条生效，V1 跟随新层")
    check(v.current_sinogram is None and v._last_recon_img is None, "换切片重置弦图并清链式源图")
    v.tabs.setCurrentIndex(0); app.processEvents()
    # MPR 悬停联动
    v.btn_mpr.setChecked(True); v.views[1]['plane'] = CORONAL
    v.sync_crosshair(QPointF(100, 60), 1)
    check(v.current_3d_pos[0] == 60 and v.slider_slice.value() == 60, "MPR 悬停同步光标与滑条")
    v.btn_mpr.setChecked(False); v.views[1]['plane'] = AXIAL
    # reset 清 HUD / _user_zoomed；load 清 hidden
    v.lbl_hud.setText("x"); v.views[1]['view']._user_zoomed = True
    v.reset_all_states(); app.processEvents()
    check(v.lbl_hud.text() == "" and v.views[1]['view']._user_zoomed is False, "重置清 HUD 并复位缩放标志")
    check(v._hidden_organs == set(), "加载后图例隐藏集合为空")
    # resize 不再 AttributeError（既有崩溃修复）
    try:
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QResizeEvent
        v.views[1]['view'].resizeEvent(QResizeEvent(QSize(400, 400), QSize(300, 300)))
        check(True, "有图像时 resize 不崩 (_user_zoomed 已初始化)")
    except AttributeError as ex:
        check(False, f"resize 崩溃: {ex}")


def test_multiorgan_and_edit(v, app):
    print("[多器官分割 / 手动编辑]")
    z = v.current_3d_pos[0]
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.volume_mask[z][50:100, 50:100] = 10
    v._update_organ_stats()
    check(any(r['id'] == 10 for r in v._organ_stats), "器官定量统计含检出器官")
    datas = [v.cb_paint_target.itemData(i) for i in range(v.cb_paint_target.count())]
    check(MANUAL_TRACK_LABEL in datas and 10 in datas, "画笔目标下拉含手动标注 + 检出器官")
    # 画笔补进器官 10（计入该器官）
    v.cb_paint_target.setCurrentIndex(v.cb_paint_target.findData(10))
    before = int((v.volume_mask[z] == 10).sum())
    v.handle_seg_paint(1, [(200, 200), (230, 200)], False)
    check(int((v.volume_mask[z] == 10).sum()) > before, "分割画笔补进指定器官标签")
    v._undo_mask_edit()
    check(int((v.volume_mask[z] == 10).sum()) == before, "Ctrl+Z 撤销分割编辑")
    # 橡皮擦 AI 标签
    v.handle_seg_paint(1, [(70, 70)], True)
    check(int((v.volume_mask[z] == 10).sum()) < before, "橡皮可擦除 AI 分割")
    # 蒙版叠加须覆盖三个平面：volume_mask 与 volume_hu 同形状，各平面按同一索引取切片。
    # 曾硬编码 `plane == AXIAL`，使冠/矢状面看不到任何分割（MPR 与 AI 两大功能未打通）。
    Z, H, W = v.volume_mask.shape
    v.volume_mask[:] = 0
    v.volume_mask[Z // 2 - 5:Z // 2 + 5, H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20] = 5
    zc, yc, xc = Z // 2, H // 2, W // 2
    for nm, msk, hu in (("Axial", v.volume_mask[zc, :, :], v.volume_hu[zc, :, :]),
                        ("Coronal", v.volume_mask[:, yc, :], v.volume_hu[:, yc, :]),
                        ("Sagittal", v.volume_mask[:, :, xc], v.volume_hu[:, :, xc])):
        check(msk.shape == hu.shape and int((msk != 0).sum()) > 0,
              f"{nm} 面蒙版切片与影像切片同形 {msk.shape} 且含分割 (得 {int((msk != 0).sum())} 体素)")


def test_roi(v, app):
    print("[椭圆 ROI 密度测量 / 拖动缩放]")
    z = v.current_3d_pos[0]
    v.views[1]['plane'] = AXIAL; v.views[1]['chk_anno'].setChecked(True)
    anno = {'id': 'roi1', 'type': 'roi', 'rect': (100.0, 100.0, 60.0, 40.0)}
    v.global_annotations[z] = [anno]
    v.update_display(); app.processEvents()
    view = v.views[1]['view']
    rois = [it for it in view.scene.items() if isinstance(it, ROIGraphicsItem)]
    txts = [it for it in view.scene.items() if isinstance(it, QGraphicsTextItem)]
    check(len(rois) == 1 and len(txts) >= 1, "ROI 渲染为可编辑椭圆 + 统计文字")
    item = rois[0]
    item.setPos(200, 150); item._commit(); app.processEvents()
    check(abs(anno['rect'][0] - 200) < 1, "拖动 ROI 写回 annotation")

    class FE:
        def __init__(s, x, y): s._p = QPointF(x, y)
        def pos(s): return s._p
        def accept(s): pass
    rois = [it for it in view.scene.items() if isinstance(it, ROIGraphicsItem)]; item = rois[0]
    w0 = item.rect().width()
    item.mousePressEvent(FE(item.rect().width() - 3, item.rect().height() - 3))
    item.mouseMoveEvent(FE(item.rect().width() + 40, item.rect().height() + 30))
    item.mouseReleaseEvent(FE(0, 0)); app.processEvents()
    check(item.rect().width() > w0 and anno['rect'][2] > 60, "拖手柄缩放 ROI 并写回")
    # 命中判定避免与平移冲突
    view.current_tool = TOOL_POINTER
    view.itemAt = lambda p: item
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(150, 120), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    check(view.dragMode() == QGraphicsView.NoDrag, "指针点在 ROI 上 -> NoDrag (不平移)")
    view.itemAt = lambda p: None
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(400, 400), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    check(view.dragMode() == QGraphicsView.ScrollHandDrag, "指针点在空白 -> ScrollHandDrag (平移)")
    v.global_annotations[z] = []


def test_mpr_ruler_spacing(v, app):
    """冠状/矢状面测距的行/列间距不得交换：垂直=Z→SliceThickness，水平→PixelSpacing。"""
    print("[MPR 测距间距轴向]")
    from constants import CORONAL, SAGITTAL
    ds = v.dicom_datasets[0]
    px = v._dcm_float(ds, 'PixelSpacing', 1.0, idx=0)
    st = v._dcm_float(ds, 'SliceThickness', 1.0)
    saved_plane = v.views[1]['plane']
    for plane, nm in ((CORONAL, "冠状"), (SAGITTAL, "矢状")):
        v.views[1]['plane'] = plane
        v.update_display(); app.processEvents()
        ps = v.views[1]['view'].pixel_spacing
        # ps[0]=垂直(Z)=SliceThickness, ps[1]=水平=PixelSpacing
        check(abs(ps[0] - st) < 1e-6 and abs(ps[1] - px) < 1e-6,
              f"{nm}面测距间距正确 ps=(ST,PS)=({ps[0]:.3f},{ps[1]:.3f})")
    v.views[1]['plane'] = saved_plane
    v.update_display(); app.processEvents()


def test_mpr_aniso_aspect(v, app):
    """各向异性面按物理比例显示：Axial 变换均匀；Coronal 变换比例=ST/PS；坐标仍=体素。"""
    print("[MPR 各向异性显示比例]")
    from PySide6.QtCore import QPointF

    from constants import AXIAL, CORONAL
    ds = v.dicom_datasets[0]
    px = v._dcm_float(ds, 'PixelSpacing', 1.0, idx=0)
    st = v._dcm_float(ds, 'SliceThickness', 1.0)
    view = v.views[1]['view']; view.resize(400, 400)
    saved_plane = v.views[1]['plane']
    # Axial（各向同性）：变换均匀，不受各向异性分支影响
    v.views[1]['plane'] = AXIAL; v.update_display(); app.processEvents()
    t = view.transform()
    check(abs(t.m11() - t.m22()) < 1e-6, "横断面变换均匀（各向同性路径不变）")
    # Coronal（各向异性）：垂直/水平缩放比 = SliceThickness/PixelSpacing
    v.views[1]['plane'] = CORONAL; v.update_display(); app.processEvents()
    t = view.transform()
    check(abs(t.m22() / t.m11() - st / px) < 1e-3,
          f"冠状面显示比例修正 m22/m11={t.m22()/t.m11():.3f}≈ST/PS={st/px:.3f}")
    # 坐标不变：scene 坐标仍严格=体素索引（mapToScene 求逆），hover/测量不受影响
    got = view.get_real_coordinates(view.mapFromScene(QPointF(50, 30)))
    check(got == (50, 30), f"各向异性变换下坐标仍=体素 {got}")
    v.views[1]['plane'] = saved_plane; v.update_display(); app.processEvents()


def test_sampling_density(v):
    print("[重建采样密度]")
    import recon
    th = recon.make_theta(180, 180 * 4)
    check(len(th) == 720 and th[-1] < 180, "180° 4× 过采样 = 720 投影且覆盖不变")


def test_compare(v, app):
    print("[双序列随访对比]")
    saved_layout = v.combo_layout.currentIndex()   # 本测试会切布局验 V3，退出前必须还原，
    saved_id = id(v.dicom_datasets)                # 否则污染后续测试的可见视图集合
    vol, dsets = v._read_compare_dir(os.path.join(_ROOT, "肺癌"))
    check(vol is not None and id(v.dicom_datasets) == saved_id, "读取对比序列不污染主序列")
    v.compare_volume = np.zeros((10, 64, 64), np.float32)
    v.compare_datasets = [type('D', (), {'StudyDate': '20200115'})()]
    v.compare_mode_active = True
    v._primary_zpos = np.arange(v.volume_hu.shape[0]).astype(float)
    v._compare_zpos = np.array([40, 45, 50, 55, 60, 65, 70, 75, 80, 85.])
    v.current_3d_pos[0] = 100
    v._render_compare(); app.processEvents()
    z2_reg = int(np.argmin(np.abs(v._compare_zpos - 100)))
    z2_ratio = min(9, max(0, round(100 / 232 * 9)))
    check(z2_reg != z2_ratio and f"{z2_reg + 1}/10" in v.views[2]['title_label'].text(),
          "按解剖坐标配准 (非索引比例)")
    before = dict(v.global_annotations)
    v.handle_annotation_added({'id': 'x', 'type': 'ruler', 'p1': (1, 1), 'p2': (2, 2)})
    check(v.global_annotations == before, "对比模式下标注被守卫")
    # 形状不同（上面的 64×64 对比卷 vs 512×512 主序列）→ 定量须如实报告不可比，而非强行缩放
    check("不可比" in v.views[2]['title_label'].text() or "n/a" in v.views[2]['title_label'].text(),
          "矩阵尺寸不同 → 标题如实标注 Δ 不可比")
    # 同形序列才走真正的差值定量路径（上面那条 64×64 用例恰好绕过了它）：
    # 构造既往 = 当前 + 40 HU，则 Δ 必为 -40、MAE/RMSE 必为 40，可精确验算。
    Zp = v.volume_hu.shape[0]
    v.compare_volume = v.volume_hu + 40.0
    v.compare_datasets = [type('D', (), {'StudyDate': '20200115'})()]
    v._primary_zpos = np.arange(Zp).astype(float)
    v._compare_zpos = np.arange(Zp).astype(float)
    v.current_3d_pos[0] = Zp // 2
    v._render_compare(); app.processEvents()
    t2 = v.views[2]['title_label'].text()
    check("Δ-40" in t2 and "40" in t2, f"同形序列差值定量：既往高 40 HU → Δ-40 (得 {t2[-42:]})")
    check(v.views[3]['container'].isHidden(), "对比模式默认双窗，V3 隐藏（差值图不做无用渲染）")
    v.combo_layout.setCurrentIndex(2); app.processEvents()   # 切四窗，V3 可见
    v._render_compare(); app.processEvents()
    check(not v.views[3]['container'].isHidden() and "差值图" in v.views[3]['title_label'].text()
          or "Difference" in v.views[3]['title_label'].text(), "切四窗后 V3 渲染差值图不崩")
    # 平面内刚性配准：造一个整体平移的"既往序列"，勾选配准后差异必须显著下降。
    # 这条端到端钉住 registration 与 compare_lab 的接线方向——符号接反时差异会更大。
    import re as _re

    import scipy.ndimage as _ndi
    v.compare_volume = np.stack([_ndi.shift(sl, (12, -9), order=1, mode='nearest')
                                 for sl in v.volume_hu])
    v._primary_zpos = np.arange(Zp).astype(float)
    v._compare_zpos = np.arange(Zp).astype(float)
    v.chk_register.setChecked(False)
    v._render_compare(); app.processEvents()
    t_off = v.views[2]['title_label'].text()
    v.chk_register.setChecked(True)
    v._render_compare(); app.processEvents()
    t_on = v.views[2]['title_label'].text()

    def _mae(t):
        mm = _re.search(r'(?:绝对差|MAE) (\d+)', t)
        return int(mm.group(1)) if mm else None
    m_off, m_on = _mae(t_off), _mae(t_on)
    check(m_off is not None and m_on is not None and m_on < m_off / 3,
          f"勾选配准后 MAE 大幅下降 ({m_off} → {m_on} HU)")
    check('12' in t_on and '-9' in t_on, f"标题标出估计位移 (+12,-9) (得 ...{t_on[-70:]})")
    check('仅z轴对齐' not in t_on and 'z-aligned only' not in t_on,
          "配准生效后标题不再自称「仅 z 轴对齐」（口径随状态变化）")
    v.chk_register.setChecked(False)
    v.combo_layout.setCurrentIndex(saved_layout); app.processEvents()   # 还原布局，勿污染后续测试
    v.compare_mode_active = False; v.compare_volume = None
    v.tabs.setCurrentIndex(0); app.processEvents()


def test_cine_keyboard(v, app):
    print("[Cine 往返/调速 + 键盘翻片]")
    ev = lambda k, mod=Qt.NoModifier: QKeyEvent(QEvent.KeyPress, k, mod)
    z0 = v.slider_slice.value()
    v.keyPressEvent(ev(Qt.Key_Down))
    check(v.slider_slice.value() == z0 + 1, "↓ 键下一层")
    mx = v.slider_slice.maximum()
    v.slider_slice.setValue(mx); v._cine_dir = 1; v._cine_step()
    check(v._cine_dir == -1 and v.slider_slice.value() < mx, "Cine 到顶往返 (bounce，非回环)")
    v.toggle_cine()
    check(v.cine_timer.isActive() and v.cine_timer.interval() == 120, "Cine 默认中速播放")
    v.cb_cine_speed.setCurrentIndex(2)
    check(v.cine_timer.interval() == 50, "播放中改速度即时生效")
    v.toggle_cine()
    v.keyPressEvent(ev(Qt.Key_Z, Qt.ControlModifier))  # Ctrl+Z 不应崩
    check(True, "Ctrl+Z 撤销键路径可用")


def test_compliance(v, app):
    print("[合规：脱敏 + 免责]")
    v.anonymize = False; v._refresh_patient_info()
    real_id = v.info_labels["ID"].text()
    v.chk_anon.setChecked(True); app.processEvents()
    check(v.info_labels["ID"].text() == "ANON", "脱敏隐去患者面板身份")
    v.views[1]['plane'] = AXIAL; v.update_display(); app.processEvents()
    tl = v.views[1]['view'].overlay_lines.get('tl', [])
    check(any("ANON" in s for s in tl), "脱敏隐去四角叠加身份")
    check(v._export_tag() == "ANON", "脱敏导出文件名用匿名前缀")
    v.chk_anon.setChecked(False); app.processEvents()
    check(v.info_labels["ID"].text() == real_id, "关闭脱敏恢复真实身份")
    check("非诊断依据" in v.lbl_disclaimer.text(), "AI 面板常驻免责声明")


def test_edge_cases(v, app):
    print("[边界/崩溃防护]")
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.handle_seg_paint(1, [(200, 200)], False)
    # 换病例清撤销栈
    v._build_volume_hu()
    check(len(v._mask_undo) == 0, "换病例清空分割撤销栈")
    # 换更小病例后撤销不越界崩溃。本段会把 volume_hu 换成合成卷，
    # 用完必须还原为真实数据——否则后续依赖真实 HU 的测试会拿到全零卷而静默失真。
    saved_hu, saved_pos = v.volume_hu, list(v.current_3d_pos)
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.handle_seg_paint(1, [(200, 200)], False)
    v.volume_hu = np.zeros((50, 512, 512), np.float32)
    v.volume_mask = np.zeros((50, 512, 512), np.uint8)
    v.current_3d_pos[0] = 25
    crashed = False
    try:
        v._undo_mask_edit()
    except Exception:
        crashed = True
    check(not crashed, "换病例后撤销不越界崩溃")
    v.volume_hu = saved_hu                      # 还原真实体数据
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.current_3d_pos = saved_pos
    check(v.volume_hu.shape[1:] == (512, 512) and float(v.volume_hu.min()) < -500,
          f"退出前还原真实 HU 体数据 (shape={v.volume_hu.shape}, min={float(v.volume_hu.min()):.0f})")
    # 脱敏隐去对比既往日期（PHI）
    vv = m.MedicalViewer(data_dir=os.path.join(_ROOT, "肺癌")); app.processEvents()
    if vv.ai_thread:
        vv.ai_thread.cancel()
    vv.anonymize = True
    vv.compare_volume = np.zeros((10, 64, 64), np.float32)
    vv.compare_datasets = [type('D', (), {'StudyDate': '20200115'})()]
    vv.compare_mode_active = True
    vv._primary_zpos = np.arange(vv.volume_hu.shape[0]).astype(float)
    vv._compare_zpos = np.arange(10).astype(float)
    vv.current_3d_pos[0] = 5
    vv._render_compare(); app.processEvents()
    check("2020" not in vv.views[2]['title_label'].text(), "脱敏模式隐去对比既往检查日期")


def _write_min_dcm(path, shape, series_uid, ipp_z, inst, pid='RID_TEST', empty_numeric=False,
                   n_frames=1, truncate=False, pix=100, slope=1, intercept=-1024):
    """写一张最小合规的 CT DICOM，供混合形状加载测试使用。ipp_z=None 则不写 ImagePositionPatient。
    empty_numeric=True 时把 RescaleSlope/Intercept/PixelSpacing/SliceThickness 写成空值（None）。
    n_frames>1 写多帧 DICOM；truncate=True 写截断的 PixelData（pixel_array 解码会抛）。
    pix/slope/intercept 给定已知像素值与线性变换，供 HU 转换正确性测试断言 HU=pix*slope+intercept。"""
    import numpy as _np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    rows, cols = shape
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = pid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = CTImageStorage
    ds.Modality = 'CT'
    ds.InstanceNumber = inst
    if ipp_z is not None:
        ds.ImagePositionPatient = [0.0, 0.0, float(ipp_z)]
    ds.PixelSpacing = None if empty_numeric else [1.0, 1.0]
    ds.SliceThickness = None if empty_numeric else 1.0
    ds.RescaleSlope = None if empty_numeric else slope
    ds.RescaleIntercept = None if empty_numeric else intercept
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    if n_frames > 1:
        ds.NumberOfFrames = n_frames
        ds.PixelData = _np.full((n_frames, rows, cols), pix, dtype=_np.int16).tobytes()
    else:
        full = _np.full((rows, cols), pix, dtype=_np.int16).tobytes()
        ds.PixelData = full[:len(full) // 3] if truncate else full
    ds.save_as(path, write_like_original=False)


def test_mixed_shape_dicom(app):
    """加载同序列/无 SeriesUID 但切片形状不一致的目录，不得崩溃（形状一致性过滤）。"""
    print("[混合形状 DICOM 加载防护]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    v2 = m.MedicalViewer(); app.processEvents()
    if v2.ai_thread:
        v2.ai_thread.cancel()
    sid = generate_uid()
    cases = [
        ("同序列混合形状", [((512, 512), sid), ((512, 512), sid), ((512, 512), sid), ((256, 256), sid)], (512, 512), 3),
        ("SeriesUID全空混合形状", [((256, 256), ''), ((256, 256), ''), ((256, 256), ''), ((64, 64), '')], (256, 256), 3),
    ]
    for label, spec, exp_yx, exp_n in cases:
        d = tempfile.mkdtemp()
        try:
            for i, (shp, s) in enumerate(spec):
                _write_min_dcm(os.path.join(d, f"s{i:03d}.dcm"), shp, s, ipp_z=i, inst=i)
            crashed = False
            try:
                v2._read_dicom_dir(d)
                v2._build_volume_hu()
            except Exception:
                crashed = True
            ok = (not crashed) and v2.volume_hu.shape[1:] == exp_yx and v2.volume_hu.shape[0] == exp_n
            check(ok, f"混合形状不崩且保留多数尺寸：{label} -> {v2.volume_hu.shape}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
            if v2.ai_thread:
                v2.ai_thread.cancel()


def test_legend_consistency(v, app):
    """图例与蒙版叠加显隐保持一致：关 Anno / 无器官切片 → 图例清空；有器官且开 Anno → 显示。"""
    print("[图例一致性]")
    from constants import AXIAL
    saved_mask = v.volume_mask
    saved_pos = list(v.current_3d_pos)
    Z, Y, X = v.volume_hu.shape
    z_org, z_empty = Z // 2, 0                       # 按当前体积形状取切片，避免依赖固定层数
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.volume_mask[z_org, Y // 4:Y // 4 + 6, X // 4:X // 4 + 6] = 5
    v.views[1]['plane'] = AXIAL
    v.views[1]['chk_anno'].setChecked(True)
    v.current_3d_pos[0] = z_org
    v.update_display(); app.processEvents()
    on = v.lbl_ai_legend.text()
    v.current_3d_pos[0] = z_empty                    # 无器官切片
    v.update_display(); app.processEvents()
    empty_slice = v.lbl_ai_legend.text()
    v.current_3d_pos[0] = z_org                       # 回到有器官层
    v.update_display(); app.processEvents()
    v.views[1]['chk_anno'].setChecked(False)        # 关闭叠加
    v.update_display(); app.processEvents()
    anno_off = v.lbl_ai_legend.text()
    check("toggle:5" in on, "有器官且开 Anno 时图例显示该器官")
    check(empty_slice == "", "无器官切片图例清空")
    check(anno_off == "", "关闭 Anno 后图例随叠加一并清空")
    v.views[1]['chk_anno'].setChecked(True)
    v.volume_mask = saved_mask
    v.current_3d_pos = saved_pos
    v.update_display(); app.processEvents()


def test_recon_finite(app):
    """重建算法对含 NaN/Inf 的病态弦图仍须输出有限可显示图（对齐 DFR 的 nan_to_num 约定）。"""
    print("[重建数值稳定性]")
    import recon as R
    out = R._finite_clip(np.array([np.nan, np.inf, -np.inf, 0.5], np.float32), 2)
    check(np.all(np.isfinite(out)) and out.shape == (2, 2), "_finite_clip 中和 NaN/Inf 为有限图")
    A = np.eye(4, dtype=np.float32)                       # 极简 4×4 系统矩阵（免 multiprocessing 建阵）
    p = np.array([1.0, np.nan, 0.5, np.inf], np.float32)  # 弦图混入非有限值
    rec_dmr, _ = R.compute_dmr(A, p, 2)
    rec_art, _ = R.compute_art(A, p, 2, 20)
    rec_sirt, _ = R.compute_sirt(A, p, 2, 20)
    check(np.all(np.isfinite(rec_dmr)), "DMR 对含 NaN/Inf 弦图输出有限")
    check(np.all(np.isfinite(rec_art)), "ART 对含 NaN/Inf 弦图输出有限")
    check(np.all(np.isfinite(rec_sirt)), "SIRT 对含 NaN/Inf 弦图输出有限")
    # 正常有限弦图：nan_to_num 为恒等，重建结果不受影响
    p_ok = np.array([1.0, 0.0, 0.5, 0.2], np.float32)
    rec_ok, _ = R.compute_dmr(A, p_ok, 2)
    check(np.allclose(rec_ok.ravel(), np.clip(p_ok, 0, 1)), "正常弦图 DMR 结果不受 finite 守卫影响")
    # 解析法（FBP/DFR）此前没有 finite 守卫，与矩阵/迭代法防御不一致——补齐后一并钉住。
    # 弦图混入 NaN/Inf 时 iradon 会把污染扩散到整幅图，NaN 在显示中静默变黑并污染 RMSE。
    th32 = R.make_theta(180, 32)
    sino_ok = R.compute_sinogram(np.random.RandomState(0).rand(16, 16).astype(np.float32), th32)
    for nm, bad_v in (("NaN", np.nan), ("+Inf", np.inf), ("-Inf", -np.inf)):
        bp, fbp = R.compute_fbp(np.full_like(sino_ok, bad_v), th32, 'ramp')
        check(bool(np.all(np.isfinite(bp))) and bool(np.all(np.isfinite(fbp))),
              f"FBP/BP 对全 {nm} 弦图输出有限")
    f2, f1, dfr = R.compute_dfr(np.full_like(sino_ok, np.inf), th32)
    check(bool(np.all(np.isfinite(f2))) and bool(np.all(np.isfinite(f1))),
          "DFR 对含 Inf 弦图的频域输出有限（入口即中和，不等 FFT 后再救）")
    # SIRT 的归一化系数曾用 np.where(cond, 1/x, 0)——np.where 不短路，1/x 仍会对 0 求值。
    # 结果虽正确但每次调用刷除零警告；改用 np.divide(where=) 后应无警告且结果不变。
    import warnings as _w
    _, _, A32 = R._matrix_worker((0, 256, 16, th32))
    p32 = sino_ok.ravel().astype(np.float32)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        rec_s, _ = R.compute_sirt(A32, p32, 16, 3)
    divzero = [x for x in caught if issubclass(x.category, RuntimeWarning) and 'divide by zero' in str(x.message)]
    check(not divzero and bool(np.all(np.isfinite(rec_s))),
          f"SIRT 无除零警告且结果有限 (捕获 {len(divzero)} 条除零警告)")


def _recon_disk(n, radius_frac=0.3, val=1.0):
    """半径 = radius_frac·n 的均匀圆盘，圆心取几何中心 (n-1)/2（与 radon 的探测器中心对齐）。"""
    c = (n - 1) / 2.0
    Y, X = np.ogrid[:n, :n]
    return (((Y - c) ** 2 + (X - c) ** 2 <= (radius_frac * n) ** 2) * val).astype(np.float32)


def _recon_phantom(n):
    """Shepp-Logan 缩到 n×n 并施加圆形掩码——与 experiments/recon_study.py:get_phantom 同一构造。"""
    import scipy.ndimage as ndimage
    from skimage.data import shepp_logan_phantom

    import recon as R
    p = shepp_logan_phantom().astype(np.float32)
    p = ndimage.zoom(p, (n / p.shape[0], n / p.shape[1]))
    return (np.clip(p, 0.0, 1.0) * R._circle_mask(n)).astype(np.float32)


def _roi_corr(a, b, n):
    """圆形 ROI 内的零均值单位方差相关系数——与幅度/偏置无关，只看结构是否对得上。
    BP 未归一化（量级差 100×），只能用相关系数与 FBP 公平比较。"""
    import recon as R
    m = R._circle_mask(n) > 0
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.mean(x * y))


def test_recon_numerics():
    """重建算法数值正确性——解析可知的模体 + 已知性质，在 "finite" 之外真正验算法对不对。

    断言全部锚在解析可知或理论恒等的性质上（质量守恒、解析弦长、算子线性、中心切片定理、
    A·x≡Radon、满秩精确还原、迭代单调收敛），而非「上次跑出来是多少」的快照值——后者会把
    bug 一起固化。n=64（解析/FBP/DFR 段）与 n=16（矩阵/迭代段），无真实数据、无 ONNX、
    无 multiprocessing、不写 .matrix_cache。阈值按实测值留足余量（注释标出实测数）。
    """
    print("[重建算法数值正确性]")
    from skimage.metrics import structural_similarity

    import recon as R
    n = 64
    # ---- 1) compute_sinogram：解析可知的正向投影性质 ----
    disk = _recon_disk(n, 0.3)
    theta = R.make_theta(180, 60)
    sino = R.compute_sinogram(disk, theta)
    check(sino.shape == (n, len(theta)), f"弦图 shape=(探测器{n}, 角度{len(theta)}) (得 {sino.shape})")
    # 每角度投影积分 = 图像总质量（线积分守恒，理论恒等）。实测偏差 3.1e-4，阈值 5e-3 留 16× 余量
    col_sums = sino.sum(axis=0)
    dev = float(np.abs(col_sums - disk.sum()).max() / disk.sum())
    check(dev < 5e-3, f"总质量守恒：每角度投影积分 = 图像总和 (最大相对偏差 {dev:.2e})")
    # 均匀圆盘 0° 投影 = 解析弦长 2√(R²-t²)。误差集中在圆盘边界像素化(partial volume)处：
    # 实测 max 0.853 / 峰值 38.4（相对 2.2%），阈值取峰值 5% = 1.92，留 2.25× 余量
    t = np.arange(n) - (n - 1) / 2.0
    chord = 2 * np.sqrt(np.maximum((0.3 * n) ** 2 - t ** 2, 0))
    err = float(np.abs(sino[:, 0] - chord).max())
    check(err < 0.05 * chord.max(), f"均匀圆盘 0° 投影剖面 = 解析弦长 2√(R²-t²) (最大绝对误差 {err:.3f}, 峰值 {chord.max():.1f})")
    # Radon 是线性算子：s(a+b)=s(a)+s(b)。实测 9.5e-06（float32 舍入），阈值 1e-4 留 10× 余量
    a, b = _recon_disk(n, 0.2), _recon_disk(n, 0.35, 0.5)
    lin = float(np.abs(R.compute_sinogram(a + b, theta) - R.compute_sinogram(a, theta) - R.compute_sinogram(b, theta)).max())
    check(lin < 1e-4, f"Radon 变换线性：s(a+b)=s(a)+s(b) (最大偏差 {lin:.2e})")
    # ---- 2) compute_fbp：与真值的 RMSE / SSIM 达阈值 ----
    gt = _recon_phantom(n)
    theta = R.make_theta(180, 180)
    sino = R.compute_sinogram(gt, theta)
    bp_ret, fbp = R.compute_fbp(sino, theta, "ramp")
    m = R._circle_mask(n) > 0
    # 实测 RMSE=0.0869 / SSIM=0.8541。误差底噪来自 n=64 下采样的锐边 + ramp 振铃，对投影数极不敏感
    # （60/90/180 投影下 RMSE=0.0882/0.0872/0.0869，SSIM=0.799/0.845/0.854），故阈值稳健。
    rmse_fbp = float(np.sqrt(np.mean((fbp - gt)[m] ** 2)))
    ssim_fbp = float(structural_similarity(gt, fbp.astype(np.float32), data_range=1.0))
    check(rmse_fbp < 0.12, f"FBP(ramp) 重建 Shepp-Logan：ROI RMSE={rmse_fbp:.4f} < 0.12")
    check(ssim_fbp > 0.75, f"FBP(ramp) 重建 SSIM={ssim_fbp:.4f} > 0.75")
    _, fbp_alias = R.compute_fbp(sino, theta, "Ram-Lak")
    check(np.array_equal(fbp, fbp_alias), "滤波器名 'Ram-Lak' 映射为 'ramp'（UI 名称契约）")
    # ---- 3) compute_bp：必须明显差于 FBP —— 证明滤波真的起作用 ----
    bp = R.compute_bp(sino, theta)
    check(np.array_equal(bp, bp_ret), "compute_fbp 返回的未滤波图 == compute_bp")
    # BP 未归一化且低频过度叠加：用与幅度无关的相关系数比较才公平。实测 BP=0.536 / FBP=0.936
    # （跨 60/90/180 投影极稳定）；阈值 0.7 两侧各留 ~30% 余量
    c_fbp, c_bp = _roi_corr(gt, fbp, n), _roi_corr(gt, bp, n)
    check(c_bp < 0.7 < c_fbp, f"纯反投影模糊：与真值相关 BP={c_bp:.3f} 显著低于 FBP={c_fbp:.3f}")
    rmse_bp = float(np.sqrt(np.mean((bp - gt)[m] ** 2)))
    check(rmse_bp > 10 * rmse_fbp, f"BP RMSE={rmse_bp:.2f} 远大于 FBP RMSE={rmse_fbp:.4f}（低频过度叠加）")
    # ---- 4) compute_dfr：中心切片定理的直流项 + 方位契约 ----
    freq2d, fft1d, dfr = R.compute_dfr(sino, theta)
    check(freq2d.shape == (n, n) and np.all(np.isfinite(freq2d)), "DFR 频域矩阵 n×n 且全有限")
    check(fft1d.shape == sino.shape and float(fft1d.min()) >= 0.0, "DFR 一维谱 log1p 幅度非负、与弦图同形")
    # 中心切片定理的直接推论：二维频域原点 = 图像总质量。实测 |1-比值| ≈ 2e-7（n=32/48/64/96 皆然），
    # 阈值 1e-3 留 4 个数量级余量——任何归一化/零频错位都会让它整数倍地跑掉。
    dc_ratio = float(abs(freq2d[n // 2, n // 2]) / gt.sum())
    check(abs(dc_ratio - 1.0) < 1e-3, f"中心切片定理：频域直流项 |F(0,0)| = 图像总质量 (比值 {dc_ratio:.6f})")
    # compute_dfr 已在内部把朝向校正为与输入同方位，直接 abs 即可（不再有"须自行 rot90"契约）。
    # 实测 n=64 治本后与真值相关 0.906（旧的 rot90 仅 0.689、完全未处理仅 0.012），阈值 0.85 留余量。
    c_dfr = _roi_corr(gt, np.abs(dfr).astype(np.float32), n)
    check(c_dfr > 0.85, f"DFR 输出直接与真值对齐（内部已校正朝向，无需调用方 rot90）：相关={c_dfr:.3f} > 0.85")
    # 偏心脉冲的重建峰值须精确落在输入位置——偶数 n 曾因 np.rot90 绕几何中心 (n-1)/2 而错位 1 像素，已修
    imp = np.zeros((n, n), np.float32); imp[n // 2 - 8, n // 2 + 6] = 1.0; imp *= R._circle_mask(n)
    _, _, dfr_imp = R.compute_dfr(R.compute_sinogram(imp, theta), theta)
    pk = tuple(int(v) for v in np.unravel_index(int(np.argmax(np.abs(dfr_imp))), (n, n)))
    check(pk == (n // 2 - 8, n // 2 + 6), f"偏心脉冲重建峰值精确对齐输入 {(n // 2 - 8, n // 2 + 6)} (得 {pk}；偶数 n 的 1 像素错位已修)")
    # ---- 5) 系统矩阵 + DMR：满秩无噪系统应精确还原 ----
    # 直接调 _matrix_worker 在进程内建阵（n=16 实测 0.08 s）：避开 multiprocessing 与 .matrix_cache 写盘副作用
    n_s = 16
    theta_s = R.make_theta(180, 32)
    _, _, A = R._matrix_worker((0, n_s * n_s, n_s, theta_s))
    img = np.zeros((n_s, n_s), np.float32)
    img[4:9, 4:9] = 1.0; img[9:12, 7:12] = 0.5; img[5:7, 10:13] = 0.75
    img *= R._circle_mask(n_s)
    sino_s = R.compute_sinogram(img, theta_s)
    check(A.shape == (sino_s.size, n_s * n_s), f"系统矩阵 shape=(射线{sino_s.size}, 像素{n_s * n_s}) (得 {A.shape})")
    # A 就是 Radon 的矩阵表示：A·x 必须等于 compute_sinogram(x)。实测 9.5e-07（float32 精度），阈值 1e-4
    fwd = float(np.abs(A @ img.ravel() - sino_s.ravel()).max())
    check(fwd < 1e-4, f"系统矩阵与 Radon 一致：max|A·x - compute_sinogram(x)| = {fwd:.2e}")
    rank = int(np.linalg.matrix_rank(A))
    check(rank == n_s * n_s, f"A 列满秩 rank={rank} = {n_s * n_s}（DMR 有唯一最小二乘解）")
    # 满秩 + 无噪 ⇒ lstsq 应还原到浮点精度。实测 max|err|=1.0e-06，阈值 1e-4 留 100× 余量
    p_vec = sino_s.ravel().astype(np.float32)
    dmr, ms = R.compute_dmr(A, p_vec, n_s)
    err_dmr = float(np.abs(dmr - img).max())
    check(err_dmr < 1e-4, f"DMR 满秩无噪系统精确还原：max|err| = {err_dmr:.2e}")
    check(ms >= 0.0, f"DMR 返回耗时 {ms:.1f} ms")
    key = (n_s, len(theta_s), round(float(theta_s[0]), 4), round(float(theta_s[-1]), 4))
    A_c, key_c = R.build_system_matrix(n_s, theta_s, A, key)
    check(A_c is A and key_c == key, "build_system_matrix 命中内存缓存直接复用（不重建、不起子进程）")
    # ---- 6) ART / SIRT：迭代应收敛，误差随迭代下降 ----
    stop = {"n": 0}
    def _cancel(): stop["n"] += 1; return True          # 定义在循环外：避免闭包捕获循环变量（ruff B023）
    for name, fn in (("ART", R.compute_art), ("SIRT", R.compute_sirt)):
        rmses, resids = [], []
        for it in (1, 5, 20):
            rec, _ = fn(A, p_vec, n_s, it)
            rmses.append(float(np.sqrt(np.mean((rec - img) ** 2))))
            resids.append(float(np.linalg.norm(A @ rec.ravel() - p_vec)))
        # 实测 ART RMSE 0.2126>0.0542>0.0167、残差 30.56>4.29>0.83；
        #      SIRT RMSE 0.2582>0.1618>0.0885、残差 35.22>15.25>5.29——严格单调，余量极大
        check(rmses[0] > rmses[1] > rmses[2], f"{name} RMSE 随迭代单调下降: {rmses[0]:.4f} > {rmses[1]:.4f} > {rmses[2]:.4f}")
        check(resids[0] > resids[1] > resids[2], f"{name} 弦图残差 ‖A·x-p‖ 单调下降: {resids[0]:.2f} > {resids[1]:.2f} > {resids[2]:.2f}")
        check(rmses[2] < 0.5 * rmses[0], f"{name} 20 轮后 RMSE={rmses[2]:.4f} 不足 1 轮的一半（确在收敛）")
        stop["n"] = 0
        rec_c, _ = fn(A, p_vec, n_s, 100, cancel_check=_cancel)
        check(stop["n"] == 1 and float(np.abs(rec_c).max()) == 0.0, f"{name} cancel_check 首轮即停，返回全零初值（不跑满 100 轮）")
        seen = []
        fn(A, p_vec, n_s, 3, progress_cb=seen.append)
        check(seen == [0, 1, 2], f"{name} progress_cb 每轮回调一次 (得 {seen})")
    # Kaczmarz 逐射线更新每轮信息利用率高于 SIRT 的同步更新——教科书性质。实测 0.0167 vs 0.0885（5×）
    art20, _ = R.compute_art(A, p_vec, n_s, 20)
    sirt20, _ = R.compute_sirt(A, p_vec, n_s, 20)
    r_art = float(np.sqrt(np.mean((art20 - img) ** 2))); r_sirt = float(np.sqrt(np.mean((sirt20 - img) ** 2)))
    check(r_art < r_sirt, f"同迭代数下 ART 收敛快于 SIRT: RMSE {r_art:.4f} < {r_sirt:.4f}")
    check(float(art20.min()) >= 0.0 and float(sirt20.min()) >= 0.0, "ART/SIRT 非负约束生效")


def test_recon_pipeline_helpers():
    """重建预处理/上采样纯函数直接单测——合成数组，无 Qt / 真实数据 / 系统矩阵。"""
    print("[重建预处理/上采样纯函数]")
    import recon as R
    check(len(R.make_theta(180)) == 180 and float(R.make_theta(180)[-1]) == 179.0, "make_theta 省略 n_proj 时每度一个投影（向后兼容）")
    big = np.zeros((100, 80), np.float32); big[20:60, 20:60] = 1.0; big[0, 0] = 1.0  # 角落故意置 1，验掩码
    img_s, sino, theta = R.prepare_small_image(big, 32, 180, 90)
    check(img_s.shape == (32, 32) and sino.shape == (32, 90) and len(theta) == 90, f"prepare_small_image 缩放到 32² 并出 (32,90) 弦图 (得 {img_s.shape}, {sino.shape})")
    check(float(img_s.min()) >= 0.0 and float(img_s.max()) <= 1.0, f"缩放后 clip 在 [0,1]（防样条插值 Gibbs 过冲，得 [{img_s.min():.3f}, {img_s.max():.3f}]）")
    outside = R._circle_mask(32) == 0
    check(bool(np.all(img_s[outside] == 0)) and float(img_s[0, 0]) == 0.0, "圆形掩码生效：圆外恒为 0（角落原有值被掩掉，避免误差图虚假大误差）")
    check(bool(np.allclose(sino, R.compute_sinogram(img_s, theta))), "返回的弦图 == compute_sinogram(返回的小图)（三元组自洽）")
    up = R.upscale_recon(img_s, 32)
    check(up.shape == (256, 256), f"upscale_recon 32² → 256²（scale=8）(得 {up.shape})")
    check(bool(np.all(up[0:8, 0:8] == img_s[0, 0])) and bool(np.all(up[8:16, 0:8] == img_s[1, 0])), "kron 严格像素块复制（非插值，保留像素块感）")
    same = np.zeros((256, 256), np.float32)
    check(R.upscale_recon(same, 256) is same, "n=256 时 scale=1，原数组原样返回（不复制）")


def test_malformed_annotations(v, app):
    """畸形/旧版本标注：渲染时逐条兜底不崩；加载 JSON 时过滤掉不合规条目。"""
    print("[畸形标注容错]")
    import json
    z = v.current_3d_pos[0]
    saved = v.global_annotations
    # 1) 渲染层兜底：各种畸形直接塞进去刷新，不得崩
    bad_sets = [
        [{'id': 'a', 'p1': (1, 1), 'p2': (2, 2)}],            # 缺 type
        [{'id': 'b', 'type': 'path', 'points': []}],          # 空 points
        [{'id': 'c', 'type': 'path'}],                        # 缺 points
        [{'id': 'd', 'type': 'roi', 'rect': [1, 2]}],         # rect 长度错
        [{'id': 'e', 'type': 'ruler', 'p1': (1, 1)}],         # 缺 p2
        "not-a-list",                                         # 顶层非 list
        ["xyz", 123],                                         # 元素非 dict
    ]
    crashed = False
    for annos in bad_sets:
        v.global_annotations = {z: annos, 'all': []}
        try:
            v.update_display(); app.processEvents()
        except Exception:
            crashed = True
            break
    check(not crashed, "渲染畸形标注逐条兜底不崩")

    # 2) 加载层过滤：真实 JSON 落盘 -> _load_annotations_json 只留合规条目
    ED = os.path.join(_ROOT, "Exported_Lesions"); os.makedirs(ED, exist_ok=True)
    pid = "ANNOFILTER_TEST"
    fp = os.path.join(ED, f"{pid}_annotations.json")
    data = {"all": [
        {'id': 'g1', 'type': 'ruler', 'p1': [1, 1], 'p2': [9, 9]},
        {'id': 'b1', 'type': 'ruler', 'p1': [1, 1]},
        {'id': 'g2', 'type': 'roi', 'rect': [3, 3, 10, 10]},
        {'id': 'b2', 'type': 'roi', 'rect': [1, 2]},
        {'id': 'b3', 'type': 'path', 'points': []},
        "not-a-dict",
    ], "7": "not-a-list"}
    try:
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        v.global_annotations = {'all': []}
        v._load_annotations_json(pid)
        ids = sorted(a.get('id') for a in v.global_annotations.get('all', []))
        ok = ids == ['g1', 'g2'] and v.global_annotations.get(7) == []
        check(ok, f"加载期过滤畸形标注 -> 保留 {ids}")
    finally:
        if os.path.exists(fp):
            os.remove(fp)
        v.global_annotations = saved


def test_close_cancels_ai(app):
    """关窗须取消仍在运行的后台 AI 推理并停止 Cine，避免内存滞留与回调到已拆除窗口。"""
    print("[关窗收尾]")
    from PySide6.QtGui import QCloseEvent
    vc = m.MedicalViewer(); app.processEvents()
    if vc.ai_thread:
        vc.ai_thread.cancel()

    class _Stub:
        def __init__(self): self.cancelled = False
        def cancel(self): self.cancelled = True
        def isRunning(self): return not self.cancelled
    stub = _Stub()
    vc.ai_thread = stub
    # 同时开着 Cine，验证一并停止
    vc.cine_timer.start(100)
    vc.closeEvent(QCloseEvent())
    app.processEvents()
    check(stub.cancelled, "关窗取消后台 AI 推理")
    check(not vc.cine_timer.isActive(), "关窗停止 Cine 定时器")


def test_malformed_pixels(app):
    """多帧 DICOM 展开为切片；坏片跳过不带崩整卷；全坏则优雅中止并恢复原序列（状态一致）。"""
    print("[多帧 / 坏片 / 全坏防护]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    vm = m.MedicalViewer(); app.processEvents()
    if vm.ai_thread:
        vm.ai_thread.cancel()

    # 1) 多帧单文件 -> 展开为 N 层
    d1 = tempfile.mkdtemp()
    try:
        _write_min_dcm(os.path.join(d1, "mf.dcm"), (16, 16), generate_uid(), ipp_z=0, inst=1, n_frames=5)
        crashed = False
        try:
            vm.load_data(d1); app.processEvents()
            if vm.ai_thread:
                vm.ai_thread.cancel()
        except Exception:
            crashed = True
        check(not crashed and vm.volume_hu.ndim == 3 and vm.volume_hu.shape[0] == 5,
              f"多帧 DICOM 展开为 5 层 -> {None if crashed else vm.volume_hu.shape}")
    finally:
        shutil.rmtree(d1, ignore_errors=True)

    # 2) 部分坏片被跳过，好片正常加载
    d2 = tempfile.mkdtemp()
    try:
        sid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(d2, f"g{i}.dcm"), (16, 16), sid, ipp_z=i, inst=i)
        _write_min_dcm(os.path.join(d2, "bad.dcm"), (16, 16), sid, ipp_z=9, inst=9, truncate=True)
        crashed = False
        try:
            vm.load_data(d2); app.processEvents()
            if vm.ai_thread:
                vm.ai_thread.cancel()
        except Exception:
            crashed = True
        ok = (not crashed and vm.volume_hu.shape[0] == 3
              and len(vm.dicom_datasets) == vm.volume_hu.shape[0])
        check(ok, f"坏片跳过、好片加载且状态一致 -> {None if crashed else vm.volume_hu.shape}")
    finally:
        shutil.rmtree(d2, ignore_errors=True)

    # 3) 全坏目录：优雅中止，保留上一次成功加载的序列，且 datasets 与 volume 一致
    prev_shape = vm.volume_hu.shape
    d3 = tempfile.mkdtemp()
    try:
        _write_min_dcm(os.path.join(d3, "b.dcm"), (16, 16), generate_uid(), ipp_z=0, inst=1, truncate=True)
        crashed = False
        try:
            vm.load_data(d3); app.processEvents()
            if vm.ai_thread:
                vm.ai_thread.cancel()
        except Exception:
            crashed = True
        consistent = len(vm.dicom_datasets) == vm.volume_hu.shape[0]
        # 试着导航，验证不越界崩
        nav_ok = True
        try:
            vm.on_slice_changed(vm.slider_slice.maximum()); app.processEvents()
        except Exception:
            nav_ok = False
        check(not crashed and vm.volume_hu.shape == prev_shape and consistent and nav_ok,
              "全坏目录优雅中止并恢复原序列（状态一致、可导航）")
    finally:
        shutil.rmtree(d3, ignore_errors=True)
        if vm.ai_thread:
            vm.ai_thread.cancel()


def test_empty_dicom_tags(app):
    """RescaleSlope/Intercept/PixelSpacing/SliceThickness 存在但为空(None)时，
    加载/定量不得因 float(None) 崩溃。"""
    print("[空数值标签 DICOM 防护]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    ve = m.MedicalViewer(); app.processEvents()
    if ve.ai_thread:
        ve.ai_thread.cancel()
    sid = generate_uid()
    d = tempfile.mkdtemp()
    try:
        for i in range(3):
            _write_min_dcm(os.path.join(d, f"e{i}.dcm"), (16, 16), sid, ipp_z=i, inst=i, empty_numeric=True)
        crashed = False
        try:
            ve.load_data(d); app.processEvents()
            if ve.ai_thread:
                ve.ai_thread.cancel()
            ve.volume_mask = np.ones(ve.volume_hu.shape, np.uint8)
            ve._compute_organ_stats()   # 用到 PixelSpacing/SliceThickness
        except Exception as ex:
            crashed = True
            print("   ", type(ex).__name__, ex)
        check(not crashed and ve.volume_hu is not None, "空数值标签 DICOM 可正常加载并定量")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if ve.ai_thread:
            ve.ai_thread.cancel()


def test_export_path_safety(app):
    """PatientID 含 '/' 或 '..' 时：不得路径穿越写到导出目录之外；净化后存取仍往返一致。"""
    print("[导出文件名路径安全]")
    import glob
    ED = os.path.join(_ROOT, "Exported_Lesions")
    # 净化器单元：普通 ID 不变（不破坏既有文件），危险字符被中和
    su = m.MedicalViewer._safe_name
    check(su("12345") == "12345" and su("RIDER-1234") == "RIDER-1234", "普通 PatientID 不被改动")
    check("/" not in su("A/B") and su("..") == "Unknown" and su("") == "Unknown", "斜杠/纯点/空被中和")

    vp = m.MedicalViewer(); app.processEvents()
    if vp.ai_thread:
        vp.ai_thread.cancel()

    class _DS:
        # SeriesInstanceUID 在真实 DICOM 中是 Type 1 必填；蒙版缓存据它校验序列身份
        # （防止把同患者另一序列的蒙版张冠李戴），故此桩必须带上才具代表性。
        def __init__(self, pid, uid="1.2.826.0.1.3680043.2.1125.1.PATHSAFE"):
            self.PatientID = pid; self.PatientName = pid; self.SeriesInstanceUID = uid

    made = []
    try:
        # 1) 路径穿越封堵
        esc = os.path.abspath(os.path.join(ED, "..", "PWNED_annotations.json"))
        before = os.path.exists(esc)
        vp.dicom_datasets = [_DS("../PWNED")]
        vp.global_annotations = {'all': [{'id': 'x', 'type': 'ruler', 'p1': (1, 1), 'p2': (2, 2)}]}
        vp.volume_mask = np.ones((3, 8, 8), np.uint8)
        vp.save_project()
        made += glob.glob(os.path.join(ED, "_PWNED_*"))
        check(not (os.path.exists(esc) and not before), "路径穿越被封堵（未写到 Exported_Lesions 之外）")

        # 2) 斜杠 PatientID 存取往返一致
        vp.dicom_datasets = [_DS("PID/WITH/SLASH")]
        vp.volume_hu = np.zeros((3, 8, 8), np.float32)
        vp.global_annotations = {'all': [{'id': 'rt', 'type': 'ruler', 'p1': (1, 1), 'p2': (5, 5)}]}
        vp.volume_mask = np.ones((3, 8, 8), np.uint8) * 7
        vp.save_project()
        made += glob.glob(os.path.join(ED, "PID_WITH_SLASH_*"))
        vp.global_annotations = {'all': []}
        vp._load_annotations_json("PID/WITH/SLASH")
        anno_ok = vp.global_annotations.get('all') and vp.global_annotations['all'][0]['id'] == 'rt'
        mask_ok = vp._load_saved_mask("PID/WITH/SLASH") and int(vp.volume_mask.max()) == 7
        check(bool(anno_ok) and bool(mask_ok), "斜杠 PatientID 净化后存取往返一致")
    finally:
        for f in made:
            try:
                os.remove(f)
            except OSError:
                pass
        for f in glob.glob(os.path.abspath(os.path.join(ED, "..", "PWNED_*"))):
            try:
                os.remove(f)
            except OSError:
                pass


def test_dicom_sort_consistency(app):
    """部分切片缺 ImagePositionPatient 时，排序键须序列级统一，不得 z/InstanceNumber 混排。"""
    print("[DICOM 排序键一致性]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    vs = m.MedicalViewer(); app.processEvents()
    if vs.ai_thread:
        vs.ai_thread.cancel()
    sid = generate_uid()
    # 前两层有 IPP(z=10,5)，后两层缺 IPP(instance=3,4)。混排会把 3,4 插到 z=5,10 之前；
    # 序列级回退应整列按 InstanceNumber -> 顺序为 inst 1,2,3,4。
    spec = [(10.0, 1), (5.0, 2), (None, 3), (None, 4)]
    d = tempfile.mkdtemp()
    try:
        for ipp, inst in spec:
            _write_min_dcm(os.path.join(d, f"i{inst}.dcm"), (64, 64), sid, ipp_z=ipp, inst=inst)
        vs._read_dicom_dir(d)
        order = [int(getattr(x, 'InstanceNumber', -1)) for x in vs.dicom_datasets]
        check(order == [1, 2, 3, 4], f"缺位置信息时整列按 InstanceNumber 排序 -> {order}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if vs.ai_thread:
            vs.ai_thread.cancel()


def test_i18n_persistent(app):
    """切换到英文后，所有常驻控件不得残留中文（语言按钮与瞬态视图标题除外）。
    用独立 viewer，避免继承其它用例故意制造的越界状态。"""
    print("[i18n 持久控件完整性]")
    import re

    from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QLabel, QPushButton
    CJK = re.compile(r'[一-鿿]')
    vi = m.MedicalViewer(); app.processEvents()
    if vi.ai_thread:
        vi.ai_thread.cancel()
    # 排除：语言按钮（英文态故意显示"中"表示可切回）+ 四个视图标题（瞬态/受模式渲染控制）
    exclude = {vi.btn_lang}
    for vid in vi.views:
        tl = vi.views[vid].get('title_label')
        if tl is not None:
            exclude.add(tl)
    vi.is_english = True
    vi.update_language(); app.processEvents()
    residue = []
    for w in vi.findChildren(QGroupBox):
        if w not in exclude and CJK.search(w.title()):
            residue.append(f"GroupBox:{w.title()!r}")
    for cls in (QLabel, QPushButton, QCheckBox):
        for w in vi.findChildren(cls):
            if w in exclude:
                continue
            t = w.text()
            if t.strip() and CJK.search(t):
                residue.append(f"{cls.__name__}:{t!r}")
    for cb in vi.findChildren(QComboBox):
        for i in range(cb.count()):
            if CJK.search(cb.itemText(i)):
                residue.append(f"Combo:{cb.itemText(i)!r}")
    check(not residue, "英文模式无中文残留常驻控件" + (f" — 残留: {residue}" if residue else ""))


def test_quantify():
    """器官定量纯函数直接单测——用合成数组，不构造 MedicalViewer，证明逻辑已解耦。"""
    print("[器官定量纯函数 quantify.compute_organ_stats]")
    import quantify
    vol = np.zeros((4, 4, 4), np.float32)
    mask = np.zeros((4, 4, 4), np.uint8)
    for z, y, x in [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0)]:  # 器官2：5 体素 HU=50
        mask[z, y, x] = 2; vol[z, y, x] = 50.0
    for z, y, x in [(1, 0, 0), (1, 0, 1), (1, 0, 2)]:                        # 器官5：3 体素 HU=100
        mask[z, y, x] = 5; vol[z, y, x] = 100.0
    spacing = (2.0, 2.0, 3.0)            # 单体素 vox_ml = 2*2*3/1000 = 0.012 mL
    names = {2: ("肾", "Kidney")}        # 5 号故意不登记，测回退命名
    rows = quantify.compute_organ_stats(vol, mask, spacing, names)
    check(len(rows) == 2, f"检出 2 个器官 (得 {len(rows)})")
    check(rows[0]['id'] == 2 and rows[1]['id'] == 5, "按体积降序（器官2 5体素在前）")
    r2, r5 = rows[0], rows[1]
    check(r2['voxels'] == 5 and abs(r2['volume_ml'] - 5 * 0.012) < 1e-9,
          f"器官2 体积 = 5×0.012 mL (得 {r2['volume_ml']:.4f})")
    check(abs(r2['mean_hu'] - 50.0) < 1e-6, f"器官2 平均 HU=50 (得 {r2['mean_hu']})")
    check(r2['name_zh'] == "肾" and r2['name_en'] == "Kidney", "器官2 名称查表命中")
    check(r5['name_zh'] == "类5" and r5['name_en'] == "cls5", "器官5 未登记 → 回退名 类5/cls5")
    check(abs(r5['mean_hu'] - 100.0) < 1e-6, "器官5 平均 HU=100")
    # 恒定区域的离散度必须为 0（上面两个器官内 HU 全同）——独立可验算，不抄实现
    check(abs(r2['sd_hu']) < 1e-6 and abs(r5['sd_hu']) < 1e-6, "HU 恒定区域 SD=0")
    check(abs(r2['median_hu'] - 50.0) < 1e-6 and abs(r2['min_hu'] - 50.0) < 1e-6
          and abs(r2['max_hu'] - 50.0) < 1e-6, "HU 恒定区域 median/min/max 均=50")
    # 非恒定区域：器官7 取 4 个已知值 [0,10,20,90]，各统计量可手算
    vol2 = np.zeros((4, 4, 4), np.float32); mask2 = np.zeros((4, 4, 4), np.uint8)
    for k, hv in enumerate([0.0, 10.0, 20.0, 90.0]):
        mask2[2, 0, k] = 7; vol2[2, 0, k] = hv
    r7 = quantify.compute_organ_stats(vol2, mask2, spacing, {})[0]
    check(abs(r7['mean_hu'] - 30.0) < 1e-6, f"器官7 mean=(0+10+20+90)/4=30 (得 {r7['mean_hu']})")
    check(abs(r7['median_hu'] - 15.0) < 1e-6, f"器官7 median=(10+20)/2=15 (得 {r7['median_hu']})")
    check(abs(r7['min_hu'] - 0.0) < 1e-6 and abs(r7['max_hu'] - 90.0) < 1e-6, "器官7 min=0 max=90")
    # 总体标准差(ddof=0)：√(((0-30)²+(10-30)²+(20-30)²+(90-30)²)/4) = √1250 = 35.3553…
    check(abs(r7['sd_hu'] - np.sqrt(1250.0)) < 1e-4, f"器官7 总体标准差=√1250≈35.36 (得 {r7['sd_hu']:.4f})")
    check(r7['p5_hu'] <= r7['median_hu'] <= r7['p95_hu'], "p5 ≤ median ≤ p95 单调有序")
    check(quantify.compute_organ_stats(vol, np.zeros((4, 4, 4), np.uint8), spacing, names) == [],
          "空蒙版返回空列表")


def test_mesh3d():
    """器官三维表面重建纯函数直接单测——用解析球体验算，不构造 MedicalViewer。"""
    print("[三维表面重建纯函数 mesh3d]")
    import mesh3d as M
    N, R = 48, 15.0
    zz, yy, xx = np.ogrid[:N, :N, :N]
    c = (N - 1) / 2
    mask = (((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= R * R).astype(np.uint8) * 7
    v_exact, a_exact = 4 / 3 * np.pi * R ** 3, 4 * np.pi * R ** 2
    # 默认流程 = 提取 → Taubin 平滑 → 顶点聚类减面（与 3D Slicer 表面模型工作流一致）
    verts, faces = M.extract_surface(mask, 7, (1.0, 1.0, 1.0), step=1)
    check(len(verts) > 0 and faces.shape[1] == 3, f"提取出三角网格 (顶点 {len(verts)}, 面 {len(faces)})")
    s = M.mesh_shape_stats(verts, faces)
    err_v = abs(s['volume_mm3'] - v_exact) / v_exact
    check(err_v < 0.02, f"默认流程体积对解析球体误差 <2% (得 {err_v * 100:.2f}%)")
    # 平滑是否真起作用：关掉后处理，表面积必须明显更差。这条同时钉住
    # 「平滑不可省」与「体积不被平滑破坏」两件事，是本模块最关键的断言。
    vr, fr = M.extract_surface(mask, 7, (1.0, 1.0, 1.0), step=1, smooth=0, decimate_grid=0)
    sr = M.mesh_shape_stats(vr, fr)
    ratio_raw = sr['surface_area_mm2'] / a_exact
    ratio_sm = s['surface_area_mm2'] / a_exact
    check(ratio_raw > 1.05, f"未平滑时表面积高估 >5%（体素阶梯效应确实存在）(得 {(ratio_raw - 1) * 100:+.2f}%)")
    # 比的是【误差】(ratio-1) 而非 ratio 本身：ratio 恒接近 1，直接比会永假
    check((ratio_sm - 1) < (ratio_raw - 1) / 2,
          f"平滑把面积误差压掉一半以上 ({(ratio_raw - 1) * 100:+.2f}% → {(ratio_sm - 1) * 100:+.2f}%)")
    check(abs(s['volume_mm3'] - sr['volume_mm3']) / sr['volume_mm3'] < 0.01,
          "Taubin 平滑几乎不改变体积（正负交替抵消收缩，非纯 Laplacian 的持续缩水）")
    check(len(faces) < len(fr), f"减面确实减少了面数 ({len(fr):,} → {len(faces):,})")
    check(s['sphericity'] > sr['sphericity'], f"平滑后球形度更接近 1 ({sr['sphericity']:.4f} → {s['sphericity']:.4f})")
    # 各向异性 spacing：层厚翻倍则体积翻倍——验证 spacing 确实按 (z,y,x) 传对了
    s2 = M.mesh_shape_stats(*M.extract_surface(mask, 7, (1.0, 1.0, 2.0), step=1))
    check(abs(s2['volume_mm3'] / s['volume_mm3'] - 2.0) < 0.05,
          f"层厚×2 → 体积×2（spacing 轴序正确）(得 {s2['volume_mm3'] / s['volume_mm3']:.3f})")
    # 空网格边界：不存在的标签不得抛异常
    ev, ef = M.extract_surface(mask, 99, (1.0, 1.0, 1.0))
    check(len(ev) == 0 and len(ef) == 0, "不存在的标签 → 返回空网格")
    es = M.mesh_shape_stats(ev, ef)
    check(es['volume_mm3'] == 0.0 and es['sphericity'] == 0.0, "空网格统计量全 0（不返回 nan）")
    check(int(M.render_mesh(ev, ef, size=32)[..., 3].sum()) == 0, "空网格渲染为全透明")
    # 渲染：形状/类型正确，且画出了东西
    img = M.render_mesh(verts, faces, size=120)
    check(img.shape == (120, 120, 4) and img.dtype == np.uint8, f"渲染输出 RGBA {img.shape}")
    check(int((img[..., 3] > 0).sum()) > 120 * 120 * 0.1, "渲染覆盖足够像素（球体应占可观面积）")
    # 不同视角应给出不同图像——否则说明旋转没生效
    img2 = M.render_mesh(verts, faces, size=120, azimuth=90.0, elevation=60.0)
    check(not np.array_equal(img, img2), "不同 azimuth/elevation 渲染结果不同（旋转生效）")
    # STL：facet 数必须等于面数，且首尾为 solid/endsolid
    stl = M.to_stl_bytes(verts, faces, "sphere")
    check(stl.count(b'facet normal') == len(faces), f"STL facet 数 = 面数 {len(faces)}")
    check(stl.startswith(b'solid sphere') and stl.rstrip().endswith(b'endsolid sphere'), "STL 首尾标记正确")
    check(M.to_stl_bytes(ev, ef, "empty").count(b'facet normal') == 0, "空网格导出合法的空 STL")


def test_projection():
    """厚层投影纯函数直接单测——合成体积，断言值均可手算。"""
    print("[厚层投影纯函数 projection]")
    import projection as P
    from constants import AXIAL, CORONAL, SAGITTAL
    # 体积 (4,3,2)：第 z 层全为 z*10 → 沿 z 投影的结果可直接心算
    vol = np.zeros((4, 3, 2), np.float32)
    for z in range(4):
        vol[z] = z * 10.0
    # 关键契约：thickness=1 必须与直接切片逐元素相同，否则会悄悄改变既有渲染行为
    for pl, idx, ref in ((AXIAL, 2, vol[2, :, :]), (CORONAL, 1, vol[:, 1, :]), (SAGITTAL, 0, vol[:, :, 0])):
        got = P.project(vol, pl, idx, thickness=1)
        check(got.shape == ref.shape and bool(np.array_equal(got, ref)),
              f"thickness=1 与直接切片完全一致 (plane={pl}, shape={got.shape})")
    # z=1..3 三层的 MIP=30、MinIP=10、AIP=20（(10+20+30)/3）
    mx = P.project(vol, AXIAL, 2, thickness=3, mode='max')
    mn = P.project(vol, AXIAL, 2, thickness=3, mode='min')
    av = P.project(vol, AXIAL, 2, thickness=3, mode='mean')
    check(float(mx.max()) == 30.0 and float(mx.min()) == 30.0, f"MIP 取层块最大值=30 (得 {float(mx.max())})")
    check(float(mn.max()) == 10.0, f"MinIP 取层块最小值=10 (得 {float(mn.max())})")
    check(abs(float(av.max()) - 20.0) < 1e-6, f"AIP 取层块均值=(10+20+30)/3=20 (得 {float(av.max())})")
    # 厚度超出体积：夹到边界且不抛异常，仍返回该平面的正确形状
    big = P.project(vol, AXIAL, 0, thickness=99, mode='max')
    check(big.shape == (3, 2) and float(big.max()) == 30.0, f"厚度超界 → 夹到全体积 (得 {big.shape}, max={float(big.max())})")
    # 层块范围：中心 5、厚度 4、长度 10 → [3,7)；靠上边界时回推保证足额厚度
    check(P.slab_bounds(5, 4, 10) == (3, 7), f"slab_bounds(5,4,10)=(3,7) (得 {P.slab_bounds(5, 4, 10)})")
    check(P.slab_bounds(9, 4, 10) == (6, 10), f"靠边界回推保足额厚度 (得 {P.slab_bounds(9, 4, 10)})")
    # 毫米换算：Axial 沿 z 用层厚，Coronal/Sagittal 沿平面内轴用像素间距——物理尺度不同
    check(abs(P.thickness_mm(5, AXIAL, 0.7, 1.25) - 6.25) < 1e-9, "Axial 5 层 × 层厚1.25 = 6.25mm")
    check(abs(P.thickness_mm(5, CORONAL, 0.7, 1.25) - 3.5) < 1e-9, "Coronal 5 层 × 像素间距0.7 = 3.5mm")
    bad = False
    try:
        P.project(vol, AXIAL, 0, mode='median')
    except ValueError:
        bad = True
    check(bad, "非法投影模式抛 ValueError")


def test_registration():
    """二维刚性配准纯函数单测——合成位移/旋转，断言可手算的量与方向。"""
    print("[刚性配准纯函数 registration]")
    import scipy.ndimage as ndi

    import registration as REG
    rng = np.random.default_rng(3)
    # 造一个有结构的图（纯噪声无法配准；纯常数则相关系数无定义）
    base = np.zeros((96, 96), np.float32)
    base[20:60, 25:70] = 300.0
    base[35:45, 40:55] = -200.0
    base += rng.normal(0, 5, base.shape).astype(np.float32)
    check(abs(REG.normalized_cross_correlation(base, base) - 1.0) < 1e-9, "NCC(自身)=1")
    check(REG.normalized_cross_correlation(base, np.zeros_like(base)) == 0.0,
          "NCC 对常数图返回 0（无定义时不返回 nan）")
    # 平移估计：返回的必须是「moving 相对 ref 的位移」，符号与构造一致。
    # 这条方向断言极其重要——符号写反时位移【数值仍完全正确】、图看着也动了，
    # 但对齐结果会更歪（实测 MAE 从 262 涨到 355），属静默失真。
    for ts in ((7, -5), (-11, 8), (0, 0)):
        moved = ndi.shift(base, ts, order=1, mode='nearest')
        est = REG.estimate_translation(base, moved)
        check(est == ts, f"平移估计 {ts} → {est}（符号即 moving 相对 ref 的位移）")
    # apply_rigid 必须把图对回去：MAE 应大幅下降、NCC 应逼近 1
    moved = ndi.shift(base, (9, -6), order=1, mode='nearest')
    r = REG.register_rigid(base, moved, max_angle=0)          # 只搜平移
    aligned = REG.apply_rigid(moved, r['angle_deg'], r['shift_yx'])
    mae0 = float(np.abs(base - moved).mean())
    mae1 = float(np.abs(base - aligned).mean())
    check(r['shift_yx'] == (9, -6) and r['applied'], f"仅平移模式估出 (9,-6) 并采用 (得 {r['shift_yx']})")
    check(mae1 < mae0 / 3, f"配准后 MAE 大幅下降 ({mae0:.1f} → {mae1:.1f} HU)")
    check(r['ncc_after'] > r['ncc_before'], f"配准后 NCC 提升 ({r['ncc_before']:.4f} → {r['ncc_after']:.4f})")
    # 旋转+平移：角度应被搜到（步长 0.5°，故允许半步误差）
    moved2 = ndi.shift(ndi.rotate(base, 3.0, reshape=False, order=1, mode='nearest'),
                       (6, -4), order=1, mode='nearest')
    r2 = REG.register_rigid(base, moved2, max_angle=5.0, angle_step=0.5)
    check(abs(r2['angle_deg'] - 3.0) <= 0.5, f"刚性模式估出旋转 ≈3.0° (得 {r2['angle_deg']:+.1f}°)")
    check(r2['ncc_after'] > r2['ncc_before'],
          f"刚性配准提升 NCC ({r2['ncc_before']:.4f} → {r2['ncc_after']:.4f})")
    # 安全阀：两张无关的图配不出提升时，必须拒绝而不是硬套一个变换
    other = rng.normal(0, 50, base.shape).astype(np.float32)
    r3 = REG.register_rigid(base, other, max_angle=0)
    if not r3['applied']:
        check(r3['angle_deg'] == 0.0 and r3['shift_yx'] == (0, 0),
              "配准无提升时安全阀生效：变换归零且 applied=False")
    else:
        check(r3['ncc_after'] >= r3['ncc_before'], "若采用配准，则 NCC 必不低于配准前")
    # 尺寸不同必须显式报错，而非静默给出无意义结果
    raised = False
    try:
        REG.register_rigid(base, np.zeros((50, 50), np.float32))
    except ValueError:
        raised = True
    check(raised, "尺寸不同抛 ValueError")
    # 非有限输入不得让估计崩溃（畸形 DICOM 可产出 NaN/Inf）
    bad = base.copy(); bad[0, 0] = np.nan; bad[1, 1] = np.inf
    est_bad = REG.estimate_translation(base, bad)
    check(isinstance(est_bad, tuple) and len(est_bad) == 2, f"含 NaN/Inf 时仍返回合法位移 {est_bad}")


def test_followup():
    """随访对比定量纯函数直接单测——合成切片，不构造 MedicalViewer。断言值均可手算。"""
    print("[随访对比定量纯函数 followup]")
    import followup as F
    # 可比性守卫：形状不同必须拒绝，而不是强行 resize 制造「看起来能比」的假象
    ok, why = F.can_compare(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32))
    check(not ok and "尺寸不同" in why, f"矩阵尺寸不同 → 拒绝比较 ({why})")
    ok, _ = F.can_compare(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32))
    check(ok, "同形切片 → 可比")
    # 差值统计：prev 全 0，cur 为 [0,10,20,90] → 差值即 cur 本身，各量可手算
    prev = np.zeros((2, 2), np.float32)
    cur = np.array([[0.0, 10.0], [20.0, 90.0]], np.float32)
    s = F.compare_slices(cur, prev)
    check(abs(s['mean_diff'] - 30.0) < 1e-6, f"mean_diff=(0+10+20+90)/4=30 (得 {s['mean_diff']})")
    check(abs(s['mae'] - 30.0) < 1e-6, f"mae=30（差值全非负，等于 mean_diff）(得 {s['mae']})")
    check(abs(s['rmse'] - np.sqrt((0 + 100 + 400 + 8100) / 4)) < 1e-4,
          f"rmse=√((0+100+400+8100)/4)=√2150≈46.37 (得 {s['rmse']:.4f})")
    check(abs(s['sd_diff'] - np.sqrt(1250.0)) < 1e-4, f"sd_diff=√1250≈35.36 (得 {s['sd_diff']:.4f})")
    check(abs(s['max_abs'] - 90.0) < 1e-6, "max_abs=90")
    # 完全相同的两张切片：差值恒为 0，相关系数为 1
    same = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    s2 = F.compare_slices(same, same)
    check(s2['mean_diff'] == 0 and s2['mae'] == 0 and s2['rmse'] == 0, "相同切片 → 差值统计全为 0")
    check(abs(s2['corr'] - 1.0) < 1e-9, f"相同切片 → 相关系数=1 (得 {s2['corr']})")
    # 有符号性：cur 比 prev 低时 mean_diff 必须为负（正=密度升高的约定）
    s3 = F.compare_slices(np.zeros((2, 2), np.float32), np.full((2, 2), 50.0, np.float32))
    check(s3['mean_diff'] == -50.0, f"当前低于既往 → mean_diff 为负 (得 {s3['mean_diff']})")
    # 常数切片方差为 0，相关系数无定义 → 必须返回 nan 而非崩溃或给出假值
    check(np.isnan(s3['corr']), "常数切片 → 相关系数返回 nan（无定义，不编造）")
    # 差值渲染：0 差异全透明；正负分属暖/冷色；饱和阈值处不透明度最高
    rgba = F.diff_to_rgba(np.array([[0.0, 300.0], [-300.0, 0.0]], np.float32), clip_hu=200.0)
    check(rgba.shape == (2, 2, 4) and rgba.dtype == np.uint8, f"差值图 RGBA 形状/类型 (得 {rgba.shape})")
    check(int(rgba[0, 0, 3]) == 0 and int(rgba[1, 1, 3]) == 0, "零差异处完全透明")
    check(int(rgba[0, 1, 0]) > int(rgba[0, 1, 2]) and int(rgba[1, 0, 2]) > int(rgba[1, 0, 0]),
          "正差值偏暖色、负差值偏冷色（超阈值已 clip）")
    # 非有限输入防御：畸形 DICOM（异常 RescaleSlope / 损坏像素）可产出 NaN/±Inf HU。
    # 不中和的话统计量整片变 nan 并直接显示到界面（"Δnan 绝对差 nan"），
    # 差值图转 uint8 更是未定义行为——会把"无法计算"渲染成看似真实的颜色。
    for nm, bad in (("NaN", np.nan), ("+Inf", np.inf), ("-Inf", -np.inf)):
        a = np.zeros((3, 3), np.float32); a[0, 0] = bad
        st = F.compare_slices(a, np.zeros((3, 3), np.float32))
        finite = all(np.isfinite(v) for k, v in st.items() if k != 'corr')
        check(finite, f"切片含 {nm} → 差值统计仍全为有限值 (mean_diff={st['mean_diff']:.1f})")
    img_bad = F.diff_to_rgba(np.array([[np.nan, np.inf], [-np.inf, 0.0]], np.float32))
    check(img_bad.dtype == np.uint8 and img_bad.shape == (2, 2, 4),
          "差值图对 NaN/±Inf 输入仍产出合法 uint8 RGBA")


def test_lung_fallback():
    """AI 数学降级纯函数直接单测——合成体积，不构造推理线程/MedicalViewer。"""
    print("[AI 数学降级纯函数 segmentation.segment_lungs_fallback]")
    import segmentation
    from constants import LUNG_FALLBACK_LABEL
    vol = np.zeros((6, 12, 12), np.float32)   # 全软组织 HU=0（非空气）
    vol[2:4, 4:8, 4:8] = -900.0               # 内部肺区（不触边界）：2×4×4=32 体素空气
    vol[0, 0, 0] = -1000.0                     # 体外空气（触边界角）：应被剔除
    mask = segmentation.segment_lungs_fallback(vol)
    check(mask.dtype == np.uint8 and mask.shape == vol.shape, "返回 uint8 同形状蒙版")
    got = int((mask == LUNG_FALLBACK_LABEL).sum())
    check(got == 32, f"内部肺区被标 32 体素 (得 {got})")
    check(mask[0, 0, 0] == 0, "体外空气（触边界）被剔除")
    check(int(segmentation.segment_lungs_fallback(np.zeros((4, 8, 8), np.float32)).sum()) == 0,
          "无空气体积返回全零")


def test_mpr_geometry():
    """MPR 坐标几何纯函数直接单测——纯整数/数组运算，无 Qt / MedicalViewer。"""
    print("[MPR 坐标几何纯函数 mpr_geometry]")
    import mpr_geometry as g
    from constants import AXIAL, CORONAL, SAGITTAL
    shape = (40, 200, 300)   # (Z, Y, X)
    cur = (10, 20, 30)       # (z, y, x)
    check(g.hover_to_voxel(AXIAL, 50, 60, cur, shape) == (10, 60, 50), "Axial 悬停 (px,py)->(x,y)，z 不变")
    check(g.hover_to_voxel(CORONAL, 50, 15, cur, shape) == (15, 20, 50), "Coronal 悬停 (px,py)->(x,z)，y 不变")
    check(g.hover_to_voxel(SAGITTAL, 70, 15, cur, shape) == (15, 70, 30), "Sagittal 悬停 (px,py)->(y,z)，x 不变")
    check(g.hover_to_voxel(AXIAL, 999, 999, cur, shape) == (10, 199, 299), "越界裁剪到体积上界")
    check(g.hover_to_voxel(AXIAL, -5, -5, cur, shape) == (10, 0, 0), "负坐标裁剪到 0")
    check(g.voxel_to_crosshair(AXIAL, 10, 20, 30) == (30, 20), "Axial 十字线 (x,y)")
    check(g.voxel_to_crosshair(CORONAL, 10, 20, 30) == (30, 10), "Coronal 十字线 (x,z)")
    check(g.voxel_to_crosshair(SAGITTAL, 10, 20, 30) == (20, 10), "Sagittal 十字线 (y,z)")
    check(g.nearest_slice([0, 5, 10, 15, 20], 12) == 2, "最近解剖切片 = 索引2 (z=10)")
    check(g.nearest_slice([0, 5, 10, 15, 20], 100) == 4, "超出范围取最末切片")


def test_mouse_interaction(app):
    """鼠标交互逐工具验证：完整 press→move→release 序列，断言发出的信号与其载荷。

    graphics_view 的三个鼠标事件处理器（press/move/release 共约 190 行）此前
    一行未测——所有工具的实际交互逻辑都在里面，是覆盖率最大的盲区。
    只断言"不崩"没有意义，故这里捕获信号并核对载荷内容。

    刻意使用**独立的 MedicalGraphicsView 实例**而非主窗口里的那个：
      1) 被测对象就是 view 自身的交互逻辑，不该连带触发 MedicalViewer 的重处理器
         （3D 追踪会扫描整卷、矩形截取会弹 QMessageBox.question 阻塞测试——
          实测确实因此挂死，测试顶部只 stub 了 information/warning 两个）；
      2) 不依赖真实数据，可进 SKIP_REAL_DATA 子集。
    """
    print("[鼠标交互逐工具]")
    from PySide6.QtGui import QPixmap

    from constants import (
        TOOL_AI_TRACK,
        TOOL_CROP,
        TOOL_DRAW,
        TOOL_POINTER,
        TOOL_RECT_CROP,
        TOOL_ROI,
        TOOL_RULER,
        TOOL_SEG_BRUSH,
        TOOL_SEG_ERASE,
    )
    from graphics_view import MedicalGraphicsView
    view = MedicalGraphicsView(1)              # 独立实例：不连 MedicalViewer 的任何处理器
    pm = QPixmap(256, 256); pm.fill(Qt.black)  # 需要有图元，坐标映射与命中检测才有意义
    view.set_image(pm)
    view.resize(300, 300); view.show()
    app.processEvents()

    def press(x, y, btn=Qt.LeftButton):
        view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), btn, btn, Qt.NoModifier))
    def move(x, y, btn=Qt.LeftButton):
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, btn, Qt.NoModifier))
    def release(x, y, btn=Qt.LeftButton):
        view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(x, y), btn, Qt.NoButton, Qt.NoModifier))

    def drag(tool, x0, y0, x1, y1, btn=Qt.LeftButton, steps=3):
        """选中工具后完整拖拽一次，返回该过程中捕获到的各信号载荷。"""
        view.current_tool = tool
        got = {}
        conns = []
        for sig_name in ('clicked_pos', 'annotation_added', 'crop_requested',
                         'track_requested', 'window_changed', 'seg_paint_requested', 'mouse_hovered'):
            sig = getattr(view, sig_name)
            got[sig_name] = []
            # 默认参数绑定当前 key，避免闭包全部捕获到最后一个 sig_name
            slot = (lambda *a, _k=sig_name, _g=got: _g[_k].append(a))
            sig.connect(slot); conns.append((sig, slot))
        try:
            press(x0, y0, btn)
            for i in range(1, steps + 1):
                move(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps, btn)
            release(x1, y1, btn)
            app.processEvents()
        finally:
            for sig, slot in conns:
                sig.disconnect(slot)
        return got

    # 指针：单击应发出 clicked_pos（用于读取该点 HU）
    g = drag(TOOL_POINTER, 120, 120, 120, 120, steps=1)
    check(len(g['clicked_pos']) >= 1, f"指针工具单击发出 clicked_pos ({len(g['clicked_pos'])} 次)")
    # 卡尺：拖拽应产出 ruler 标注，且两端点即拖拽起止
    g = drag(TOOL_RULER, 100, 100, 200, 160)
    ann = [a[0] for a in g['annotation_added'] if a[0].get('type') == 'ruler']
    check(len(ann) == 1, f"卡尺拖拽产出 1 条 ruler 标注 (得 {len(ann)})")
    if ann:
        # 标注记的是【场景坐标】，而拖拽给的是 view 坐标——fitInView 缩放后两者不等，
        # 故不能断言绝对值。改断言方向性：终点必在起点的右下方，与拖拽方向一致。
        p1, p2 = ann[0]['p1'], ann[0]['p2']
        check(p2[0] > p1[0] and p2[1] > p1[1],
              f"卡尺端点方向与拖拽一致（右下）(p1={tuple(round(c) for c in p1)} → p2={tuple(round(c) for c in p2)})")
    # 画笔：自由绘制产出的类型是 'path'（不是 'draw'），载荷含逐点轨迹
    g = drag(TOOL_DRAW, 80, 80, 180, 180, steps=5)
    dr = [a[0] for a in g['annotation_added'] if a[0].get('type') == 'path']
    check(len(dr) == 1 and len(dr[0].get('points', [])) >= 3,
          f"画笔产出 path 标注且含多点 (得 {len(dr[0].get('points', [])) if dr else 0} 点)")
    # 套索：闭合后应发 crop_requested，携带多边形顶点
    g = drag(TOOL_CROP, 90, 90, 150, 150, steps=4)
    check(len(g['crop_requested']) == 1 and len(g['crop_requested'][0][0]) >= 3,
          f"套索发出 crop_requested 且顶点 ≥3 (得 {len(g['crop_requested'][0][0]) if g['crop_requested'] else 0})")
    # 矩形截取：同样走 crop_requested，但顶点应是矩形四角
    g = drag(TOOL_RECT_CROP, 100, 100, 180, 150)
    check(len(g['crop_requested']) == 1, f"矩形截取发出 crop_requested ({len(g['crop_requested'])} 次)")
    # 3D 追踪：应发 track_requested，携带框选矩形
    g = drag(TOOL_AI_TRACK, 110, 110, 170, 160)
    check(len(g['track_requested']) == 1, f"3D 追踪发出 track_requested ({len(g['track_requested'])} 次)")
    if g['track_requested']:
        r = g['track_requested'][0][0]
        check(r.width() > 0 and r.height() > 0, f"追踪矩形非退化 ({r.width():.0f}×{r.height():.0f})")
    # 分割画笔 / 橡皮：应发 seg_paint_requested，第二参数区分补画与擦除
    g = drag(TOOL_SEG_BRUSH, 120, 120, 160, 160, steps=4)
    check(len(g['seg_paint_requested']) == 1 and g['seg_paint_requested'][0][1] is False,
          "分割画笔发出 seg_paint_requested 且 erase=False")
    g = drag(TOOL_SEG_ERASE, 120, 120, 160, 160, steps=4)
    check(len(g['seg_paint_requested']) == 1 and g['seg_paint_requested'][0][1] is True,
          "分割橡皮发出 seg_paint_requested 且 erase=True")
    # 右键拖拽调窗宽窗位：应发 window_changed
    g_win = drag(TOOL_POINTER, 100, 100, 200, 150, btn=Qt.RightButton, steps=3)
    check(len(g_win['window_changed']) >= 1, f"右键拖拽发出 window_changed ({len(g_win['window_changed'])} 次)")
    # 调窗期间【有意不发】mouse_hovered——见 graphics_view 注释「调窗期间不更新十字线，
    # 避免光标移动引起视图混乱」。这条断言把该设计意图钉住，防止日后被"顺手补上"。
    check(len(g_win['mouse_hovered']) == 0, "调窗拖拽期间不发 mouse_hovered（有意设计，避免十字线乱跳）")
    # 非调窗的左键拖拽则必须持续发 mouse_hovered，MPR 联动十字线依赖它
    g_hov = drag(TOOL_POINTER, 100, 100, 180, 140, steps=3)
    check(len(g_hov['mouse_hovered']) >= 1, f"非调窗移动发出 mouse_hovered ({len(g_hov['mouse_hovered'])} 次)")
    # ROI：椭圆图元由 MedicalViewer 依 annotation_added(type='roi') 创建，
    # 独立 view 中没有那个处理器，故此处断言信号载荷而非场景图元。
    g_roi = drag(TOOL_ROI, 130, 130, 190, 180)
    roi = [a[0] for a in g_roi['annotation_added'] if a[0].get('type') == 'roi']
    check(len(roi) == 1, f"ROI 拖拽发出 type='roi' 的 annotation_added (得 {len(roi)})")
    # 退化拖拽（起止同点）不得产出标注，也不得崩
    crashed = False
    try:
        g = drag(TOOL_RULER, 150, 150, 150, 150, steps=1)
    except Exception:
        crashed = True
    check(not crashed, "零长度拖拽不崩")
    view.close()          # 独立实例用完即弃，无需还原状态，也不会污染主窗口
    app.processEvents()


def test_mesh3d_ui(v, app):
    """三维重建接入：按钮随器官有无启停，端到端不崩，网格体积与体素法互相印证。"""
    print("[三维重建接入]")
    from PySide6.QtWidgets import QDialog

    import mesh3d as M
    saved_mask = v.volume_mask
    saved_exec = QDialog.exec
    QDialog.exec = lambda self: None            # 不阻塞在模态窗
    try:
        v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
        v._update_organ_stats(); app.processEvents()
        check(not v.btn_mesh3d.isEnabled(), "无器官时三维重建按钮禁用")
        Z, H, W = v.volume_hu.shape
        zz, yy, xx = np.ogrid[:Z, :H, :W]
        v.volume_mask[((zz - Z // 2) ** 2 / 20 ** 2 + (yy - H // 2) ** 2 / 50 ** 2
                       + (xx - W // 2) ** 2 / 40 ** 2) <= 1] = 5
        v._update_organ_stats(); app.processEvents()
        check(v.btn_mesh3d.isEnabled(), "检出器官后三维重建按钮启用")
        v.cb_paint_target.setCurrentIndex(v.cb_paint_target.findData(5))
        crashed = False
        try:
            v.show_mesh3d(); app.processEvents()
        except Exception as ex:
            crashed = True; print("   ", type(ex).__name__, ex)
        check(not crashed, "show_mesh3d 端到端不崩（提取+渲染+弹窗）")
        # 网格体积须与体素计数法互相印证——两种独立算法，差异应在 2% 内
        ds = v.dicom_datasets[0]
        ps = v._dcm_float(ds, 'PixelSpacing', 1.0, idx=0)
        st = v._dcm_float(ds, 'SliceThickness', ps * 3)
        verts, faces = M.extract_surface(v.volume_mask, 5, (ps, ps, st), step=2)
        mesh_ml = M.mesh_shape_stats(verts, faces)['volume_mm3'] / 1000.0
        vox_ml = int((v.volume_mask == 5).sum()) * ps * ps * st / 1000.0
        rel = abs(mesh_ml - vox_ml) / vox_ml
        check(rel < 0.02, f"网格体积与体素法互印证，相对差 <2% (得 {rel * 100:.2f}%: {mesh_ml:.1f} vs {vox_ml:.1f} mL)")
    finally:
        QDialog.exec = saved_exec
        v.volume_mask = saved_mask
        v._update_organ_stats(); app.processEvents()


def test_projection_ui(v, app):
    """厚层投影接入渲染路径：默认单层必须与原行为完全一致，切到 MIP 才改变画面。"""
    print("[厚层投影接入渲染]")
    import projection as P
    from constants import AXIAL
    vd = v.views[1]
    check(vd['cb_proj'].currentIndex() == 0 and vd['sp_thick'].value() == 1
          and not vd['sp_thick'].isEnabled(),
          "默认单层模式、厚度1、厚度框禁用（不改变既有默认体验）")
    z = v.current_3d_pos[0]
    # 默认路径与直接切片逐元素相同——这是接入投影后最重要的不回归契约
    check(bool(np.array_equal(P.project(v.volume_hu, AXIAL, z, 1, 'max'), v.volume_hu[z, :, :])),
          "单层投影结果 == 直接切片（真实数据逐元素比对）")
    # 切到 MIP 后厚度框启用，且渲染不崩
    vd['cb_proj'].setCurrentIndex(1); vd['sp_thick'].setValue(10)
    app.processEvents(); v.update_display(); app.processEvents()
    check(vd['sp_thick'].isEnabled(), "选到投影模式后厚度框启用")
    mip = P.project(v.volume_hu, AXIAL, z, 10, 'max')
    check(mip.shape == v.volume_hu[z].shape and float(mip.mean()) > float(v.volume_hu[z].mean()),
          f"MIP 均值高于单层 ({mip.mean():.1f} > {v.volume_hu[z].mean():.1f})")
    vd['cb_proj'].setCurrentIndex(2)      # MinIP
    app.processEvents(); v.update_display(); app.processEvents()
    minip = P.project(v.volume_hu, AXIAL, z, 10, 'min')
    check(float(minip.mean()) < float(v.volume_hu[z].mean()), "MinIP 均值低于单层")
    vd['cb_proj'].setCurrentIndex(0); vd['sp_thick'].setValue(1)   # 还原，勿污染后续测试
    app.processEvents(); v.update_display(); app.processEvents()


def test_mask_cache_guard():
    """分割蒙版磁盘缓存的恢复守卫纯函数直接单测——无 Qt / 真实数据。

    缓存按 PatientID 命名，只比 shape 会把同一患者另一序列（随访/复扫，常同为 512²）
    的蒙版静默套到当前序列上，器官定量随之给出错误体积。故必须 UID+shape 双匹配。"""
    print("[分割蒙版缓存守卫纯函数 annotation_lab.mask_cache_matches]")
    import annotation_lab as al
    shp = (233, 512, 512)
    ok, why = al.mask_cache_matches("1.2.840.A", shp, "1.2.840.A", shp)
    check(ok and why == "", "同一序列（UID 与 shape 皆同）→ 恢复缓存")
    ok, why = al.mask_cache_matches("1.2.840.A", shp, "1.2.840.B", shp)
    check(not ok and "SeriesInstanceUID" in why,
          "同患者另一序列（shape 相同、UID 不同）→ 拒绝套用（核心回归：防串序列）")
    ok, why = al.mask_cache_matches("1.2.840.A", (233, 512, 512), "1.2.840.A", (200, 512, 512))
    check(not ok and "shape" in why, "shape 不匹配 → 拒绝")
    ok, why = al.mask_cache_matches("", shp, "1.2.840.A", shp)
    check(not ok, "缓存缺 UID（旧版本产物）→ 拒绝，宁可重跑 AI")
    ok, why = al.mask_cache_matches("1.2.840.A", shp, "", shp)
    check(not ok, "当前序列缺 UID → 拒绝")


def test_mask_cache_roundtrip(app):
    """蒙版缓存 save→reload 往返：同序列恢复、同患者另一序列拒绝（合成 DICOM，无真实数据）。"""
    print("[蒙版缓存 save→reload 往返]")
    import glob
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    ed = os.path.join(_ROOT, "Exported_Lesions")
    pid = "RID_CACHE_TEST"
    made = []

    def _mkdir_series(uid, z=3):
        d = tempfile.mkdtemp()
        for i in range(z):
            _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (16, 16), uid, ipp_z=i, inst=i + 1, pid=pid)
        return d

    uid_a, uid_b = generate_uid(), generate_uid()
    da, db = _mkdir_series(uid_a), _mkdir_series(uid_b)
    try:
        vc = m.MedicalViewer(); app.processEvents()
        if vc.ai_thread:
            vc.ai_thread.cancel()
        # 序列 A：造一个非空蒙版并保存
        vc.load_data(da); app.processEvents()
        if vc.ai_thread:
            vc.ai_thread.cancel()
        vc.volume_mask = np.zeros_like(vc.volume_hu, dtype=np.uint8)
        vc.volume_mask[0, :4, :4] = 5          # 标记为器官5，便于区分
        vc.save_project()
        made = glob.glob(os.path.join(ed, f"{pid}_*"))
        check(any(f.endswith("_mask.npz") for f in made), "save_project 落盘 _mask.npz")

        # 重开序列 A（同 UID 同 shape）→ 应恢复
        vc.volume_mask = None
        restored_a = vc._load_saved_mask(pid)
        check(restored_a and vc.volume_mask is not None and int(vc.volume_mask[0, 0, 0]) == 5,
              "重开同一序列 → 缓存被恢复（省掉 ~100s 重算）")

        # 切到序列 B（同 PatientID、同 shape、不同 SeriesInstanceUID）→ 必须拒绝
        vc.load_data(db); app.processEvents()
        if vc.ai_thread:
            vc.ai_thread.cancel()
        vc.volume_mask = None
        restored_b = vc._load_saved_mask(pid)
        check(not restored_b and vc.volume_mask is None,
              "切到同患者另一序列（同 shape 不同 UID）→ 拒绝套用旧蒙版（核心回归）")
    finally:
        for f in made:
            try:
                os.remove(f)
            except OSError:
                pass
        shutil.rmtree(da, ignore_errors=True); shutil.rmtree(db, ignore_errors=True)


def test_hu_conversion(app):
    """DICOM 像素 → HU 的数值正确性：HU = pixel × RescaleSlope + RescaleIntercept。

    这是探针 HU / ROI 统计 / 器官定量 / AI 归一化的共同地基，此前只被“不崩溃”覆盖，
    从未断言过算得对（test_quantify 直接喂合成 HU，绕过了本转换）。"""
    print("[DICOM→HU 转换正确性]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    vh = m.MedicalViewer(); app.processEvents()
    if vh.ai_thread:
        vh.ai_thread.cancel()
    # (像素值, slope, intercept, 期望 HU)
    cases = [(100, 1, -1024, -924.0),     # GE 典型：空气≈-1000 附近
             (500, 2, -1000, 0.0),        # 非 1 的 slope，验证真在乘
             (0, 1, -1024, -1024.0),      # 像素 0 → 纯 intercept
             (1200, 1, 0, 1200.0)]        # intercept=0 → 原值
    for pix, slope, icpt, want in cases:
        d = tempfile.mkdtemp()
        try:
            uid = generate_uid()
            for i in range(3):
                _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (8, 8), uid, ipp_z=i, inst=i + 1,
                               pix=pix, slope=slope, intercept=icpt)
            vh.load_data(d); app.processEvents()
            if vh.ai_thread:
                vh.ai_thread.cancel()
            got = float(vh.volume_hu[0, 0, 0])
            check(abs(got - want) < 1e-3 and bool(np.allclose(vh.volume_hu, want)),
                  f"pix={pix} slope={slope} intercept={icpt} → HU={want}（得 {got}）")
        finally:
            shutil.rmtree(d, ignore_errors=True)


def main_run():
    app = QApplication([])
    # 有真实数据（本地开发）跑全套；无数据或 CI（SKIP_REAL_DATA=1）只跑数据无关的自包含测试。
    has_data = (os.path.isdir(os.path.join(_ROOT, "肺癌"))
                and not os.environ.get("SKIP_REAL_DATA"))
    if not has_data:
        print("WARN: 无 ../肺癌 真实数据（或 SKIP_REAL_DATA=1），仅运行数据无关的自包含测试")
        # 这些测试自建合成 DICOM / 用 /nonexistent.onnx 走数学降级，不依赖真实数据或 119MB 权重
        for t in (test_ai_engine, test_mixed_shape_dicom, test_recon_finite,
                  test_close_cancels_ai, test_malformed_pixels, test_empty_dicom_tags,
                  test_export_path_safety, test_dicom_sort_consistency, test_i18n_persistent,
                  test_hu_conversion, test_mask_cache_roundtrip, test_mouse_interaction):
            t(app)
        test_quantify()      # 纯函数单测，无需 app / 真实数据
        test_lung_fallback()
        test_followup()
        test_registration()
        test_projection()
        test_mesh3d()
        test_mpr_geometry()
        test_mask_cache_guard()
        test_recon_numerics()          # 重建数值正确性：解析模体，无 Qt / 真实数据
        test_recon_pipeline_helpers()
    else:
        v = m.MedicalViewer(data_dir=os.path.join(_ROOT, "肺癌"))
        app.processEvents()
        if v.ai_thread:
            v.ai_thread.cancel()
        test_startup(v)
        test_ai_engine(app)
        test_prior_fixes(v, app)
        test_multiorgan_and_edit(v, app)
        test_roi(v, app)
        test_mpr_ruler_spacing(v, app)
        test_mpr_aniso_aspect(v, app)
        test_sampling_density(v)
        test_compare(v, app)
        test_cine_keyboard(v, app)
        test_compliance(v, app)
        test_edge_cases(v, app)
        test_mixed_shape_dicom(app)
        test_legend_consistency(v, app)
        test_recon_finite(app)
        test_malformed_annotations(v, app)
        test_close_cancels_ai(app)
        test_malformed_pixels(app)
        test_empty_dicom_tags(app)
        test_export_path_safety(app)
        test_dicom_sort_consistency(app)
        test_i18n_persistent(app)
        test_projection_ui(v, app)
        test_mesh3d_ui(v, app)
        test_hu_conversion(app)
        test_mask_cache_roundtrip(app)
        test_mouse_interaction(app)
        test_quantify()
        test_lung_fallback()
        test_followup()
        test_registration()
        test_projection()
        test_mesh3d()
        test_mpr_geometry()
        test_mask_cache_guard()
        test_recon_numerics()          # 重建数值正确性：解析模体，无 Qt / 真实数据
        test_recon_pipeline_helpers()
    print("\n" + ("全部通过" if not _FAILS else f"{len(_FAILS)} 项失败: " + "; ".join(_FAILS)))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main_run())
