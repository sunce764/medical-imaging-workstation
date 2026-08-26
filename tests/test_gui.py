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
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QGraphicsTextItem, QGraphicsView, QMessageBox

import ai_engine
import main as m
from constants import AXIAL, CORONAL, MANUAL_TRACK_LABEL, TOOL_POINTER, TOOL_RULER
from graphics_view import ROIGraphicsItem

# 静音弹窗，避免离屏阻塞。三个都必须 stub：question 曾被遗漏，导致触发
# 「矩形截取 → 是否保存?」的测试挂死（模态框弹出后无人应答，进程永久阻塞）。
# question 返回 No：测试默认不触发保存副作用，需要保存路径的用例自行临时改写。
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

_FAILS = []


def _record_uncaught_exception(exc_type, exc_value, exc_tb):
    """把 Qt signal/slot 吞掉的 Python exception 计为测试失败。"""
    _FAILS.append(f"uncaught {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _record_uncaught_exception


def check(cond, label):
    print(("  PASS " if cond else "  FAIL ") + label)
    if not cond:
        _FAILS.append(label)


def drain(engine, app, extra=40):
    while engine.isRunning():
        app.processEvents()
    for _ in range(extra):
        app.processEvents()


def _mark_supported_capabilities(viewer, slice_spacing=1.0):
    """手工 volume fixture 显式声明其测试所需的完整 CT contract。"""
    from types import SimpleNamespace

    from dicom_geometry import SeriesGeometry
    viewer.hu_calibrated = True
    viewer.canonical_orientation = True
    viewer.inplane_spacing_valid = True
    viewer.uniform_z_geometry_valid = True
    viewer.series_geometry = SeriesGeometry(
        True, True, True, True, None, None, float(slice_spacing))
    if getattr(viewer, 'dicom_datasets', None):
        viewer.dicom_datasets = [
            (SimpleNamespace(PatientID='SYNTH', PatientName='SYNTH', Modality='CT',
                             SeriesInstanceUID='1.2.3', SOPInstanceUID=f'1.2.3.{i + 1}',
                             ImageOrientationPatient=(1, 0, 0, 0, 1, 0),
                             ImagePositionPatient=(0, 0, i * float(slice_spacing)),
                             PixelSpacing=(1.0, 1.0), RescaleSlope=1,
                             RescaleIntercept=-1024)
             if dataset is None else dataset)
            for i, dataset in enumerate(viewer.dicom_datasets)]
        for i, dataset in enumerate(viewer.dicom_datasets):
            defaults = {
                'PatientID': 'SYNTH', 'PatientName': 'SYNTH', 'Modality': 'CT',
                'SeriesInstanceUID': '1.2.3', 'SOPInstanceUID': f'1.2.3.{i + 1}',
                'ImageOrientationPatient': (1, 0, 0, 0, 1, 0),
                'ImagePositionPatient': (0, 0, i * float(slice_spacing)),
                'PixelSpacing': (1.0, 1.0), 'RescaleSlope': 1, 'RescaleIntercept': -1024,
            }
            for name, value in defaults.items():
                if not hasattr(dataset, name):
                    setattr(dataset, name, value)


def test_runner_catches_qt_slot_exceptions():
    """PySide 会吞掉 slot exception；自定义 runner 必须把它转成 FAIL / exit 1。"""
    from contextlib import redirect_stderr
    from io import StringIO

    from PySide6.QtCore import QObject, Signal

    class _Probe(QObject):
        fired = Signal()

    def explode():
        raise RuntimeError("intentional Qt-slot probe")

    probe = _Probe()
    probe.fired.connect(explode)
    before = len(_FAILS)
    # 这是已知坏输入的自检；隐藏预期 traceback，再移除它刻意注入的 FAIL。
    with redirect_stderr(StringIO()):
        probe.fired.emit()
    captured = _FAILS[before:]
    del _FAILS[before:]
    check(captured == ["uncaught RuntimeError: intentional Qt-slot probe"],
          "Qt slot 未捕获异常会进入 FAIL 列表（不再出现 traceback + exit 0 假绿）")


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
    expected_z = v.volume_hu.shape[0] - 1 - 60
    check(v.current_3d_pos[0] == expected_z and v.slider_slice.value() == expected_z,
          "MPR 上 S / 下 I 悬停同步翻转后的光标与滑条")
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
    # 本地 RIDER 副本的 ImageType=DERIVED 且无 RescaleType，产品现已正确 viewer-only。
    # 这里测的是 organ-edit UI 接线，故显式把该测试 fixture 标成具备完整 synthetic contract，
    # 用完立即恢复真实 series capability；不把这一 override 当作 DICOM HU 证据。
    saved_geometry = v.series_geometry
    _mark_supported_capabilities(v, saved_geometry.slice_spacing_mm or 1.0)
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
    v.series_geometry = saved_geometry
    v._apply_series_capabilities(); v._update_organ_stats(); app.processEvents()


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


def test_sampling_density():
    print("[重建采样密度]")
    import recon
    th = recon.make_theta(180, 180 * 4)
    check(len(th) == 720 and th[-1] < 180, "180° 4× 过采样 = 720 投影且覆盖不变")


def test_compare(v, app):
    print("[双序列随访对比]")
    saved_layout = v.combo_layout.currentIndex()   # 本测试会切布局验 V3，退出前必须还原，
    saved_id = id(v.dicom_datasets)                # 否则污染后续测试的可见视图集合
    # 「配准」只在对比模式下有意义：未加载对比序列时可勾但毫无效果，是误导性控件
    check(not v.chk_register.isEnabled(), "未进对比模式时「配准」禁用")
    # RIDER 副本是 DERIVED 且无 explicit RescaleType=HU；按新单位 contract 应拒绝作为
    # HU follow-up。另用标准 classic synthetic CT 证明成功路径，二者都必须恢复主序列对象。
    vol, dsets = v._read_compare_dir(os.path.join(_ROOT, "肺癌"))
    check(vol is None and dsets == [] and id(v.dicom_datasets) == saved_id,
          "DERIVED/无 explicit HU 的 RIDER compare fail closed，且不污染主序列")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    compare_dir = tempfile.mkdtemp()
    try:
        uid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(compare_dir, f"s{i}.dcm"), (8, 8), uid,
                           ipp_z=i, inst=i + 1)
        vol, dsets = v._read_compare_dir(compare_dir)
        check(vol is not None and len(dsets) == 3 and id(v.dicom_datasets) == saved_id,
              "标准 classic HU compare 成功，且不污染主序列")
    finally:
        shutil.rmtree(compare_dir, ignore_errors=True)
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
    # 「配准」的启停由进出对比模式驱动（上面的用例直接置 compare_mode_active，绕过了
    # 这两个方法，故在此单独走一遍真实入口）
    v._enter_compare_mode(); app.processEvents()
    check(v.chk_register.isEnabled(), "进入对比模式后「配准」启用")
    v._exit_compare_mode(); app.processEvents()
    check(not v.chk_register.isEnabled(), "退出对比模式后「配准」重新禁用")
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
    check(v._export_tag().startswith("ANON-"), "脱敏导出文件名用 per-load 匿名前缀")
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
                   n_frames=1, truncate=False, pix=100, slope=1, intercept=-1024,
                   ipp=None, iop=(1, 0, 0, 0, 1, 0), modality='CT', sop_class_uid=None,
                   pixel_spacing=(1.0, 1.0), pixels=None,
                   image_type=('ORIGINAL', 'PRIMARY', 'AXIAL'), rescale_type=None,
                   multi_energy=None):
    """写一张最小合规的 CT DICOM，供混合形状加载测试使用。ipp_z=None 则不写 ImagePositionPatient。
    empty_numeric=True 时把 RescaleSlope/Intercept/PixelSpacing/SliceThickness 写成空值（None）。
    n_frames>1 写多帧 DICOM；truncate=True 写截断的 PixelData（pixel_array 解码会抛）。
    pix/slope/intercept 给定已知像素值与线性变换，供 HU 转换正确性测试断言 HU=pix*slope+intercept。"""
    import numpy as _np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    sop_class_uid = sop_class_uid or CTImageStorage
    rows, cols = shape
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = pid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = sop_class_uid
    ds.Modality = modality
    if image_type is not None:
        ds.ImageType = list(image_type)
    if rescale_type is not None:
        ds.RescaleType = rescale_type
    if multi_energy is not None:
        # DICOM keyword spelling follows pydicom's data dictionary for (0018,9361).
        ds.MultienergyCTAcquisition = multi_energy
    ds.InstanceNumber = inst
    if ipp is not None:
        ds.ImagePositionPatient = [float(x) for x in ipp]
    elif ipp_z is not None:
        ds.ImagePositionPatient = [0.0, 0.0, float(ipp_z)]
    if iop is not None:
        ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.PixelSpacing = None if empty_numeric else [float(x) for x in pixel_spacing]
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
    if pixels is not None:
        pixel_array = _np.asarray(pixels, dtype=_np.int16)
        if pixel_array.shape != (rows, cols):
            raise ValueError(f"pixels shape {pixel_array.shape} != {(rows, cols)}")
        ds.PixelData = pixel_array.tobytes()
    elif n_frames > 1:
        ds.NumberOfFrames = n_frames
        ds.PixelData = _np.full((n_frames, rows, cols), pix, dtype=_np.int16).tobytes()
    else:
        full = _np.full((rows, cols), pix, dtype=_np.int16).tobytes()
        ds.PixelData = full[:len(full) // 3] if truncate else full
    ds.save_as(path, write_like_original=False)


def test_load_clears_stale_hu_probe(app):
    """成功换序列清掉旧 probe，并为新序列重建 HUD；失败 load 保留旧状态。"""
    print("[DICOM load：成功换序列清 stale probe / 失败保留旧 readout]")
    import shutil
    import tempfile
    import unittest.mock as _mock

    from pydicom.uid import generate_uid
    from PySide6.QtCore import QPointF

    root = tempfile.mkdtemp()
    persistence_dir = tempfile.mkdtemp()
    v = None
    try:
        valid_a = os.path.join(root, "valid-a")
        valid_b = os.path.join(root, "valid-b")
        raw_dir = os.path.join(root, "raw")
        empty_dir = os.path.join(root, "empty")
        for directory in (valid_a, valid_b, raw_dir, empty_dir):
            os.makedirs(directory)
        for directory, uid, pix, image_type in (
                (valid_a, generate_uid(), 100, ('ORIGINAL', 'PRIMARY', 'AXIAL')),
                (valid_b, generate_uid(), 300, ('ORIGINAL', 'PRIMARY', 'AXIAL')),
                (raw_dir, generate_uid(), 700, ('DERIVED', 'SECONDARY', 'PROCESSED'))):
            for i in range(3):
                _write_min_dcm(os.path.join(directory, f"s{i}.dcm"), (8, 8), uid,
                               ipp_z=i, inst=i + 1, pid="STALE_PROBE", pix=pix,
                               image_type=image_type)

        v = m.MedicalViewer(); app.processEvents()
        if v.ai_thread: v.ai_thread.cancel()
        v.persistence_dir = persistence_dir
        v._kickoff_ai = lambda: None
        view_t = type(v.views[1]['view'])

        def leave_real_probe_readout():
            v.views[1]['plane'] = AXIAL
            with _mock.patch.object(view_t, 'get_real_coordinates', lambda _s, _p: (3, 3)):
                v.measure_hu(QPointF(0, 0), 1)
            v._update_hud(*v.current_3d_pos)
            check(bool(v.lbl_hu_value.text()) and "HU" in v.lbl_hu_value.text(),
                  "fixture 通过真实 measure_hu 路径留下旧 HU probe")

        v.load_data(valid_a); app.processEvents()
        leave_real_probe_readout()
        v.load_data(raw_dir); app.processEvents()
        check(v.lbl_hu_value.text() == "",
              "valid HU → 成功 raw load：无需 mouse move 即清空旧 probe")
        check(not v.hu_calibrated and "HU" not in v.lbl_hud.text()
              and ("stored value" in v.lbl_hud.text().lower() or "原始值" in v.lbl_hud.text()),
              f"成功 raw load 为新序列重建 raw HUD（{v.lbl_hud.text()}）")

        v.load_data(valid_a); app.processEvents()
        leave_real_probe_readout()
        v.load_data(valid_b); app.processEvents()
        check(v.hu_calibrated and v.lbl_hu_value.text() == "",
              "valid HU → 成功另一 valid load：清空旧 probe")

        leave_real_probe_readout()
        before_datasets = v.dicom_datasets
        before_volume = v.volume_hu
        before_probe = v.lbl_hu_value.text()
        before_hud = v.lbl_hud.text()
        v.load_data(empty_dir); app.processEvents()
        check(v.dicom_datasets is before_datasets and v.volume_hu is before_volume
              and v.lbl_hu_value.text() == before_probe and v.lbl_hud.text() == before_hud,
              "空目录 load 失败：保留旧 series、probe 与 HUD")
    finally:
        if v is not None:
            if v.ai_thread: v.ai_thread.cancel()
            v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(persistence_dir, ignore_errors=True)


def test_noncanonical_dicom_gating(app):
    """Sagittal classic CT 只能进入可辨识的 viewer-only 状态，不能伪装成 canonical axial。

    这是 patient-space contract 的第一条 tracer bullet：IOP 的 normal 沿 patient +x，
    若产品仍把数组轴当 (S/P, L/R, S/I) canonical volume，静态方位字母、MPR、AI 和
    mL/STL 都会有确定性的语义错误。单层 HU 仍可由逐片 calibration 正确得到，故不应
    因 orientation 无效而把已经校准的 2-D window/probe 一并关闭。
    """
    print("[DICOM contract：non-canonical sagittal viewer-only gating]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    d = tempfile.mkdtemp()
    v = None
    try:
        sid = generate_uid()
        # IOP: columns -> Posterior, rows -> Superior, normal -> Left (+x).
        iop = (0, 1, 0, 0, 0, 1)
        for i, x in enumerate((4.0, 2.0, 0.0), start=1):
            _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (8, 10), sid, ipp_z=None,
                           inst=i, pid=f"SAG_{sid[-8:]}", ipp=(x, 0, 0), iop=iop,
                           pixel_spacing=(2.0, 3.0), pix=500, slope=2, intercept=-1000)
        v = m.MedicalViewer(); app.processEvents()
        kicked = {'n': 0}
        v._kickoff_ai = lambda: kicked.__setitem__('n', kicked['n'] + 1)
        v.load_data(d); app.processEvents()
        flags = tuple(getattr(v, name, None) for name in
                      ('hu_calibrated', 'canonical_orientation',
                       'inplane_spacing_valid', 'uniform_z_geometry_valid'))
        check(v.volume_hu is not None and bool(np.allclose(v.volume_hu, 0.0)),
              "逐片 slope/intercept 仍得到已校准 HU（500×2−1000=0）")
        check(flags == (True, False, True, True),
              f"能力 flags 分轴保存为 HU=True/canonical=False/inplane=True/z=True（得 {flags}）")
        check(kicked['n'] == 0, "non-canonical orientation 不启动器官 AI")
        check(not v.btn_mpr.isEnabled(), "non-canonical orientation 禁用 anatomical MPR")
        first_view = v.views[min(v.views)]
        check(first_view['view'].orient_labels == {}
              and any("原始体素平面" in line
                      for line in first_view['view'].overlay_lines.get('tr', [])),
              "viewer-only 不显示伪 A/P/L/R/S/I 或 Axial/Coronal/Sagittal 声明")
        check(all(not vd['cb_plane'].isEnabled() for vd in v.views.values()),
              "viewer-only 禁用 anatomical plane 切换")
        check(not v.btn_export_stats.isEnabled() and not v.btn_mesh3d.isEnabled(),
              "non-canonical orientation 禁用器官定量与 physical 3-D")
    finally:
        if v is not None:
            if v.ai_thread: v.ai_thread.cancel()
            v.close(); app.processEvents()
        shutil.rmtree(d, ignore_errors=True)


def test_unsupported_dicom_contract(app):
    """非 CT、Enhanced CT 与 multi-frame classic CT 必须在 pixel decode 前拒绝。"""
    print("[DICOM contract：unsupported modality/SOP/multiframe fail closed]")
    import shutil
    import tempfile

    from pydicom.uid import EnhancedCTImageStorage, MRImageStorage, generate_uid
    root = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()
    try:
        cases = (
            ("non_ct", {"modality": "MR", "sop_class_uid": MRImageStorage}),
            ("enhanced_ct", {"sop_class_uid": EnhancedCTImageStorage, "n_frames": 2}),
            ("multiframe_classic", {"n_frames": 2}),
        )
        for name, kwargs in cases:
            case_dir = os.path.join(root, name); os.makedirs(case_dir)
            _write_min_dcm(os.path.join(case_dir, "one.dcm"), (8, 8), generate_uid(),
                           ipp_z=0, inst=1, **kwargs)
            accepted = v._read_dicom_dir(case_dir)
            check(accepted is False,
                  f"{name} 在 _read_dicom_dir contract 阶段拒绝（pixel_array 尚未访问）")
    finally:
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


def test_missing_series_uid_contract(app):
    """无 SeriesInstanceUID 的多文件输入拒绝；单文件无混序风险，允许 viewer-only。"""
    print("[DICOM contract：missing SeriesInstanceUID]")
    import shutil
    import tempfile

    import pydicom
    from pydicom.uid import generate_uid
    root = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()
    try:
        multi = os.path.join(root, "multi"); os.makedirs(multi)
        for i in range(2):
            path = os.path.join(multi, f"s{i}.dcm")
            _write_min_dcm(path, (8, 8), generate_uid(), ipp_z=i, inst=i + 1)
            ds = pydicom.dcmread(path); del ds.SeriesInstanceUID; ds.save_as(path)
        check(v._read_dicom_dir(multi) is False,
              "缺 SeriesInstanceUID 的 multi-file 输入 fail closed（不按空字符串归组）")

        single = os.path.join(root, "single"); os.makedirs(single)
        path = os.path.join(single, "one.dcm")
        _write_min_dcm(path, (8, 8), generate_uid(), ipp_z=0, inst=1)
        ds = pydicom.dcmread(path); del ds.SeriesInstanceUID; ds.save_as(path)
        check(v._read_dicom_dir(single) is True and len(v.dicom_datasets) == 1,
              "单文件缺 SeriesInstanceUID 明确允许（后续因无 z 间距保持 viewer-only）")
    finally:
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


def test_patient_space_geometry_contract():
    """纯函数验证 LPS affine、normal 排序、容差边界与 uniform-z fail-closed。"""
    print("[DICOM geometry：LPS affine / tolerance / z contract]")
    from types import SimpleNamespace

    import dicom_geometry as dg

    axial = (1, 0, 0, 0, 1, 0)

    def ds(z, iop=axial, ipp=True, slope=1, intercept=-1024):
        kw = dict(ImageOrientationPatient=iop, PixelSpacing=(2, 3),
                  RescaleSlope=slope, RescaleIntercept=intercept,
                  ImageType=('ORIGINAL', 'PRIMARY', 'AXIAL'))
        if ipp:
            kw['ImagePositionPatient'] = (0, 0, z)
        return SimpleNamespace(**kw)

    reversed_series = [ds(4, slope=2), ds(0, intercept=-1000), ds(2, slope=3)]
    g = dg.analyze_series(reversed_series)
    check(g.sort_indices == (1, 2, 0) and g.slice_spacing_mm == 2.0,
          f"按 dot(IPP, normal) 统一排序并推导 2mm gap（得 {g.sort_indices}/{g.slice_spacing_mm}）")
    check(g.hu_calibrated and g.canonical_orientation and g.inplane_spacing_valid
          and g.uniform_z_geometry_valid,
          "逐片不同 slope/intercept 合法，四项能力独立为 True")

    eps_in = dg.CANONICAL_ORIENTATION_ATOL * 0.5
    eps_out = dg.CANONICAL_ORIENTATION_ATOL * 2.0
    inside = (1, eps_in, 0, -eps_in, 1, 0)
    outside = (1, eps_out, 0, -eps_out, 1, 0)
    check(dg.analyze_series([ds(0, inside), ds(1, inside)]).canonical_orientation,
          "IOP perturbation 在命名 tolerance 内视为 canonical")
    check(not dg.analyze_series([ds(0, outside), ds(1, outside)]).canonical_orientation,
          "IOP perturbation 越过 tolerance 后不再伪称 canonical")

    inconsistent = dg.analyze_series([ds(0), ds(1), ds(2, (0, 1, 0, 0, 0, 1))])
    duplicate = dg.analyze_series([ds(0), ds(1), ds(1)])
    irregular = dg.analyze_series([ds(0), ds(1), ds(3)])
    missing = dg.analyze_series([ds(0), ds(1, ipp=False)])
    drifted = [ds(0), ds(1), ds(2)]
    for i, item in enumerate(drifted):
        item.ImagePositionPatient = (i * 0.25, 0, i)
    jittered = [ds(0), ds(1), ds(2)]
    for i, item in enumerate(jittered):
        item.ImagePositionPatient = (i * dg.POSITION_ATOL_MM * 0.25, 0, i)
    check(not inconsistent.uniform_z_geometry_valid and not inconsistent.canonical_orientation,
          "slice IOP 不一致关闭 canonical 与 uniform-z")

    # 第二片的 column direction 只偏 5e-5，仍落在 series consistency tolerance 内；
    # 若只验证首片的 norm/dot，它会被错误接受，故必须逐片先验正交性。
    later_nonorthogonal = (1, 0, 0, 5e-5, 1, 0)
    per_slice_bad = dg.analyze_series([ds(0), ds(1, later_nonorthogonal)])
    check(not per_slice_bad.canonical_orientation
          and not per_slice_bad.uniform_z_geometry_valid,
          "首片合法、后片轻微非正交：逐片 IOP 校验 fail closed")
    check(not duplicate.uniform_z_geometry_valid and not irregular.uniform_z_geometry_valid
          and not missing.uniform_z_geometry_valid,
          "重复、不规则或缺失位置均关闭 uniform-z")
    check(not dg.analyze_series(drifted).uniform_z_geometry_valid
          and dg.analyze_series(jittered).uniform_z_geometry_valid,
          "slice origin 面内漂移 fail closed，容差内数值 jitter 不误伤")

    patient_coordinate = getattr(dg, 'patient_coordinate', None)
    edge_labels = getattr(dg, 'voxel_plane_edge_labels', None)
    check(callable(patient_coordinate) and callable(edge_labels),
          "产品 geometry 模块公开 DICOM array→patient LPS 与 edge derivation")
    if callable(patient_coordinate) and callable(edge_labels):
        # 不对称 landmark 位于 array(r=9,c=7)：3mm column direction、2mm row direction。
        point = patient_coordinate((-10, -20, 30), axial, (2, 3), row=9, column=7)
        check(np.allclose(point, (11, -2, 30)),
              f"不对称 landmark 的实际 LPS=(11,-2,30)，即向 Left/Posterior 移动（得 {point}）")
        check(edge_labels(axial) == {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'},
              "由真实 LPS edge 推导 canonical axial 上A/下P/左R/右L")


def test_invalid_calibration_raw_gating(app):
    """任一 slice 无法证明 CT calibration 时，全序列 raw 显示且不产生伪 HU。"""
    print("[DICOM intensity：invalid calibration raw viewer gating]")
    import shutil
    import tempfile

    import pydicom
    from pydicom.uid import generate_uid
    d = tempfile.mkdtemp()
    valid = tempfile.mkdtemp()
    v = None
    try:
        valid_sid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(valid, f"s{i}.dcm"), (8, 8), valid_sid,
                           ipp_z=i, inst=i + 1)
        sid = generate_uid()
        for i in range(3):
            path = os.path.join(d, f"s{i}.dcm")
            _write_min_dcm(path, (8, 8), sid, ipp_z=i, inst=i + 1, pix=100,
                           slope=2, intercept=-1000)
            if i == 1:
                ds = pydicom.dcmread(path); del ds.RescaleSlope; ds.save_as(path)
        v = m.MedicalViewer(); app.processEvents()
        kicked = {'n': 0}; v._kickoff_ai = lambda: kicked.__setitem__('n', kicked['n'] + 1)
        v.load_data(valid); app.processEvents()
        for vdata in v.views.values():
            vdata['preset'].setCurrentIndex(3)  # Bone/骨窗，模拟上一序列留下 named CT preset
        kicked['n'] = 0
        v.load_data(d); app.processEvents()
        check(not v.hu_calibrated and bool(np.all(v.volume_hu == 100)),
              "calibration 不完整时整卷保留 raw stored values，不混合伪 HU")
        v._update_hud(0, 0, 0)
        check("HU" not in v.lbl_hud.text() and ("stored" in v.lbl_hud.text().lower()
                                                or "原始值" in v.lbl_hud.text()),
              f"HUD 明示 raw unit 且不写 HU（{v.lbl_hud.text()}）")
        check(kicked['n'] == 0 and all(not vd['preset'].isEnabled() for vd in v.views.values()),
              "raw 序列不启动 AI，禁用 CT window presets")
        check(all(not button.isEnabled() for button in v.preset_btns),
              "raw 序列同时禁用右侧六个命名 CT preset 按钮")
        check(all(vdata['preset'].currentIndex() == 0 for vdata in v.views.values()),
              "valid HU → invalid HU 时主动清除上一序列的 named CT preset")
        # 纵深防御：disabled combo 仍可被程序化写入旧文本；renderer 必须独立看 capability。
        v.slider_ww.setValue(200); v.slider_wl.setValue(100)
        first = v.views[min(v.views)]
        first['preset'].setCurrentIndex(3)  # 强制残留 Bone/骨窗
        v.update_display(); app.processEvents()
        rendered = first['view'].image_item.pixmap().toImage().pixelColor(0, 0).red()
        check(abs(rendered - 127) <= 1,
              f"raw renderer 忽略 disabled named preset，使用 Global 200/100（pixel={rendered}）")
        check(not v.tool_btns['btn_rec'].isEnabled() and not v.tool_btns['btn_roi'].isEnabled()
              and not v.tool_btns['btn_trk'].isEnabled() and not v.btn_compare.isEnabled(),
              "raw 序列关闭 ROI/HU CSV、HU tracking 与 HU follow-up")
        check(v.tool_btns['btn_rul'].isEnabled() and v.btn_mpr.isEnabled(),
              "HU 无效不误伤仍有有效 in-plane/z geometry 的测距与 MPR")
        check(not v.btn_export_stats.isEnabled() and not v.btn_mesh3d.isEnabled(),
              "raw 序列不产生 organ HU/mL 或 physical STL")
    finally:
        if v is not None:
            if v.ai_thread: v.ai_thread.cancel()
            v.close(); app.processEvents()
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(valid, ignore_errors=True)


def test_hu_unit_semantics_gating(app):
    """标准 HU 必须有逐片单位证据；不支持的 multi-energy 与混合单位整卷 raw。"""
    print("[DICOM intensity：RescaleType/ImageType/multi-energy unit contract]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid

    root = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()

    def make_series(name, *, image_type=('ORIGINAL', 'PRIMARY', 'AXIAL'),
                    rescale_type=None, multi_energy=None, mixed_last_type=None):
        directory = os.path.join(root, name); os.makedirs(directory)
        sid = generate_uid()
        for i in range(3):
            unit = mixed_last_type if i == 2 and mixed_last_type is not None else rescale_type
            _write_min_dcm(os.path.join(directory, f"s{i}.dcm"), (8, 8), sid,
                           ipp_z=i, inst=i + 1, pix=100, slope=1, intercept=-1024,
                           image_type=image_type, rescale_type=unit,
                           multi_energy=multi_energy)
        return directory

    try:
        v._kickoff_ai = lambda: None
        classic = make_series("classic")
        explicit_hu = make_series(
            "explicit_hu", image_type=('DERIVED', 'PRIMARY', 'AXIAL'), rescale_type=' hu ')
        missing_image_type = make_series("missing_image_type", image_type=None)
        derived = make_series("derived", image_type=('DERIVED', 'PRIMARY', 'AXIAL'))
        localizer = make_series("localizer", image_type=('ORIGINAL', 'PRIMARY', 'LOCALIZER'))
        multi = make_series("multi", rescale_type='HU', multi_energy='YES')
        mixed = make_series("mixed", mixed_last_type='ED')
        non_hu = {
            unit: make_series(f"unit_{unit}", rescale_type=unit)
            for unit in ('Z_EFF', 'ED', 'EDW', 'MGML', 'HU_MOD', 'PCT', 'US')
        }

        for label, directory in (("classic missing RescaleType", classic),
                                 ("explicit normalized HU", explicit_hu)):
            v.load_data(directory); app.processEvents()
            check(v.hu_calibrated and float(v.volume_hu[0, 0, 0]) == -924.0,
                  f"{label}：有标准 HU 证据并应用 slope/intercept")

        rejected = {
            "missing ImageType": missing_image_type,
            "DERIVED without explicit HU": derived,
            "LOCALIZER without explicit HU": localizer,
            "unsupported multi-energy": multi,
            "mixed-slice ED": mixed,
            **{f"RescaleType={unit}": directory for unit, directory in non_hu.items()},
        }
        for label, directory in rejected.items():
            v.load_data(directory); app.processEvents()
            check(not v.hu_calibrated and bool(np.all(v.volume_hu == 100)),
                  f"{label}：整卷 raw，不把 stored values 伪称 HU")
            check(all(not vd['preset'].isEnabled() for vd in v.views.values())
                  and not v.tool_btns['btn_rec'].isEnabled()
                  and not v.tool_btns['btn_roi'].isEnabled()
                  and not v.btn_compare.isEnabled(),
                  f"{label}：产品路径关闭 CT preset/AI/HU ROI/compare")

        compare_volume, compare_datasets = v._read_compare_dir(non_hu['Z_EFF'])
        check(compare_volume is None and compare_datasets == [],
              "compare loader 复用 HU unit contract，拒绝 Z_EFF follow-up")
    finally:
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


def test_spacing_capability_gating(app):
    """in-plane 与 z geometry 分轴 gating，overlay 不得填入 1mm/px×3 等伪单位。"""
    print("[DICOM spacing：independent capability gating / no fake units]")
    import shutil
    import tempfile

    import pydicom
    from pydicom.uid import generate_uid
    root = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()
    try:
        valid = os.path.join(root, "valid"); os.makedirs(valid)
        valid_sid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(valid, f"s{i}.dcm"), (8, 8), valid_sid,
                           ipp_z=i * 2, inst=i + 1)
        v._kickoff_ai = lambda: None
        v.load_data(valid); app.processEvents()
        v.tool_btns['btn_rul'].setChecked(True)
        v.change_active_tool(TOOL_RULER)
        measure_view = v.views[min(v.views)]['view']
        measure_view.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(10, 10), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        measure_view.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, QPointF(30, 10), Qt.NoButton,
            Qt.LeftButton, Qt.NoModifier))
        check(measure_view.is_drawing and "mm" in measure_view.temp_text.toPlainText(),
              "fixture 真实经过 Ruler press→move 并产生 mm preview")

        no_px = os.path.join(root, "no_px"); os.makedirs(no_px)
        sid = generate_uid()
        for i in range(3):
            path = os.path.join(no_px, f"s{i}.dcm")
            _write_min_dcm(path, (8, 8), sid, ipp_z=i * 2, inst=i + 1)
            ds = pydicom.dcmread(path); del ds.PixelSpacing; ds.save_as(path)
        v.load_data(no_px); app.processEvents()
        br = v.views[min(v.views)]['view'].overlay_lines.get('br', [])
        check(v.hu_calibrated and v.canonical_orientation
              and not v.inplane_spacing_valid and v.uniform_z_geometry_valid,
              "缺 PixelSpacing 只关闭 in-plane capability，不误伤 HU/canonical/z")
        check(not v.tool_btns['btn_rul'].isEnabled() and not v.tool_btns['btn_roi'].isEnabled()
              and not v.btn_mpr.isEnabled(),
              "缺 in-plane spacing 关闭 mm/mm² 与 physical MPR")
        check(all(not vd['cb_plane'].isEnabled() for vd in v.views.values()),
              "缺 in-plane spacing 时 plane selector 不能绕过 MPR geometry gate")
        check(v.active_tool == TOOL_POINTER and v.tool_btns['btn_ptr'].isChecked()
              and not v.tool_btns['btn_rul'].isChecked()
              and all(vd['view'].current_tool == TOOL_POINTER for vd in v.views.values()),
              "valid spacing → invalid spacing 同步撤销 Ruler button/global/view 状态")
        check(not measure_view.is_drawing
              and not any(isinstance(item, QGraphicsTextItem) and "mm" in item.toPlainText()
                          for item in measure_view.scene.items()),
              "spacing 失效时取消进行中的 measurement preview，不留下伪 mm")
        check(any("Px unavailable" in x or "像素间距不可用" in x for x in br)
              and not any("Px 1.00mm" in x for x in br),
              f"overlay 明示 PixelSpacing unavailable，不伪造 1mm（{br}）")

        irregular = os.path.join(root, "irregular"); os.makedirs(irregular)
        sid = generate_uid()
        for i, z in enumerate((0, 1, 3), start=1):
            _write_min_dcm(os.path.join(irregular, f"s{i}.dcm"), (8, 8), sid,
                           ipp_z=z, inst=i, pixel_spacing=(0.7, 0.9))
        v.load_data(irregular); app.processEvents()
        br = v.views[min(v.views)]['view'].overlay_lines.get('br', [])
        check(v.inplane_spacing_valid and not v.uniform_z_geometry_valid,
              "不规则 projected gaps 保留 in-plane，关闭 uniform-z")
        check(v.tool_btns['btn_rul'].isEnabled() and v.tool_btns['btn_roi'].isEnabled()
              and not v.btn_mpr.isEnabled(),
              "不规则 z 仍允许有效 axial mm/mm²/HU，关闭 z-dependent MPR")
        check(any("Z spacing unavailable" in x or "层间距不可用" in x for x in br)
              and not any("Thk 2.1mm" in x for x in br),
              f"overlay 明示 z spacing unavailable，不用 px×3/SliceThickness 伪装（{br}）")
        check(not v.btn_export_stats.isEnabled() and not v.btn_mesh3d.isEnabled(),
              "不规则 z 不产生 mL 或 physical STL")
    finally:
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


def test_compare_dicom_contract(app):
    """follow-up loader 复用同一 classic CT/geometry/calibration contract，且不污染主序列。"""
    print("[DICOM compare：same contract / primary state isolation]")
    import shutil
    import tempfile

    from pydicom.uid import EnhancedCTImageStorage, generate_uid
    root = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()
    try:
        primary = os.path.join(root, "primary"); os.makedirs(primary)
        sid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(primary, f"s{i}.dcm"), (8, 8), sid,
                           ipp_z=i, inst=i + 1)
        v._kickoff_ai = lambda: None
        v.load_data(primary); app.processEvents()
        primary_ids = tuple(ds.SOPInstanceUID for ds in v.dicom_datasets)
        primary_geometry = v.series_geometry

        enhanced = os.path.join(root, "enhanced"); os.makedirs(enhanced)
        _write_min_dcm(os.path.join(enhanced, "e.dcm"), (8, 8), generate_uid(),
                       ipp_z=0, inst=1, n_frames=2, sop_class_uid=EnhancedCTImageStorage)
        vol, dsets = v._read_compare_dir(enhanced)
        check(vol is None and dsets == [], "compare 在 decode 前同样拒绝 Enhanced/multiframe")

        irregular = os.path.join(root, "irregular"); os.makedirs(irregular)
        sid = generate_uid()
        for i, z in enumerate((0, 1, 3), start=1):
            _write_min_dcm(os.path.join(irregular, f"s{i}.dcm"), (8, 8), sid,
                           ipp_z=z, inst=i)
        vol, dsets = v._read_compare_dir(irregular)
        check(vol is None and dsets == [],
              "compare 拒绝无法满足 anatomical follow-up 的 irregular-z series")
        check(tuple(ds.SOPInstanceUID for ds in v.dicom_datasets) == primary_ids
              and v.series_geometry == primary_geometry,
              "失败的 compare load 不污染主序列 datasets/geometry contract")

        valid = os.path.join(root, "valid"); os.makedirs(valid)
        sid = generate_uid()
        for inst, z, slope, intercept, _want in ((1, 2, 1, -1000, -900),
                                                 (2, 0, 2, -1000, -800),
                                                 (3, 1, 3, -1000, -700)):
            _write_min_dcm(os.path.join(valid, f"s{inst}.dcm"), (8, 8), sid,
                           ipp_z=z, inst=inst, pix=100, slope=slope, intercept=intercept)
        vol, dsets = v._read_compare_dir(valid)
        check(vol is not None and tuple(float(x) for x in vol[:, 0, 0]) == (-800, -700, -900),
              "valid compare 按 patient-space 顺序并逐 slice 应用 calibration")
        check(tuple(ds.SOPInstanceUID for ds in v.dicom_datasets) == primary_ids
              and v.series_geometry == primary_geometry,
              "成功的 compare load 也恢复主序列 datasets/geometry contract")

        from types import SimpleNamespace
        sagittal = [SimpleNamespace(ImageOrientationPatient=(0, 1, 0, 0, 0, 1),
                                    ImagePositionPatient=(x, 9, 11))
                    for x in (5, 7)]
        check(np.array_equal(v._zpos_array(sagittal), np.array((5.0, 7.0))),
              "compare 配准坐标使用 dot(IPP, normal)，不写死 IPP[2]")
    finally:
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


def test_deid_export_and_persistence_contract(app):
    """匿名显式导出用 per-load nonce 且不覆盖；内部缓存另行警告仍含 identifiers。"""
    print("[De-ID：per-load nonce / unique export / persistence warning]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    from PySide6.QtWidgets import QMessageBox
    root = tempfile.mkdtemp(); out = tempfile.mkdtemp(); internal = tempfile.mkdtemp()
    v = m.MedicalViewer(); app.processEvents()
    saved_warning = QMessageBox.warning
    warnings = []
    try:
        dirs = []
        for tag in ("a", "b"):
            d = os.path.join(root, tag); os.makedirs(d); dirs.append(d)
            sid = generate_uid()
            for i in range(3):
                _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (8, 8), sid,
                               ipp_z=i, inst=i + 1, pid="SECRET_PATIENT")
        v._kickoff_ai = lambda: None
        v.load_data(dirs[0]); app.processEvents(); v._toggle_anonymize(True)
        tag_a1, tag_a2 = v._export_tag(), v._export_tag()
        uid_a = str(v.dicom_datasets[0].SeriesInstanceUID)
        v.load_data(dirs[1]); app.processEvents()
        tag_b = v._export_tag()
        check(tag_a1 == tag_a2 and tag_a1.startswith("ANON-") and tag_a1 != tag_b,
              f"匿名 nonce 同 load 稳定、跨 load 改变（{tag_a1} → {tag_b}）")
        check("SECRET" not in tag_a1 and uid_a not in tag_a1,
              "匿名 tag 不含 PatientID、原始 UID 或其裸露片段")

        v.export_dir = out
        v._organ_stats = [{
            'id': 5, 'name_zh': '肝', 'name_en': 'Liver', 'voxels': 8,
            'volume_ml': 1.0, 'mean_hu': 50.0, 'sd_hu': 2.0, 'median_hu': 50.0,
            'p5_hu': 47.0, 'p95_hu': 53.0, 'min_hu': 45.0, 'max_hu': 55.0,
        }]
        v.export_organ_stats(); v.export_organ_stats()
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        v._export_stl("Liver", verts, faces); v._export_stl("Liver", verts, faces)
        names = sorted(os.listdir(out))
        check(len(names) == 4 and len(set(names)) == 4,
              f"连续同类型匿名导出生成 4 个唯一文件，不静默覆盖（{names}）")
        check(all(name.startswith(tag_b) and "SECRET" not in name and uid_a not in name
                  for name in names),
              "quantification CSV/STL 文件名均使用当前 session nonce，不含身份/UID")

        QMessageBox.warning = staticmethod(lambda _p, title, msg, *a, **k: warnings.append((title, msg)))
        v.persistence_dir = internal
        v.global_annotations = {'all': []}
        v.volume_mask = np.ones_like(v.volume_hu, np.uint8)
        v.save_project()
        check(bool(warnings) and any("identifier" in msg.lower() or "标识" in msg
                                     for _title, msg in warnings),
              "De-ID 开启时 save_project 明示内部 JSON/NPZ 仍含身份/序列标识")
        tip = v.chk_anon.toolTip().lower()
        check("burned-in" in tip or "烧录" in tip,
              "De-ID UI 保留 burned-in pixel text 不会自动清除的提示")
    finally:
        QMessageBox.warning = saved_warning
        if v.ai_thread: v.ai_thread.cancel()
        v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(internal, ignore_errors=True)


def test_save_project_atomic_contract(app):
    """save_project 的 precondition 与 per-target atomicity 只在临时目录验证。"""
    print("[Project persistence：fingerprint precondition / atomic targets]")
    import json
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    from PySide6.QtWidgets import QMessageBox

    import annotation_lab

    dicom_dir = tempfile.mkdtemp()
    persistence_dir = tempfile.mkdtemp()
    v = None
    saved_information = QMessageBox.information
    saved_warning = QMessageBox.warning
    saved_question = QMessageBox.question
    extra_viewers = []
    infos, warnings = [], []
    try:
        sid = generate_uid()
        for i in range(3):
            _write_min_dcm(os.path.join(dicom_dir, f"s{i}.dcm"), (8, 8), sid,
                           ipp_z=i, inst=i + 1, pid="SAVE_ATOMIC")
        v = m.MedicalViewer(); app.processEvents()
        v._kickoff_ai = lambda: None
        v.load_data(dicom_dir); app.processEvents()
        v.persistence_dir = persistence_dir
        v.global_annotations = {'all': []}
        v.volume_mask = np.ones_like(v.volume_hu, dtype=np.uint8)

        json_path = os.path.join(persistence_dir, "SAVE_ATOMIC_annotations.json")
        npz_path = os.path.join(persistence_dir, "SAVE_ATOMIC_mask.npz")
        json_sentinel = b"SENTINEL-JSON"
        npz_sentinel = b"SENTINEL-NPZ"
        with open(json_path, "wb") as f:
            f.write(json_sentinel)
        with open(npz_path, "wb") as f:
            f.write(npz_sentinel)

        QMessageBox.information = staticmethod(
            lambda _p, title, msg, *a, **k: infos.append((title, msg)))
        QMessageBox.warning = staticmethod(
            lambda _p, title, msg, *a, **k: warnings.append((title, msg)))
        actual_uid = v._current_series_uid()
        actual_fingerprint = v._current_geometry_fingerprint()

        v._current_geometry_fingerprint = lambda: ""
        result = v.save_project()
        check(result is False
              and open(json_path, "rb").read() == json_sentinel
              and open(npz_path, "rb").read() == npz_sentinel,
              "empty fingerprint fail closed，既有 JSON/NPZ sentinel bytes 不变")
        check(not infos and bool(warnings),
              "precondition failure 不显示 Project saved，并给出明确 warning")

        # UID 与 fingerprint 都是恢复缓存所需的身份/几何 provenance；任一为空均不得覆盖。
        infos.clear(); warnings.clear()
        v._current_geometry_fingerprint = lambda: actual_fingerprint
        v._current_series_uid = lambda: ""
        result = v.save_project()
        check(result is False
              and open(json_path, "rb").read() == json_sentinel
              and open(npz_path, "rb").read() == npz_sentinel
              and not infos and bool(warnings),
              "empty SeriesInstanceUID fail closed，既有目标不变且无 success")

        v._current_series_uid = lambda: actual_uid

        # JSON 序列化先写同目录临时文件；失败不得截断既有 JSON，也不得提前替换 NPZ。
        infos.clear(); warnings.clear()
        saved_json_dump = annotation_lab.json.dump
        annotation_lab.json.dump = lambda *a, **k: (_ for _ in ()).throw(
            OSError("injected JSON serialization failure"))
        try:
            result = v.save_project()
        finally:
            annotation_lab.json.dump = saved_json_dump
        leftovers = [name for name in os.listdir(persistence_dir)
                     if name.startswith(".SAVE_ATOMIC_")]
        check(result is False
              and open(json_path, "rb").read() == json_sentinel
              and open(npz_path, "rb").read() == npz_sentinel
              and not leftovers,
              "JSON serialization failure 保留两份 sentinel，并清理临时文件")
        check(not infos and bool(warnings),
              "JSON failure 返回 False、不给 success、给出 warning")

        # NPZ 写入也发生在任何 replace 之前；故第二目标写失败时 JSON 目标仍不变。
        infos.clear(); warnings.clear()
        saved_savez = annotation_lab.np.savez_compressed
        annotation_lab.np.savez_compressed = lambda *a, **k: (_ for _ in ()).throw(
            OSError("injected NPZ write failure"))
        try:
            result = v.save_project()
        finally:
            annotation_lab.np.savez_compressed = saved_savez
        leftovers = [name for name in os.listdir(persistence_dir)
                     if name.startswith(".SAVE_ATOMIC_")]
        check(result is False
              and open(json_path, "rb").read() == json_sentinel
              and open(npz_path, "rb").read() == npz_sentinel
              and not leftovers,
              "NPZ write failure 发生在 replace 前，两份 sentinel 均不变")
        check(not infos and bool(warnings),
              "NPZ failure 返回 False、不给 success、给出 warning")

        infos.clear(); warnings.clear()
        result = v.save_project()
        with open(json_path, encoding="utf-8") as f:
            saved_json = json.load(f)
        with np.load(npz_path) as saved_npz:
            saved_mask = saved_npz["mask"]
            saved_uid = str(saved_npz["series_uid"].item())
            saved_fingerprint = str(saved_npz["geometry_fingerprint"].item())
        check(result is True and bool(infos) and not warnings,
              "成功路径仅在 JSON/NPZ 均替换后返回 True 并显示 success")
        check(saved_json["__meta__"]["series_uid"] == actual_uid
              and saved_json["__meta__"]["geometry_fingerprint"] == actual_fingerprint
              and saved_uid == actual_uid and saved_fingerprint == actual_fingerprint
              and np.array_equal(saved_mask, v.volume_mask),
              "成功落盘的 JSON/NPZ 同时绑定当前 UID、fingerprint 与 mask bytes")

        # 两个 os.replace 不是跨文件事务：第二个失败时必须准确报告已替换/未替换目标，
        # 返回 False 且不显示完整成功；不得把 partial completion 冒充 Project saved。
        with open(json_path, "wb") as f:
            f.write(json_sentinel)
        with open(npz_path, "wb") as f:
            f.write(npz_sentinel)
        infos.clear(); warnings.clear()
        saved_replace = annotation_lab.os.replace

        def fail_npz_replace(src, dst):
            if dst == npz_path:
                raise OSError("injected second-target replace failure")
            return saved_replace(src, dst)

        annotation_lab.os.replace = fail_npz_replace
        try:
            result = v.save_project()
        finally:
            annotation_lab.os.replace = saved_replace
        warning_text = "\n".join(msg for _title, msg in warnings)
        check(result is False and not infos
              and open(json_path, "rb").read() != json_sentinel
              and open(npz_path, "rb").read() == npz_sentinel,
              "第二个 replace 失败：JSON 已替换、NPZ 保持 sentinel、整体返回 False")
        check(os.path.basename(json_path) in warning_text
              and os.path.basename(npz_path) in warning_text
              and ("cross-file" in warning_text or "跨文件" in warning_text),
              "partial replace warning 明列成功/失败目标并声明无跨文件原子性")

        # fresh placeholder zero 只是 AI pending 的占位，不得落成可命中的全零 cache。
        for target in (json_path, npz_path):
            if os.path.exists(target):
                os.unlink(target)
        infos.clear(); warnings.clear()
        v.volume_mask = np.zeros_like(v.volume_hu, dtype=np.uint8)
        v._ai_state = 'running'
        result = v.save_project()
        check(result is True and os.path.exists(json_path) and not os.path.exists(npz_path),
              "fresh placeholder zero + AI running：只保存 annotations，不制造零 NPZ")
        placeholder_kickoffs = {'n': 0}
        vp = m.MedicalViewer(); extra_viewers.append(vp); app.processEvents()
        if vp.ai_thread: vp.ai_thread.cancel()
        vp.persistence_dir = persistence_dir
        vp._kickoff_ai = lambda: placeholder_kickoffs.__setitem__('n',
                                                                  placeholder_kickoffs['n'] + 1)
        vp.load_data(dicom_dir); app.processEvents()
        check(placeholder_kickoffs['n'] == 1 and not getattr(
                  vp, '_mask_cache_clear_requested', False),
              "placeholder zero 重开仍 cache miss，不跳过 AI kickoff")

        # 先持久化一份真实非零 cache，再由新 viewer 恢复，并通过真实清空入口确认 empty。
        v.volume_mask = np.zeros_like(v.volume_hu, dtype=np.uint8)
        v.volume_mask[0, 1:3, 1:3] = 5
        infos.clear(); warnings.clear()
        check(v.save_project() is True, "fixture 保存匹配当前 geometry 的非零 cache")
        restored_kickoffs = {'n': 0}
        vc = m.MedicalViewer(); extra_viewers.append(vc); app.processEvents()
        if vc.ai_thread: vc.ai_thread.cancel()
        vc.persistence_dir = persistence_dir
        vc._kickoff_ai = lambda: restored_kickoffs.__setitem__('n', restored_kickoffs['n'] + 1)
        vc.load_data(dicom_dir); app.processEvents()
        check(bool(vc.volume_mask.any()) and restored_kickoffs['n'] == 0
              and not getattr(vc, '_mask_cache_clear_requested', False),
              "匹配的非零 cache 恢复成功，且恢复状态不是 pending clear")

        class _RunningAI:
            def __init__(self):
                self.cancelled = False
                self.resampled_from = None
                self.used_fallback = False
                self.confidence = None
            def isRunning(self): return True
            def cancel(self): self.cancelled = True

        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        running = _RunningAI()
        vc.ai_thread = running
        vc._ai_state = 'running'
        old_generation = vc._ai_generation
        vc.clear_mask_and_annotations(); app.processEvents()
        check(not vc.volume_mask.any()
              and getattr(vc, '_mask_cache_clear_requested', False),
              "真实全局清空入口把已有非零 mask 标为 explicit empty")
        check(running.cancelled and vc._ai_generation > old_generation,
              "explicit clear 取消并作废仍可能回调的旧 AI generation")
        stale_mask = np.full_like(vc.volume_mask, 9, dtype=np.uint8)
        vc.on_auto_ai_finished(stale_mask, 1.0, old_generation); app.processEvents()
        check(not vc.volume_mask.any(), "explicit clear 后旧 AI callback 不能覆盖 empty mask")
        vc.volume_mask.fill(0)  # 保持后续断言独立；正确实现上这是 no-op。

        # load 失败不得清掉尚未持久化的 explicit-empty intent。
        empty_dir = os.path.join(persistence_dir, "empty-load")
        os.makedirs(empty_dir, exist_ok=True)
        vc.load_data(empty_dir); app.processEvents()
        check(getattr(vc, '_mask_cache_clear_requested', False),
              "explicit clear 后 load 失败：pending clear intent 保留")

        infos.clear(); warnings.clear()
        result = vc.save_project()
        with np.load(npz_path) as explicit_npz:
            explicit_mask = explicit_npz['mask']
            explicit_uid = str(explicit_npz['series_uid'].item())
            explicit_fingerprint = str(explicit_npz['geometry_fingerprint'].item())
        check(result is True and not explicit_mask.any()
              and explicit_uid == actual_uid and explicit_fingerprint == actual_fingerprint,
              "explicit empty 保存全零 NPZ，并绑定当前 UID/fingerprint")
        check(not getattr(vc, '_mask_cache_clear_requested', True),
              "explicit-empty 成功保存后清除 pending intent")

        zero_kickoffs = {'n': 0}
        vr = m.MedicalViewer(); extra_viewers.append(vr); app.processEvents()
        if vr.ai_thread: vr.ai_thread.cancel()
        vr.persistence_dir = persistence_dir
        vr._kickoff_ai = lambda: zero_kickoffs.__setitem__('n', zero_kickoffs['n'] + 1)
        vr.load_data(dicom_dir); app.processEvents()
        check(not vr.volume_mask.any() and zero_kickoffs['n'] == 0,
              "重载 explicit-empty cache 仍为全零，旧标签不复活且不重跑 AI")

        # clear → Ctrl+Z 撤销恢复非零时，不能再把本次保存当作 explicit empty。
        vr.volume_mask[1, 2:4, 2:4] = 7
        undo_expected = vr.volume_mask.copy()
        vr.clear_mask_and_annotations(); app.processEvents()
        check(getattr(vr, '_mask_cache_clear_requested', False),
              "再次 explicit clear 建立 pending intent")
        vr._undo_mask_edit(); app.processEvents()
        check(np.array_equal(vr.volume_mask, undo_expected)
              and not getattr(vr, '_mask_cache_clear_requested', True),
              "clear → Ctrl+Z 恢复非零 mask，并撤销 pending clear intent")
        infos.clear(); warnings.clear()
        check(vr.save_project() is True, "undo 后非零 mask 正常保存")
        with np.load(npz_path) as undo_npz:
            check(np.array_equal(undo_npz['mask'], undo_expected),
                  "undo 后保存的是恢复的非零 mask，不是 explicit empty")

        # None 与 fresh zero 都不是 explicit clear；wrong-shape zero 也必须先 fail closed。
        none_dir = os.path.join(persistence_dir, "none-mask")
        os.makedirs(none_dir, exist_ok=True)
        v.persistence_dir = none_dir
        v.volume_mask = None
        infos.clear(); warnings.clear()
        check(v.save_project() is True
              and not os.path.exists(os.path.join(none_dir, "SAVE_ATOMIC_mask.npz")),
              "volume_mask is None：只保存 annotations，不创建零 NPZ")

        wrong_dir = os.path.join(persistence_dir, "wrong-shape")
        os.makedirs(wrong_dir, exist_ok=True)
        wrong_json = os.path.join(wrong_dir, "SAVE_ATOMIC_annotations.json")
        wrong_npz = os.path.join(wrong_dir, "SAVE_ATOMIC_mask.npz")
        with open(wrong_json, 'wb') as f: f.write(json_sentinel)
        with open(wrong_npz, 'wb') as f: f.write(npz_sentinel)
        v.persistence_dir = wrong_dir
        v.volume_mask = np.zeros((1, 1, 1), dtype=np.uint8)
        infos.clear(); warnings.clear()
        result = v.save_project()
        check(result is False and not infos and bool(warnings)
              and open(wrong_json, 'rb').read() == json_sentinel
              and open(wrong_npz, 'rb').read() == npz_sentinel,
              "wrong-shape zero fail closed，既有 JSON/NPZ bytes 不变")

        # explicit-empty 的 NPZ 序列化与最终替换失败都必须保留 intent，供用户重试。
        failure_dir = os.path.join(persistence_dir, "explicit-failure")
        os.makedirs(failure_dir, exist_ok=True)
        failure_json = os.path.join(failure_dir, "SAVE_ATOMIC_annotations.json")
        failure_npz = os.path.join(failure_dir, "SAVE_ATOMIC_mask.npz")
        vr.persistence_dir = failure_dir
        vr.volume_mask = undo_expected.copy()
        vr.clear_mask_and_annotations(); app.processEvents()
        with open(failure_json, 'wb') as f: f.write(json_sentinel)
        with open(failure_npz, 'wb') as f: f.write(npz_sentinel)
        infos.clear(); warnings.clear()
        saved_savez = annotation_lab.np.savez_compressed
        annotation_lab.np.savez_compressed = lambda *a, **k: (_ for _ in ()).throw(
            OSError("injected explicit-empty NPZ serialization failure"))
        try:
            result = vr.save_project()
        finally:
            annotation_lab.np.savez_compressed = saved_savez
        check(result is False and not infos and bool(warnings)
              and open(failure_json, 'rb').read() == json_sentinel
              and open(failure_npz, 'rb').read() == npz_sentinel
              and getattr(vr, '_mask_cache_clear_requested', False),
              "explicit-empty NPZ serialization failure：目标不变、无 success、intent 保留")

        with open(failure_json, 'wb') as f: f.write(json_sentinel)
        with open(failure_npz, 'wb') as f: f.write(npz_sentinel)
        infos.clear(); warnings.clear()
        saved_replace = annotation_lab.os.replace

        def fail_explicit_npz_replace(src, dst):
            if dst == failure_npz:
                raise OSError("injected explicit-empty NPZ replace failure")
            return saved_replace(src, dst)

        annotation_lab.os.replace = fail_explicit_npz_replace
        try:
            result = vr.save_project()
        finally:
            annotation_lab.os.replace = saved_replace
        check(result is False and not infos and bool(warnings)
              and open(failure_json, 'rb').read() != json_sentinel
              and open(failure_npz, 'rb').read() == npz_sentinel
              and getattr(vr, '_mask_cache_clear_requested', False),
              "explicit-empty NPZ replace failure：准确 partial failure、intent 保留")
    finally:
        QMessageBox.information = saved_information
        QMessageBox.warning = saved_warning
        QMessageBox.question = saved_question
        for extra in extra_viewers:
            if extra.ai_thread: extra.ai_thread.cancel()
            extra.close()
        if v is not None:
            if v.ai_thread: v.ai_thread.cancel()
            v.close(); app.processEvents()
        shutil.rmtree(dicom_dir, ignore_errors=True)
        shutil.rmtree(persistence_dir, ignore_errors=True)


def test_mixed_shape_dicom(app):
    """同一有效 SeriesUID 内切片形状不一致时，保留多数尺寸且不崩。"""
    print("[混合形状 DICOM 加载防护]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    v2 = m.MedicalViewer(); app.processEvents()
    if v2.ai_thread:
        v2.ai_thread.cancel()
    sid = generate_uid()
    cases = [
        ("同序列混合形状", [((512, 512), sid), ((512, 512), sid),
                            ((512, 512), sid), ((256, 256), sid)], (512, 512), 3),
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


def test_dl_recon_guard():
    """学习式重建的可用性守卫与推理契约（纯函数，无 Qt / 真实数据）。

    重点不是「模型好不好」——那是 experiments/recon_dl.py 的事——而是：
    模型或 onnxruntime 缺失时必须如实返回不可用（调用方据此保持按钮禁用），
    而不是让用户点一个会报错的按钮；模型在场时推理须保形状、保有限、保值域。
    """
    print("[学习式重建守卫]")
    import recon as R
    from constants import RECON_DL_MODEL, RECON_DL_VIEWS
    check(RECON_DL_VIEWS == 20, f"模型训练视角常量为 20（得 {RECON_DL_VIEWS}）")
    check(R.dl_available("/nonexistent/model.onnx") is False, "模型文件不存在 → 报告不可用")
    check(R.dl_available("") is False, "空路径 → 报告不可用")
    have = R.dl_available(RECON_DL_MODEL)
    print(f"    本机模型就绪={have}（缺失时下列推理断言自动跳过，不算失败）")
    if not have:
        return
    rng = np.random.RandomState(0)
    for shape, tag in (((128, 128), '128²·训练尺寸'), ((64, 64), '64²·更小'),
                       ((100, 100), '100²·非8倍数')):
        x = rng.rand(*shape).astype(np.float32)
        out, ms = R.compute_dl_recon(x, RECON_DL_MODEL)
        check(out.shape == shape, f"{tag} 输出形状与输入一致（得 {out.shape}）")
        check(bool(np.isfinite(out).all()), f"{tag} 输出全为有限值")
    # 值域必须【还原到输入的量级】，不能只验「没跑飞」——初版断言写成 -3000<out<3000，
    # 而实现里误用了 _finite_clip（为 DMR/ART 而写，硬 clip 到 [0,1]），输出被压成
    # [0,1] 却照样满足那个宽松区间，bug 就这样被放过了。显示时又会重新归一化，肉眼看不出。
    hu = (rng.rand(64, 64).astype(np.float32) * 1400 - 1000)
    out, _ = R.compute_dl_recon(hu, RECON_DL_MODEL)
    check(bool(np.isfinite(out).all()), "HU 量级输入输出全有限")
    check(out.min() < -100 and out.max() > 100,
          f"HU 输入的输出仍在 HU 量级，未被压到 [0,1]（得 [{out.min():.0f}, {out.max():.0f}]）")
    # 与输入量级同阶：允许网络改变数值，但不该整体漂移一个数量级
    check(abs(out.mean() - hu.mean()) < abs(hu.max() - hu.min()),
          f"输出均值与输入同阶（输入 {hu.mean():.0f} vs 输出 {out.mean():.0f}）")
    # NaN 防护：与 compute_fbp/compute_sinogram 一致，NaN 经卷积会扩散到整幅输出
    bad = rng.rand(64, 64).astype(np.float32); bad[10:14, 10:14] = np.nan
    out, _ = R.compute_dl_recon(bad, RECON_DL_MODEL)
    check(bool(np.isfinite(out).all()), "含 NaN 输入仍产出全有限结果")
    # 会话缓存：第二次调用不应重建 InferenceSession（否则每次重建白等数百毫秒）
    import time as _t
    x = rng.rand(128, 128).astype(np.float32)
    R.compute_dl_recon(x, RECON_DL_MODEL)
    t0 = _t.perf_counter(); R.compute_dl_recon(x, RECON_DL_MODEL)
    t1 = _t.perf_counter(); R.compute_dl_recon(x, RECON_DL_MODEL)
    t2 = _t.perf_counter()
    check(abs((t2 - t1) - (t1 - t0)) < max(t1 - t0, 1e-3) * 3,
          "重复调用耗时稳定（会话已缓存，未每次重建）")


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
    # 键由 build_system_matrix 自己给出，不在测试里复刻它的构成——此前这里手拼
    # (n, 角度数, 首, 尾)，于是缓存键的定义改动测试无法发现，而「首尾相同、中间不同」
    # 的角度表撞键这个缺陷，恰恰就藏在那个定义里。
    _, key = R.build_system_matrix(n_s, theta_s, A, None)
    A_c, key_c = R.build_system_matrix(n_s, theta_s, A, key)
    check(A_c is A and key_c == key, "build_system_matrix 命中内存缓存直接复用（不重建、不起子进程）")
    # 长度、首、尾全同但中间不同的两张角度表必须给出不同的键，否则会静默复用错矩阵
    ta = np.array([0, 60, 120, 150, 170], float)
    tb = np.array([0, 5, 10, 15, 170], float)
    ka = R.build_system_matrix(n_s, ta, A, None)[1]
    kb = R.build_system_matrix(n_s, tb, A, None)[1]
    check(ka != kb, f"首尾相同、中间不同的角度表不撞键（{ka} vs {kb}）")
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


def test_asdpocs_numerics():
    """ASD-POCS（TV 正则化）纯函数单测——合成小系统，无 Qt / 真实数据。

    由来：ASD-POCS 的三个易错点都不会让代码报错，只会让它安静地变成另一个算法：
    dtvg 每轮重新初始化（永不退火）、TV 步不做 L2 归一化（α 绑死在 A/p 的尺度上）、
    返回 TV 步之后的图而非 POCS 步之后的 f_res。故逐条锁死。
    """
    print("[ASD-POCS / TV 正则化重建]")
    import numpy as np

    import recon as R
    n_s = 12
    img = np.zeros((n_s, n_s), np.float32)
    img[3:9, 3:9] = 1.0          # 分片常数方块：TV 先验的理想对象
    theta = R.make_theta(180.0, 16)
    sino = R.compute_sinogram(img, theta)
    p_vec = sino.ravel().astype(np.float32)
    A, _ = R.build_system_matrix(n_s, theta, None, None)

    # ---- 1) TV 梯度必须与有限差分一致 ----
    # 这是整个算法的承重细节：梯度错了，TV 步会朝错误方向走，而 RMSE 仍可能下降
    # （因为 POCS 步本身就在收敛），缺陷会被掩盖成「TV 没什么用」。
    rng = np.random.default_rng(1)
    f = rng.random((9, 9))

    def _tv_norm(g):
        gp = np.pad(g, 1, mode="edge")
        c = gp[1:-1, 1:-1]
        return float(np.sum(np.sqrt((c - gp[0:-2, 1:-1]) ** 2
                                    + (c - gp[1:-1, 0:-2]) ** 2 + 1e-8)))

    ana = R._tv_grad(f)
    h, worst = 1e-6, 0.0
    for s_i in range(9):
        for t_i in range(9):
            fp = f.copy(); fp[s_i, t_i] += h
            fm = f.copy(); fm[s_i, t_i] -= h
            worst = max(worst, abs((_tv_norm(fp) - _tv_norm(fm)) / (2 * h) - ana[s_i, t_i]))
    check(worst < 1e-6, f"_tv_grad 与有限差分一致（最大偏差 {worst:.2e}）")

    # ---- 2) a=0 且 beta_red=1 时必须逐 bit 退化为 ART ----
    # 这一条同时锁住两件事：POCS 步就是带松弛的 ART（beta=1 时即原式），以及
    # 返回的是 f_res（POCS 之后、TV 之前）。若误返回 TV 步之后的图，即便 dtvg=0
    # 也会因多一次 reshape/拷贝路径而暴露差异；若 dtvg 未按 a 归零则直接不等。
    for k in (1, 3, 7):
        art, _ = R.compute_art(A, p_vec, n_s, k)
        asd, _ = R.compute_asdpocs(A, p_vec, n_s, k, a=0.0, beta_red=1.0)
        check(np.array_equal(art, asd),
              f"a=0、beta_red=1 时 ASD-POCS 逐 bit 等于 ART（{k} 轮，"
              f"最大差 {float(np.abs(art - asd).max()):.1e}）")

    # ---- 3) _tv_grad 必须是零次齐次（尺度不变）----
    # 这是 α 能跨几何/离散化移植的根据：TV 梯度尺度不变 ⇒ ‖df‖ 尺度不变 ⇒
    # 步长 dtvg/‖df‖·df 与 dtvg 同尺度，而 dtvg = a·d_p 随 d_p 走，于是 a 无量纲。
    # 少了 L2 归一化这一步，a 就绑死在特定的 A 与 p 尺度上，文献值不再可搬。
    # （注意不能在 compute_asdpocs 的出口测尺度协变：_finite_clip 截到绝对区间
    #   [0,1]，那一步本就不是尺度协变的，实测偏差达 24%——测错了层。）
    # 齐次性只是**近似**成立：sqrt 里的 eps=1e-8 在相邻像素几乎相等处会主导分母，
    # 那些位置的比值对尺度敏感。实测最坏 4.75e-05（c=0.5），故阈值取 1e-3——
    # 仍比"漏掉 L2 归一化"的后果小三个量级以上（那会给出 O(1) 或 O(c) 的偏差）。
    for c in (0.5, 2.0, 7.0):
        dev = float(np.abs(R._tv_grad(c * f) - ana).max())
        check(dev < 1e-3, f"_tv_grad(c·f) ≈ _tv_grad(f)，c={c}（最大偏差 {dev:.2e}）")

    # ---- 4) TV 步确实生效，且 a 越大偏离纯 ART 越远 ----
    # 若 dtvg 被误写成每轮重新按 a·d_p 赋值，随 d_p 衰减 a 的影响会被冲淡；
    # 只初始化一次时，a 的影响必须单调体现在与 a=0（纯 ART）的距离上。
    base, _ = R.compute_asdpocs(A, p_vec, n_s, 15, a=0.0)
    devs = [float(np.abs(R.compute_asdpocs(A, p_vec, n_s, 15, a=aa)[0] - base).max())
            for aa in (0.05, 0.2, 0.8)]
    check(devs[0] < devs[1] < devs[2],
          f"与纯 ART 的偏离随 a 单调增大（{devs[0]:.3f} < {devs[1]:.3f} < {devs[2]:.3f}）")

    # ---- 5) POCS 步保证数据残差随轮数下降 ----
    resids = [float(np.linalg.norm(A @ R.compute_asdpocs(A, p_vec, n_s, k)[0].ravel()
                                   - p_vec)) for k in (1, 5, 20)]
    check(resids[0] > resids[1] > resids[2],
          f"数据残差 ‖A·x−p‖ 随轮数单调下降（{resids[0]:.2f} > {resids[1]:.2f} > {resids[2]:.2f}）")

    # ---- 6) 非负约束与接口约定 ----
    asd20, _ = R.compute_asdpocs(A, p_vec, n_s, 20)
    check(float(asd20.min()) >= 0.0, "ASD-POCS 非负约束生效")
    stop = {"n": 0}
    def _cancel(): stop["n"] += 1; return True
    rec_c, _ = R.compute_asdpocs(A, p_vec, n_s, 100, cancel_check=_cancel)
    check(stop["n"] == 1 and float(np.abs(rec_c).max()) == 0.0,
          "ASD-POCS cancel_check 首轮即停，返回全零初值")
    seen = []
    R.compute_asdpocs(A, p_vec, n_s, 3, progress_cb=seen.append)
    check(seen == [0, 1, 2], f"ASD-POCS progress_cb 每轮回调一次（得 {seen}）")

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

    # 2) 加载层过滤：临时 JSON 落盘 -> _load_annotations_json 只留合规条目
    import tempfile
    ED = tempfile.mkdtemp(); v.persistence_dir = ED
    pid = "ANNOFILTER_TEST"
    fp = os.path.join(ED, f"{pid}_annotations.json")
    data = {"__meta__": {"series_uid": v._current_series_uid(),
                          "geometry_fingerprint": v._current_geometry_fingerprint()}, "all": [
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
        import shutil
        shutil.rmtree(ED, ignore_errors=True)
        v.global_annotations = saved


def test_ai_failure_visible(app):
    """AI 彻底失败（含兜底路径）必须如实反映到界面，而不是永远停在「运算中」。

    背景：ONNX 分支自带 try，但兜底的数学降级路径原本没有——异常直接冲出 _run，
    线程死掉、终端打印 Exception in thread，而界面 _ai_state 永远是 'running'、
    状态栏永远显示「AI 引擎自动运算中…」，用户在等一个永远不会来的结果。
    本测试把「失败必须可见」钉死。不依赖真实数据与权重，可进 SKIP_REAL_DATA 子集。
    """
    print("[AI 失败可见性]")
    import time

    import ai_engine
    import segmentation
    orig_fb = segmentation.segment_lungs_fallback
    orig_onnx = ai_engine.AutoAIEngineThread._run_onnx_multiorgan

    def boom(vol):
        raise ValueError("构造的兜底失败")

    def boom_onnx(self, norm_vol):
        # 刻意抛非 RuntimeError：RuntimeError 会被 _run_body 当作拆卸期竞态吞掉并置
        # _cancelled，那是另一条路径。这里要走的是「ONNX 真失败 → 降级 → 降级也失败」。
        raise ValueError("构造的 ONNX 失败")

    vf = None
    try:
        segmentation.segment_lungs_fallback = boom
        ai_engine.segmentation.segment_lungs_fallback = boom
        # 不能改 ai_engine.MODEL_PATH 来绕过 ONNX：它是 __init__ 的默认参数值，
        # 在函数定义时就已绑定，改模块变量对已定义的签名无效（本测试初版即栽在这）。
        ai_engine.AutoAIEngineThread._run_onnx_multiorgan = boom_onnx
        vf = m.MedicalViewer(); app.processEvents()
        if vf.ai_thread: vf.ai_thread.cancel()
        vf.volume_hu = np.random.RandomState(0).randint(-1000, 400, (6, 48, 48)).astype(np.int16)
        vf.dicom_datasets = [None] * 6
        vf._kickoff_ai()                                # 产品路径，不手工构造线程
        for _ in range(100):
            app.processEvents()
            if vf.ai_thread and not vf.ai_thread.isRunning(): break
            time.sleep(0.03)
        for _ in range(10):
            app.processEvents(); time.sleep(0.01)
        check(vf._ai_state == 'failed', f"兜底路径抛异常后状态机进入 failed（得 '{vf._ai_state}'）")
        txt = vf.lbl_ai_status.text()
        check('失败' in txt or 'failed' in txt.lower(), f"状态栏如实写明失败（得「{txt}」）")
        # 措辞必须与「跑成功了但没检出器官」区分开——混为一谈会让用户以为影像里真没器官
        check('检出' not in txt and 'Ready' not in txt, "失败文案不冒充「检出 0 个器官」")
        check('#E74C3C' in vf.lbl_ai_status.styleSheet(), "失败状态用红色，视觉上可区分")
    finally:
        segmentation.segment_lungs_fallback = orig_fb
        ai_engine.segmentation.segment_lungs_fallback = orig_fb
        ai_engine.AutoAIEngineThread._run_onnx_multiorgan = orig_onnx
        if vf is not None:
            if vf.ai_thread: vf.ai_thread.cancel()
            vf.close()
        app.processEvents()


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
    """multiframe fail closed；坏片跳过；全坏优雅中止并恢复原序列。"""
    print("[multiframe fail-closed / 坏片 / 全坏防护]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    vm = m.MedicalViewer(); app.processEvents()
    if vm.ai_thread:
        vm.ai_thread.cancel()

    # 1) multi-frame classic CT 在 pixel decode 前拒绝，不再“展开即支持”
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
        check(not crashed and vm.volume_hu is None,
              "multi-frame classic CT fail closed，未构建伪 3-D volume")
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

    # 3) decode 后必须按实际保留切片重算 geometry。若坏片恰在规则栈的中间，
    # pre-decode 的 0/1/2/3 mm 看似 uniform；跳过 z=1 后实际是 0/2/3 mm，
    # 不重算会让 MPR/mL/STL/AI 继续把缺层体积伪装成 1 mm 等距栈。
    d_gap = tempfile.mkdtemp()
    try:
        sid = generate_uid()
        for i in (0, 2, 3):
            _write_min_dcm(os.path.join(d_gap, f"g{i}.dcm"), (16, 16), sid,
                           ipp_z=i, inst=i)
        _write_min_dcm(os.path.join(d_gap, "bad_middle.dcm"), (16, 16), sid,
                       ipp_z=1, inst=1, truncate=True)
        vm.load_data(d_gap); app.processEvents()
        if vm.ai_thread:
            vm.ai_thread.cancel()
        check(vm.volume_hu.shape[0] == 3
              and not vm.uniform_z_geometry_valid
              and vm._slice_spacing() == 0.0
              and not vm.btn_mpr.isEnabled(),
              "坏片位于中间时按 decode 后实际切片重算 z geometry，关闭伪等距物理功能")
    finally:
        shutil.rmtree(d_gap, ignore_errors=True)

    # 4) 全坏目录：优雅中止，保留上一次成功加载的序列，且 datasets 与 volume 一致
    prev_shape = vm.volume_hu.shape
    prev_geometry = vm.series_geometry
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
        check(not crashed and vm.volume_hu.shape == prev_shape and consistent and nav_ok
              and vm.series_geometry == prev_geometry,
              "全坏目录优雅中止并恢复原序列（datasets/volume/geometry 一致、可导航）")
    finally:
        shutil.rmtree(d3, ignore_errors=True)
        if vm.ai_thread:
            vm.ai_thread.cancel()


def test_nonfinite_dicom_tags(app):
    """DICOM 数值标签为 NaN/Inf 时不得静默流入下游。

    NaN 是【合法的 float】——_dcm_float 原有的 None 检查与 float() 都拦不住它。
    实测后果比崩溃更糟，因为全程无任何异常提示：
      · RescaleSlope=NaN → HU 全 NaN → 弦图 100% 非有限，而 BP/FBP/DFR 照常「跑通」
        出图，用户看到的是从垃圾数据算出来的图；
      · PixelSpacing=NaN → 所有距离/面积/体积测量静默变成 nan。
    不依赖真实数据与权重，可进 SKIP_REAL_DATA 子集。
    """
    print("[非有限数值标签 DICOM 防护]")
    import shutil
    import tempfile

    import recon as R
    vn = m.MedicalViewer(); app.processEvents()
    if vn.ai_thread: vn.ai_thread.cancel()

    # 1) _dcm_float 本身：NaN / ±Inf 一律退回 default
    for bad, tag in ((float('nan'), 'NaN'), (float('inf'), '+Inf'), (float('-inf'), '-Inf')):
        ds = type('D', (), {'RescaleSlope': bad, 'PixelSpacing': [bad, 1.0]})()
        got = vn._dcm_float(ds, 'RescaleSlope', 7.0)
        got_i = vn._dcm_float(ds, 'PixelSpacing', 7.0, idx=0)
        check(got == 7.0 and got_i == 7.0,
              f"_dcm_float 把 {tag} 退回默认值（得 {got} / {got_i}）")
    # 正常值不受影响——防护不能把好数据也一并兜掉
    ds_ok = type('D', (), {'RescaleSlope': 2.5, 'PixelSpacing': [0.7, 0.7]})()
    check(vn._dcm_float(ds_ok, 'RescaleSlope', 1.0) == 2.5
          and vn._dcm_float(ds_ok, 'PixelSpacing', 1.0, idx=0) == 0.7,
          "正常有限值仍原样返回（防护未误伤）")

    # 2) 端到端：RescaleSlope=NaN 不能证明 HU，整卷须降级为有限 raw stored values。
    d = tempfile.mkdtemp()
    try:
        for i in range(4):
            _write_min_dcm(os.path.join(d, f"n{i:03d}.dcm"), (32, 32), '9.9.9',
                           ipp_z=i, inst=i, slope=float('nan'))
        vn._read_dicom_dir(d); vn._build_volume_hu(); app.processEvents()
        finite = bool(np.isfinite(vn.volume_hu).all())
        check(finite and not vn.series_geometry.hu_calibrated,
              f"RescaleSlope=NaN 的序列降级为有限 raw values 且 HU disabled（finite={finite}）")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if vn.ai_thread: vn.ai_thread.cancel()

    # 3) 纵深防御：compute_sinogram 对含 NaN 的输入也须产出全有限弦图
    #    （一个 NaN 像素经 Radon 线积分会污染所有穿过它的射线——实测局部 4×4 的 NaN
    #     就让整幅弦图 100% 非有限）
    img = np.random.RandomState(0).rand(48, 48).astype(np.float32)
    img[10:14, 10:14] = np.nan
    sino = R.compute_sinogram(img, R.make_theta(180))
    check(bool(np.isfinite(sino).all()),
          f"含 NaN 输入的弦图仍全有限（有限占比 {np.isfinite(sino).mean() * 100:.0f}%）")
    img2 = np.random.RandomState(1).rand(48, 48).astype(np.float32)
    s_ref = R.compute_sinogram(img2, R.make_theta(180))
    s_again = R.compute_sinogram(img2, R.make_theta(180))
    check(bool(np.allclose(s_ref, s_again)), "无 NaN 时弦图不受防护影响（结果可复现且未改变）")
    vn.close(); app.processEvents()


def test_empty_dicom_tags(app):
    """RescaleSlope/Intercept/PixelSpacing/SliceThickness 存在但为空(None)时，
    加载不得因 float(None) 崩溃，HU/physical quantification 必须安全关闭。"""
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
        crashed = False; stats = None
        try:
            ve.load_data(d); app.processEvents()
            if ve.ai_thread:
                ve.ai_thread.cancel()
            ve.volume_mask = np.ones(ve.volume_hu.shape, np.uint8)
            stats = ve._compute_organ_stats()   # 无可信单位/spacing 时应安全返回空结果
        except Exception as ex:
            crashed = True
            print("   ", type(ex).__name__, ex)
        check(not crashed and ve.volume_hu is not None and not ve.hu_calibrated
              and not ve.inplane_spacing_valid and stats == [],
              "空数值标签 DICOM 可阅片，但 HU/physical quantification fail closed")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if ve.ai_thread:
            ve.ai_thread.cancel()


def test_export_path_safety(app):
    """PatientID 含 '/' 或 '..' 时：不得路径穿越写到导出目录之外；净化后存取仍往返一致。"""
    print("[导出文件名路径安全]")
    import glob
    import shutil
    import tempfile

    from PySide6.QtWidgets import QMessageBox
    ED = tempfile.mkdtemp()
    # 净化器单元：普通 ID 不变（不破坏既有文件），危险字符被中和
    su = m.MedicalViewer._safe_name
    check(su("12345") == "12345" and su("RIDER-1234") == "RIDER-1234", "普通 PatientID 不被改动")
    check("/" not in su("A/B") and su("..") == "Unknown" and su("") == "Unknown", "斜杠/纯点/空被中和")

    vp = m.MedicalViewer(); app.processEvents()
    vp.persistence_dir = ED; vp.export_dir = ED
    if vp.ai_thread:
        vp.ai_thread.cancel()

    class _DS:
        # SeriesInstanceUID 在真实 DICOM 中是 Type 1 必填；蒙版缓存据它校验序列身份
        # （防止把同患者另一序列的蒙版张冠李戴），故此桩必须带上才具代表性。
        def __init__(self, pid, index,
                     uid="1.2.826.0.1.3680043.2.1125.1.314159"):
            self.PatientID = pid; self.PatientName = pid; self.SeriesInstanceUID = uid
            self.SOPInstanceUID = f"{uid}.{index + 1}"
            self.ImageOrientationPatient = (1, 0, 0, 0, 1, 0)
            self.ImagePositionPatient = (0, 0, index)
            self.PixelSpacing = (1, 1)

    def datasets(pid):
        return [_DS(pid, index) for index in range(3)]

    made = []
    saved_information = QMessageBox.information
    saved_warning = QMessageBox.warning
    try:
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        # 1) 路径穿越封堵
        esc = os.path.abspath(os.path.join(ED, "..", "PWNED_annotations.json"))
        before = os.path.exists(esc)
        vp.dicom_datasets = datasets("../PWNED")
        vp.volume_hu = np.zeros((3, 8, 8), np.float32)
        vp.global_annotations = {'all': [{'id': 'x', 'type': 'ruler', 'p1': (1, 1), 'p2': (2, 2)}]}
        vp.volume_mask = np.ones((3, 8, 8), np.uint8)
        saved = vp.save_project()
        made += glob.glob(os.path.join(ED, "_PWNED_*"))
        check(saved and not (os.path.exists(esc) and not before),
              "路径穿越被封堵，且有效 project 写入安全目录")

        # 2) 斜杠 PatientID 始终映射到安全的同一 basename（往返由专门 cache test 覆盖）
        vp.dicom_datasets = datasets("PID/WITH/SLASH")
        vp.volume_hu = np.zeros((3, 8, 8), np.float32)
        vp.global_annotations = {'all': [{'id': 'rt', 'type': 'ruler', 'p1': (1, 1), 'p2': (5, 5)}]}
        vp.volume_mask = np.ones((3, 8, 8), np.uint8) * 7
        saved = vp.save_project()
        made += glob.glob(os.path.join(ED, "PID_WITH_SLASH_*"))
        check(saved and any(os.path.basename(f).startswith("PID_WITH_SLASH_") for f in made),
              "斜杠 PatientID 只在临时目录生成净化后的安全 basename")
    finally:
        QMessageBox.information = saved_information
        QMessageBox.warning = saved_warning
        shutil.rmtree(ED, ignore_errors=True)
        vp.close(); app.processEvents()


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


def test_dialog_i18n_coverage(app):
    """所有用户可见的 QMessageBox 都必须带语言分支。

    对话框不像常驻控件那样能被 test_i18n_persistent 扫到——它们只在特定操作时弹出，
    逐个触发既慢又难覆盖全。故改用静态扫描源码：凡 QMessageBox.xxx( 调用，其参数里
    必须出现语言分支——self.is_english，或项目惯用的局部别名 e（e = self.is_english）。
    此前实测漏了 8 处：既有英文硬编码（"Project Saved."、"Save Failed" 等，中文界面
    也弹英文），也有反向的中文硬编码（3D 追踪的「仅支持 Axial」提示，英文用户会看到
    中文）。双语切换是对外宣传的功能，对话框漏译等于功能没做完。
    纯静态分析，不依赖真实数据与 Qt 显示。
    """
    print("[对话框 i18n 覆盖]")
    import re
    src_dir = _ROOT
    call_re = re.compile(r'QMessageBox\.(information|warning|critical|question)\s*\(')
    bad = []
    scanned = 0
    for fn in ('main.py', 'annotation_lab.py', 'compare_lab.py', 'recon_lab.py',
               'interaction.py', 'ui_builder.py', 'graphics_view.py'):
        fp = os.path.join(src_dir, fn)
        if not os.path.exists(fp):
            continue
        text = open(fp, encoding='utf-8').read()
        for mt in call_re.finditer(text):
            # 从调用起点向后取到括号配平处，得到完整实参串
            i = text.index('(', mt.start()); depth = 0; j = i
            while j < len(text):
                if text[j] == '(': depth += 1
                elif text[j] == ')':
                    depth -= 1
                    if depth == 0: break
                j += 1
            args = text[i:j + 1]
            # 判据认「语言分支」而非字面量：项目惯例是先取局部别名 e = self.is_english，
            # 只认 is_english 会把这类写法误报（初版即如此，误报 3 处）。
            if 'is_english' not in args and not re.search(r'\bif e\b', args):
                line = text[:mt.start()].count('\n') + 1
                bad.append(f"{fn}:{line} {mt.group(1)}")
            scanned += 1
    # 下限自检：本扫描器认的是字面 `QMessageBox.xxx(`。一次合法重构（模块级
    # `MB = QMessageBox` 后改用 `MB.warning(...)`）就能让 12 处调用整体消失，
    # 而 `check(not bad)` 会照样报「缺失 0 处」——即「解析出 0 条却通过」那一类。
    # 故先断言扫到的调用点数量，定位写错时当场失败而不是伪装成通过。
    check(scanned >= 15,
          f"扫到 {scanned} 处 QMessageBox 调用（过少说明本测试的定位写错了）")
    check(not bad, f"每个 QMessageBox 都有语言分支（缺失 {len(bad)} 处"
                   + (f"：{'; '.join(bad[:4])}" if bad else "") + "）")

    # 静态扫描保证「有分支」，再抽一条端到端确认分支方向没写反
    import re as _re

    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QMessageBox as _QMB

    from constants import CORONAL
    seen = []
    saved_info = _QMB.information
    _QMB.information = staticmethod(lambda p, t, msg='', *a, **k: seen.append((t, msg)))
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        vi.volume_hu = np.zeros((4, 32, 32), np.float32)
        vi.dicom_datasets = [None] * 4
        _mark_supported_capabilities(vi)
        vi.views[1]['plane'] = CORONAL          # 触发「3D 追踪仅支持 Axial」提示
        cjk = _re.compile(r'[一-鿿]')
        for en in (False, True):
            vi.is_english = en; seen.clear()
            vi.handle_3d_track_requested(1, QRectF(1, 1, 5, 5))
            txt = "".join(str(t) + str(msg) for t, msg in seen)
            has_cjk = bool(cjk.search(txt))
            check(bool(seen) and has_cjk != en,
                  f"{'英文' if en else '中文'}界面下该提示语言正确"
                  f"（含中文={has_cjk}，文案「{txt[:34]}」）")
    finally:
        _QMB.information = saved_info
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


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


def test_quantify_high_label():
    """标签含 254/255 时的器官定量：scipy 在 uint8 下 labels.max()+2 溢出会崩。

    背景：MANUAL_TRACK_LABEL = 255 是 3D 追踪工具的专属标签，追踪完立刻调用
    _update_organ_stats() → 必崩。而 299 项测试全绿——因为测试里的定量用例
    从没用过 255 这一档标签，是纯粹的取值盲区。
    """
    print("[高位标签器官定量（uint8 溢出防护）]")
    import quantify
    from constants import MANUAL_TRACK_LABEL
    rng = np.random.RandomState(0)
    vol = rng.uniform(-1000, 400, (6, 32, 32)).astype(np.float32)
    names = {5: ("肝", "Liver"), MANUAL_TRACK_LABEL: ("手动追踪", "Manual")}
    for tag, labels in (("只含 255（3D 追踪后）", [MANUAL_TRACK_LABEL]),
                        ("同时含 5 与 255", [5, MANUAL_TRACK_LABEL]),
                        ("只含 254（边界）", [254])):
        msk = np.zeros(vol.shape, np.uint8)
        for i, lab in enumerate(labels):
            msk[i, 4 + i * 8:12 + i * 8, 4:12] = lab
        crashed, st = False, []
        try:
            st = quantify.compute_organ_stats(vol, msk, (1., 1., 1.), names)
        except Exception as ex:
            crashed = True; print("   ", type(ex).__name__, ex)
        check(not crashed and len(st) == len(labels),
              f"{tag} 定量不崩且返回 {len(labels)} 项（得 {len(st)}）")
        # 不能只验「不崩」：加宽类型后数值必须仍与逐标签手算一致
        if st:
            ok = all(abs(r['mean_hu'] - float(vol[msk == r['id']].mean())) < 1e-3 for r in st)
            check(ok, f"{tag} 均值与手算逐项一致")


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

    # 【降级失败必须冒泡到守卫】此前本函数 except Exception → 返回全零蒙版，于是
    # ai_engine._run 的顶层守卫永远收不到异常：它 emit 的是 finished(全零)，界面显示
    # 「检出 0 个器官」——正是那道守卫的 docstring 声称已杜绝的「把失败谎报成成功」。
    # 这里直接断言纯函数会抛：吞异常一旦回归，此处立刻失败。
    import scipy.ndimage as _ndi
    _orig_label = _ndi.label
    _ndi.label = lambda *a, **k: (_ for _ in ()).throw(MemoryError("注入：连通域 OOM"))
    try:
        raised = False
        try:
            segmentation.segment_lungs_fallback(vol)
        except MemoryError:
            raised = True
        check(raised, "内部失败时向上抛异常（而非吞成全零蒙版交给 UI）")
    finally:
        _ndi.label = _orig_label


def test_undo_restores_confidence():
    """撤销必须把 volume_conf 一并还原，且「清空」的快照不被后续整卷编辑顶掉。

    两条都来自实测：画笔/追踪会把改动体素的 conf 清成哨兵 0，而 quantify 用 conf==0
    剔除非模型体素——只还原 mask 的话，撤销后该器官的 conf_cover 永久 < 1，定量面板
    于是给一个 100% 来自模型的器官标上「模型判定 XX%」。另一条：整卷快照原先共用一个
    槽位，「清空蒙版」存的那份会被之后任意一次 3D 追踪顶掉，而清空的确认框刚写着
    「可用 Ctrl+Z 还原蒙版」——那 ~100 秒的推理产物就真的回不来了。

    直接调 MedicalViewer 的未绑定方法，不建窗口、不触发 AI、不依赖真实数据。
    """
    print("[撤销：置信度一并还原 / 清空快照不被顶掉]")
    import main as _m
    v = _m.MedicalViewer.__new__(_m.MedicalViewer)
    v.recon_mode_active = True                    # 让 _undo_mask_edit 跳过重绘
    v._update_organ_stats = lambda *a, **k: None
    v.volume_hu = np.zeros((6, 16, 16), np.float32)
    v.volume_mask = np.zeros((6, 16, 16), np.uint8); v.volume_mask[2:4, 4:9, 4:9] = 5
    v.volume_conf = np.full((6, 16, 16), 242, np.uint8)
    v._mask_undo = []
    om, oc = v.volume_mask.copy(), v.volume_conf.copy()

    _m.MedicalViewer._push_mask_undo(v, 2)
    v.volume_mask[2][6:8, 6:8] = 255
    v.volume_conf[2][6:8, 6:8] = 0                # 画笔把 conf 清成哨兵
    _m.MedicalViewer._undo_mask_edit(v)
    check(np.array_equal(v.volume_mask, om), "切片撤销：蒙版还原")
    check(np.array_equal(v.volume_conf, oc),
          f"切片撤销：置信度一并还原（残留哨兵 {int((v.volume_conf == 0).sum())} 体素）")

    v.volume_mask, v.volume_conf, v._mask_undo = om.copy(), oc.copy(), []
    _m.MedicalViewer._push_volume_undo(v, slot=v._VOL_UNDO_CLEAR, adopt=True)
    v.volume_mask = np.zeros_like(om); v.volume_conf = None      # 清空
    _m.MedicalViewer._push_volume_undo(v)                        # 随后一次整卷编辑
    v.volume_mask[1][2:4, 2:4] = 7
    slots = [e[0] for e in v._mask_undo]
    check(v._VOL_UNDO_CLEAR in slots and v._VOL_UNDO in slots,
          f"清空快照与后续整卷编辑各占一槽（栈 {slots}）")
    _m.MedicalViewer._undo_mask_edit(v); _m.MedicalViewer._undo_mask_edit(v)
    check(np.array_equal(v.volume_mask, om), "两次撤销回到清空前的蒙版")
    check(v.volume_conf is not None and np.array_equal(v.volume_conf, oc),
          "两次撤销回到清空前的置信度")


def test_model_card_bad_fields():
    """产物「能解析但字段不对」时，说明卡必须降级而不是崩到界面。

    既有的 test_model_card_fallback 只覆盖了完全无法解析的情形（NUL 字节、非法 JSON），
    而实验脚本演进时最常见的形态是字段改名或类型变了——合法 JSON、读得进来，消费端
    lobe['overall_mean'] 直接 KeyError，用户点一下「模型说明卡」就崩。纯文件操作，
    不依赖 Qt 与真实数据。
    """
    print("[模型说明卡：字段不对时降级而非崩溃]")
    import json as _json
    import shutil
    import tempfile

    import model_card
    saved = model_card._RESULTS
    for tag, payload, expect_num in (
            ("字段名换了", {"summary": "n=57", "mean": 0.87}, False),
            ("ci 是标量", {"overall_mean": 0.87, "overall_ci": 0.05, "n_cases": 57}, False),
            ("ci 元素不足", {"overall_mean": 0.87, "overall_ci": [0.85], "n_cases": 57}, False),
            ("字段完整", {"overall_mean": 0.87, "overall_ci": [0.85, 0.89], "n_cases": 57}, True)):
        tmp = tempfile.mkdtemp()
        try:
            model_card._RESULTS = tmp
            with open(os.path.join(tmp, "seg3d_teacher_summary.json"), "w") as f:
                _json.dump(payload, f)
            ok, txt = True, ""
            try:
                txt = model_card.build_model_card(False)
            except Exception as e:                    # noqa: BLE001 — 正是要断言它不发生
                ok = False; txt = f"{type(e).__name__}: {e}"
            check(ok, f"  {tag}：不抛异常（{'' if ok else txt}）")
            if ok:
                check(("0.870" in txt) == expect_num,
                      f"  {tag}：{'印出' if expect_num else '不印出'}该 Dice")
        finally:
            model_card._RESULTS = saved
            shutil.rmtree(tmp, ignore_errors=True)


def test_stale_ai_cannot_overwrite_restored_mask():
    """从磁盘恢复蒙版时，上一序列仍在跑的推理必须被作废且代次推进。

    作废与代次自增原本只写在 _kickoff_ai 里，而 load_data 在磁盘已有缓存蒙版时会跳过它：
    旧推理继续跑到底（~8.8GB 不释放），代次没变，于是它完成时 generation 比对【放行】，
    旧序列的蒙版盖掉新序列刚恢复的那份，界面还照常显示绿色的「检出 N 个器官」。
    shape 比对挡不住——两个序列常常都是 512²。纯逻辑，不建窗口、不跑推理。
    """
    print("[过期 AI 回调不得覆盖已恢复的蒙版]")
    import main as _m

    class _FakeThread:
        def __init__(self): self.cancelled = False
        def isRunning(self): return True
        def cancel(self): self.cancelled = True

    v = _m.MedicalViewer.__new__(_m.MedicalViewer)
    v.ai_thread = _FakeThread(); v._ai_generation = 1
    v.volume_hu = np.zeros((4, 8, 8), np.float32)
    v.volume_mask = np.zeros((4, 8, 8), np.uint8); v.volume_mask[1][2:4, 2:4] = 12
    v.volume_conf = None; v._ai_state = 'done'; v._ai_time_ms = 0.0; v._ai_fallback = False
    v.recon_mode_active = True
    v._update_organ_stats = lambda *a, **k: None
    v.lbl_ai_status = type('L', (), {'setStyleSheet': lambda *a: None,
                                     'setText': lambda *a: None})()
    old_gen = v._ai_generation
    new_gen = _m.MedicalViewer._invalidate_running_ai(v)
    check(v.ai_thread.cancelled, "跳过 _kickoff_ai 时也作废了仍在跑的旧推理")
    check(new_gen > old_gen, f"代次已推进（{old_gen} → {new_gen}）")

    before = v.volume_mask.copy()
    stale = np.zeros((4, 8, 8), np.uint8); stale[2][5:7, 5:7] = 5
    _m.MedicalViewer.on_auto_ai_finished(v, stale, 99999.0, old_gen)
    check(np.array_equal(v.volume_mask, before),
          f"携带旧代次的回调被丢弃，蒙版仍是恢复的那份"
          f"（标签 {sorted(int(x) for x in np.unique(v.volume_mask))}）")


def test_anisotropic_pixel_spacing(app):
    """面内各向异性（PixelSpacing[0] != [1]）下，三个平面的 mm 换算与网格尺度都要对。

    这条本地真实数据永远覆盖不到：RIDER 与 TotalSegmentator-Lite 面内都是方形像素，
    行距等于列距，取错了也看不出来。而全链路曾经只读 PixelSpacing[0] 并把它同时当行距
    与列距——横断面与冠状面的水平 mm 读数会错（0.5/1.5 下差 3 倍），横断面还会因
    sp[0]==sp[1] 而跳过各向异性适配、把非方形体素画成方的；mesh 的体积/表面积与导出
    STL 的尺寸整体错，数量级却仍对得上，肉眼看不出来。

    合成 DICOM，不触发 AI 推理，不依赖真实数据。
    """
    print("[各向异性 PixelSpacing：三平面 mm 换算与网格尺度]")
    import shutil
    import tempfile

    import pydicom
    from pydicom.uid import generate_uid

    import main as _m
    import mesh3d as _mesh
    from constants import AXIAL, CORONAL, SAGITTAL
    d = tempfile.mkdtemp()
    try:
        uid = generate_uid()
        for i in range(6):
            _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (16, 16), uid, ipp_z=i * 2.0, inst=i + 1)
        for fn in os.listdir(d):                       # 覆写为各向异性
            ds = pydicom.dcmread(os.path.join(d, fn))
            ds.PixelSpacing = [0.5, 1.5]; ds.SliceThickness = 2.0
            ds.save_as(os.path.join(d, fn))
        v = _m.MedicalViewer(data_dir=d)
        app.processEvents()
        if v.ai_thread:
            v.ai_thread.cancel()
        vid = list(v.views)[0]
        # sp = (垂直/Y 的 mm/px, 水平/X 的 mm/px)
        want = {AXIAL: (0.5, 1.5), CORONAL: (2.0, 1.5), SAGITTAL: (2.0, 0.5)}
        for pl, exp in want.items():
            v.views[vid]['plane'] = pl
            v.update_display(); app.processEvents()
            got = tuple(round(float(t), 3) for t in v.views[vid]['view'].pixel_spacing)
            check(got == exp, f"  plane={pl} 的 (垂直,水平) mm/px = {exp}（得 {got}）")
        br = v.views[vid]['view'].overlay_lines.get('br', [])
        check(any("Px 0.50×1.50mm" in line for line in br),
              f"  overlay 同时呈现 anisotropic row×column spacing（{br}）")
        v.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 网格尺度：行距≠列距时，三根轴必须各取各的
    mask = np.zeros((10, 20, 20), np.uint8); mask[3:7, 6:14, 6:14] = 1   # z4 × row8 × col8
    vt, fc = _mesh.extract_surface(mask, 1, (0.5, 1.5, 2.0), step=1, smooth=0, decimate_grid=0)
    vol = _mesh.mesh_shape_stats(vt, fc)['volume_mm3']
    exact = 4 * 2.0 * 8 * 0.5 * 8 * 1.5                 # = 384 mm³
    check(abs(vol - exact) / exact < 0.10,
          f"  网格体积 {vol:.0f} 对解析值 {exact:.0f} 在 10% 内（行列距混用会得 128 或 1152）")


def test_mpr_geometry():
    """MPR 坐标几何纯函数直接单测——纯整数/数组运算，无 Qt / MedicalViewer。"""
    print("[MPR 坐标几何纯函数 mpr_geometry]")
    import mpr_geometry as g
    from constants import AXIAL, CORONAL, SAGITTAL
    shape = (40, 200, 300)   # (Z, Y, X)
    cur = (10, 20, 30)       # (z, y, x)
    check(g.hover_to_voxel(AXIAL, 50, 60, cur, shape) == (10, 60, 50), "Axial 悬停 (px,py)->(x,y)，z 不变")
    check(g.hover_to_voxel(CORONAL, 50, 15, cur, shape) == (24, 20, 50),
          "Coronal 上方从 superior slice 映射：z=Z-1-py")
    check(g.hover_to_voxel(SAGITTAL, 70, 15, cur, shape) == (24, 70, 30),
          "Sagittal 上方从 superior slice 映射：z=Z-1-py")
    check(g.hover_to_voxel(AXIAL, 999, 999, cur, shape) == (10, 199, 299), "越界裁剪到体积上界")
    check(g.hover_to_voxel(AXIAL, -5, -5, cur, shape) == (10, 0, 0), "负坐标裁剪到 0")
    check(g.voxel_to_crosshair(AXIAL, 10, 20, 30, shape) == (30, 20), "Axial 十字线 (x,y)")
    check(g.voxel_to_crosshair(CORONAL, 10, 20, 30, shape) == (30, 29),
          "Coronal 十字线 z 映射到上 S / 下 I 的 screen y")
    check(g.voxel_to_crosshair(SAGITTAL, 10, 20, 30, shape) == (20, 29),
          "Sagittal 十字线 z 映射到上 S / 下 I 的 screen y")
    check(g.nearest_slice([0, 5, 10, 15, 20], 12) == 2, "最近解剖切片 = 索引2 (z=10)")
    check(g.nearest_slice([0, 5, 10, 15, 20], 100) == 4, "超出范围取最末切片")


def test_dicom_landmark_orientation(app):
    """不对称亮点必须穿过 synthetic DICOM loader 与真实 render path 验证六向。"""
    print("[DICOM landmark：loader/render 的 A/P/L/R/S/I 闭环]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid

    from constants import AXIAL, CORONAL, SAGITTAL

    root = tempfile.mkdtemp()
    v = None
    try:
        sid = generate_uid()
        rows, cols = 6, 10
        for z in range(3):
            pixels = np.zeros((rows, cols), dtype=np.int16)
            if z == 2:
                pixels[1, 8] = 1000  # superior + anterior + patient-left 的不对称 landmark
            _write_min_dcm(os.path.join(root, f"s{z}.dcm"), (rows, cols), sid,
                           ipp_z=z, inst=z + 1, slope=1, intercept=0, pixels=pixels)
        v = m.MedicalViewer(); app.processEvents()
        v._kickoff_ai = lambda: None
        v.load_data(root); app.processEvents()
        vid = min(v.views)
        view = v.views[vid]['view']
        v.current_3d_pos = [2, 1, 8]
        v.slider_slice.setValue(2)

        def brightest_xy():
            image = view.image_item.pixmap().toImage()
            samples = [((x, y), image.pixelColor(x, y).red())
                       for y in range(image.height()) for x in range(image.width())]
            return max(samples, key=lambda item: item[1])[0]

        expected = {
            AXIAL: ((8, 1), {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'}),
            CORONAL: ((8, 0), {'top': 'S', 'bottom': 'I', 'left': 'R', 'right': 'L'}),
            SAGITTAL: ((1, 0), {'top': 'S', 'bottom': 'I', 'left': 'A', 'right': 'P'}),
        }
        for plane, (xy, labels) in expected.items():
            v.views[vid]['plane'] = plane
            v.update_display(); app.processEvents()
            check(brightest_xy() == xy and view.orient_labels == labels,
                  f"plane={plane} landmark={xy} 且 edge labels={labels}"
                  f"（得 {brightest_xy()} / {view.orient_labels}）")
    finally:
        if v is not None:
            if v.ai_thread: v.ai_thread.cancel()
            v.close(); app.processEvents()
        shutil.rmtree(root, ignore_errors=True)


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
    # 不崩还不够：纯点击（不拖动）原本会生成一条长度 0 的卡尺，并被 save_project
    # 持久化进工程 JSON。相邻工具本就有误触过滤（套索 ≥3 点、矩形截取 >5 像素），
    # 卡尺与自由画笔此前漏了。
    n0 = len(g['annotation_added']) if not crashed else -1
    check(n0 == 0, f"零长度卡尺不产生退化标注（得 {n0} 条）")
    n1 = len(drag(TOOL_RULER, 150, 150, 150.4, 150.3, steps=1)['annotation_added'])
    check(n1 == 0, f"亚像素抖动（<1px）同样视为误触（得 {n1} 条）")
    n2 = len(drag(TOOL_RULER, 100, 100, 140, 130, steps=3)['annotation_added'])
    check(n2 == 1, f"真实拖拽仍正常产出卡尺（得 {n2} 条）")
    n3 = len(drag(TOOL_DRAW, 60, 60, 60, 60, steps=1)['annotation_added'])
    check(n3 == 0, f"原地自由画笔不产生退化路径（得 {n3} 条）")
    n4 = len(drag(TOOL_DRAW, 60, 60, 120, 90, steps=4)['annotation_added'])
    check(n4 == 1, f"真实自由画笔仍正常产出路径（得 {n4} 条）")
    view.close()          # 独立实例用完即弃，无需还原状态，也不会污染主窗口
    app.processEvents()


def test_mesh3d_ui(v, app):
    """三维重建接入：按钮随器官有无启停，端到端不崩，网格体积与体素法互相印证。"""
    print("[三维重建接入]")
    from PySide6.QtWidgets import QDialog

    import mesh3d as M
    saved_mask = v.volume_mask
    saved_geometry = v.series_geometry
    _mark_supported_capabilities(v, saved_geometry.slice_spacing_mm or 1.0)
    saved_exec = QDialog.exec
    dlgs = []
    QDialog.exec = lambda self: dlgs.append(self)   # 不阻塞在模态窗，同时留下弹窗以便检查
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
        # 弹窗里必须真的装着可拖动的视图，并且拖动后画面确实换了一帧——
        # 只测 MeshView 单体不够，接线断了（信号没连、paint 没绑）单体测试照样全绿
        # 真实器官规模下的降质比例——拖动跟不跟手全看这一条（合成小体积测不出来）
        dv, df = M.decimate_vertex_clustering(verts, faces, grid=16)
        check(0 < len(df) < len(faces) * 0.5,
              f"真实器官规模下粗网格减面过半（{len(faces):,} → {len(df):,} 面）")
        from annotation_lab import MeshView
        mvs = dlgs[-1].findChildren(MeshView) if dlgs else []
        check(len(mvs) == 1, f"三维弹窗内含 1 个可拖动视图（得 {len(mvs)}）")
        if mvs:
            mv = mvs[0]
            before = mv.pixmap().toImage()
            check(not before.isNull(), "弹窗打开即已渲染出画面（不是空白等交互）")
            mv.set_angles(mv.azimuth + 90.0, mv.elevation)
            app.processEvents()
            check(mv.pixmap().toImage() != before, "转 90° 后视图画面确实刷新（信号已接通）")
    finally:
        QDialog.exec = saved_exec
        v.volume_mask = saved_mask
        v.series_geometry = saved_geometry
        v._apply_series_capabilities()
        v._update_organ_stats(); app.processEvents()


def test_mesh_view(app):
    """三维预览的鼠标拖动旋转：角度换算、夹紧、回绕、信号，以及画面确实随角度变。

    用独立的 MeshView 实例（同 test_mouse_interaction 的理由）：被测的是控件自身的
    交互换算，不依赖真实数据也不需要弹窗，故可进 SKIP_REAL_DATA 子集。
    """
    print("[三维交互旋转]")
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    import mesh3d as M
    from annotation_lab import MeshView

    def drag(view, dx, dy):
        """合成 press → move → release，驱动真实的事件处理器而非直接改属性。"""
        def mk(kind, x, y, btn):
            return QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, btn, Qt.NoModifier)
        view.mousePressEvent(mk(QMouseEvent.Type.MouseButtonPress, 100.0, 100.0, Qt.LeftButton))
        view.mouseMoveEvent(mk(QMouseEvent.Type.MouseMove, 100.0 + dx, 100.0 + dy, Qt.LeftButton))
        view.mouseReleaseEvent(mk(QMouseEvent.Type.MouseButtonRelease,
                                  100.0 + dx, 100.0 + dy, Qt.NoButton))

    vw = MeshView(azimuth=30.0, elevation=20.0)
    rot, stl = [], []
    vw.rotated.connect(lambda a, e: rot.append((a, e)))
    vw.settled.connect(lambda: stl.append(1))
    drag(vw, 60.0, -40.0)
    # 灵敏度 0.5°/px；纵向取负（鼠标下拉 = 视角下移 = elevation 减小，符合直觉）
    check(abs(vw.azimuth - 60.0) < 1e-6 and abs(vw.elevation - 40.0) < 1e-6,
          f"拖 (+60,-40)px 后角度 = (60, 40)（得 {vw.azimuth:.1f}, {vw.elevation:.1f}）")
    check(len(rot) == 1 and len(stl) == 1, "拖动发 rotated、松手发 settled 各一次")

    vw2 = MeshView(azimuth=30.0, elevation=20.0)
    vw2.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(200.0, 200.0),
                                   Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    check((vw2.azimuth, vw2.elevation) == (30.0, 20.0), "未按下时移动鼠标不旋转")

    vw3 = MeshView(azimuth=0.0, elevation=0.0)
    drag(vw3, 0.0, -1000.0)                     # 猛推：不夹紧就会越过 90° 万向节锁翻面
    check(vw3.elevation == 89.0, f"俯仰角上夹到 +89（得 {vw3.elevation:.1f}）")
    drag(vw3, 0.0, 2000.0)
    check(vw3.elevation == -89.0, f"俯仰角下夹到 -89（得 {vw3.elevation:.1f}）")
    vw4 = MeshView(azimuth=350.0, elevation=0.0)
    drag(vw4, 100.0, 0.0)
    check(abs(vw4.azimuth - 40.0) < 1e-6, f"方位角越过 360 回绕到 40（得 {vw4.azimuth:.1f}）")

    vw5 = MeshView()
    vw5.set_angles(400.0, 200.0)                # 预设视角按钮走的这条路，同样须夹紧/回绕
    check(vw5.azimuth == 40.0 and vw5.elevation == 89.0, "set_angles 同样回绕并夹紧")

    # 旋转不能是摆设：画面必须真的不同，且同角度须逐像素可复现
    N = 48; c = (N - 1) / 2
    zz, yy, xx = np.ogrid[:N, :N, :N]
    # 椭球而非球——球从任何角度看都一样，根本测不出旋转是否生效
    ell = ((((zz - c) / 1.8) ** 2 + ((yy - c) / 1.0) ** 2 + ((xx - c) / 0.7) ** 2) <= 14.0 ** 2)
    vt, fc = M.extract_surface(ell.astype(np.uint8) * 3, 3, (1.0, 1.0, 1.0), step=2)
    a0 = M.render_mesh(vt, fc, size=160, azimuth=30, elevation=20)
    a1 = M.render_mesh(vt, fc, size=160, azimuth=120, elevation=20)
    a2 = M.render_mesh(vt, fc, size=160, azimuth=30, elevation=20)
    d01 = float(np.abs(a0.astype(int) - a1.astype(int)).mean())
    check(d01 > 1.0, f"换方位角后画面确实改变（平均像素差 {d01:.1f}）")
    check(bool(np.array_equal(a0, a2)), "同角度渲染逐像素一致（渲染确定性）")

    # 拖动降质的前提：粗网格必须真的少很多面，且形状不能跑偏
    # 只断言方向（更少且非空）：减面比例随网格规模变化，这个 48³ 合成体本就面数不多
    # （实测减 31%），而真实器官规模减得多得多——比例断言放在有真实数据的 test_mesh3d_ui。
    dv, df = M.decimate_vertex_clustering(vt, fc, grid=16)
    check(0 < len(df) < len(fc), f"拖动用粗网格面数更少且非空（{len(fc):,} → {len(df):,}）")
    v_full = M.mesh_shape_stats(vt, fc)['volume_mm3']
    v_low = M.mesh_shape_stats(dv, df)['volume_mm3']
    rel = abs(v_low - v_full) / v_full
    check(rel < 0.05, f"粗网格体积偏差 <5%，拖动中看到的仍是同一形状（得 {rel * 100:.1f}%）")


def test_dl_recon_ui(v, app):
    """学习式重建接入重建实验室：按钮状态随工作流与模型可用性，视角不匹配须标注。"""
    print("[学习式重建接入]")
    import recon as R
    from constants import RECON_DL_MODEL, RECON_DL_VIEWS
    saved_tab = v.tabs.currentIndex()
    try:
        v.tabs.setCurrentIndex(1); app.processEvents()
        check(not v.btn_dl.isEnabled(), "未生成弦图时「深度学习重建」禁用")
        ready = R.dl_available(RECON_DL_MODEL)
        v.generate_sinogram(); app.processEvents()
        check(v.btn_dl.isEnabled() == ready,
              f"生成弦图后按钮状态 == 模型可用性（模型就绪={ready}，按钮={v.btn_dl.isEnabled()}）")
        if not ready:
            print("    本机无模型，跳过端到端断言"); return
        # 视角不匹配：默认 180°×1 采样 ≠ 模型训练的 20 视角，标题必须标注出来
        v.run_dl_recon(); app.processEvents()
        t4 = v.views[4]['title_label'].text()
        n_now = len(v.current_theta)
        if n_now != RECON_DL_VIEWS:
            check('不匹配' in t4 or 'mismatch' in t4.lower(),
                  f"视角不匹配（{n_now} vs {RECON_DL_VIEWS}）时标题明确标注（得 …{t4[-40:]}）")
        check('CNN' in t4, f"V4 标题标明是 CNN 后处理（得 {t4[:40]}）")
        t3 = v.views[3]['title_label'].text()
        check('ramp' in t3.lower(), f"V3 标明网络输入是 ramp-FBP（得 {t3[:40]}）")
        # 换切片后按钮须重新禁用——旧弦图已作废，不能拿旧图配新层
        v.slider_slice.setValue(v.slider_slice.value() + 1); app.processEvents()
        check(not v.btn_dl.isEnabled(), "换切片后按钮重新禁用（旧弦图作废）")
    finally:
        v.tabs.setCurrentIndex(saved_tab); app.processEvents()


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
    的蒙版静默套到当前序列上，器官定量随之给出错误体积。故必须同时匹配 UID、shape 和
    geometry/order fingerprint；legacy cache 缺 fingerprint 时 fail closed。"""
    print("[分割蒙版缓存守卫纯函数 annotation_lab.mask_cache_matches]")
    import annotation_lab as al
    shp, fp_a, fp_b = (233, 512, 512), "a" * 64, "b" * 64
    try:
        ok, why = al.mask_cache_matches("1.2.840.A", shp, fp_a,
                                        "1.2.840.A", shp, fp_a)
    except TypeError:
        ok, why = False, "旧接口尚未接收 fingerprint"
    check(ok and why == "", "同 UID/shape/fingerprint → 恢复缓存")
    if ok:
        ok, why = al.mask_cache_matches("1.2.840.A", shp, fp_a,
                                        "1.2.840.A", shp, fp_b)
        check(not ok and "fingerprint" in why.lower(),
              "同 UID/shape 但 slice-order fingerprint 不同 → 拒绝")
        ok, why = al.mask_cache_matches("1.2.840.A", shp, "",
                                        "1.2.840.A", shp, fp_a)
        check(not ok and "fingerprint" in why.lower(),
              "legacy cache 缺 fingerprint → 默认拒绝，不凭用户确认猜测")
        ok, why = al.mask_cache_matches("1.2.840.A", shp, fp_a,
                                        "1.2.840.B", shp, fp_a)
        check(not ok and "SeriesInstanceUID" in why,
              "同患者另一序列（shape 相同、UID 不同）→ 拒绝")
        ok, why = al.mask_cache_matches("1.2.840.A", (233, 512, 512), fp_a,
                                        "1.2.840.A", (200, 512, 512), fp_a)
        check(not ok and "shape" in why, "shape 不匹配 → 拒绝")


def test_geometry_fingerprint_contract():
    """cache/annotation fingerprint 必须稳定绑定有序 SOP、geometry 与 volume shape。"""
    print("[cache geometry/order fingerprint：deterministic SHA-256]")
    from types import SimpleNamespace

    import dicom_geometry as dg

    def one(sop, z):
        return SimpleNamespace(
            SOPInstanceUID=sop,
            SeriesInstanceUID="1.2.3",
            ImageOrientationPatient=(1, 0, 0, 0, 1, 0),
            ImagePositionPatient=(0, 0, z),
            PixelSpacing=(0.7, 0.9),
        )

    fn = getattr(dg, 'series_fingerprint', None)
    check(callable(fn), "geometry 模块提供 deterministic series_fingerprint")
    if callable(fn):
        a = [one("1.2.3.1", 0), one("1.2.3.2", 1), one("1.2.3.3", 2)]
        same = [one("1.2.3.1", 0), one("1.2.3.2", 1), one("1.2.3.3", 2)]
        reordered_identity = [one("1.2.3.2", 0), one("1.2.3.1", 1), one("1.2.3.3", 2)]
        fp = fn(a, (3, 8, 9))
        check(fp == fn(same, (3, 8, 9)) and len(fp) == 64
              and all(c in "0123456789abcdef" for c in fp),
              "相同输入跨对象得到同一 64-char SHA-256")
        check(fp != fn(reordered_identity, (3, 8, 9)),
              "同 UID/shape 但 SOP→slice 绑定改变时 fingerprint 改变")
        check(fp != fn(a, (3, 9, 8)), "volume shape 改变时 fingerprint 改变")


def test_mpr_linkage(app):
    """MPR 联动的入口与十字线同步：开关默认平面、平面切换的越界防护、光标 HUD。

    这几段是 interaction.py 里剩下的零覆盖区，而 MPR 联动正是四窗阅片的核心交互。
    三个平面共用同一个 3D 光标，任一处坐标映射写错都会让十字线指向别的解剖位置——
    这种错误在界面上看是「联动有点怪」，很难归因，故用断言把映射钉死。
    """
    print("[MPR 联动与十字线同步]")
    from PySide6.QtCore import QPointF

    from constants import SAGITTAL
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 10, 30, 40
        vi.volume_hu = np.zeros((Z, H, W), np.float32)
        vi.volume_hu[5, 10, 20] = 123.0
        vi.dicom_datasets = [None] * Z
        vi.slider_slice.setRange(0, Z - 1); vi.slider_slice.setValue(5)
        vi.current_3d_pos = [5, 15, 20]

        vi.btn_mpr.setChecked(True); vi.on_mpr_toggled(True); app.processEvents()
        planes = [vi.views[v]['cb_plane'].currentIndex() for v in (1, 2, 3, 4)]
        check(planes == [AXIAL, CORONAL, SAGITTAL, AXIAL],
              f"开启联动时四视图落到默认平面 {planes}")

        vi.on_mpr_toggled(False); app.processEvents()
        check(True, "关闭联动时隐藏十字线，不崩")

        # 下拉框清空重填会发出 index=-1，必须过滤掉而不是写进 plane
        before = vi.views[1]['plane']
        vi.change_view_plane(1, -1); app.processEvents()
        check(vi.views[1]['plane'] == before, "plane_idx=-1（下拉重填）被过滤，不写坏视图状态")
        vi.change_view_plane(1, CORONAL); app.processEvents()
        check(vi.views[1]['plane'] == CORONAL, "正常切换平面生效")
        vi.change_view_plane(1, AXIAL); app.processEvents()

        # 十字线同步：三平面共用一个 3D 光标，映射必须与 mpr_geometry 一致
        vi.btn_mpr.setChecked(True)
        vi.views[1]['plane'] = AXIAL
        vi.sync_crosshair(QPointF(20, 10), 1); app.processEvents()
        check(vi.current_3d_pos[1] == 10 and vi.current_3d_pos[2] == 20,
              f"Axial 上悬停 (20,10) → 光标 y=10 x=20（得 {vi.current_3d_pos}）")
        vi.views[2]['plane'] = CORONAL
        vi.sync_crosshair(QPointF(20, 3), 2); app.processEvents()
        check(vi.current_3d_pos[0] == 6,
              f"Coronal 上 S / 下 I 显示中 py=3 → z=Z-1-py（得 {vi.current_3d_pos}）")
        check(vi.slider_slice.value() == 6, "切片滑条同步到翻转后的新 z，不与光标脱节")

        # HUD 报出坐标、HU 与所在器官
        vi.volume_mask = np.zeros((Z, H, W), np.uint8); vi.volume_mask[5, 10, 20] = 5
        vi.views[1]['plane'] = AXIAL
        vi.current_3d_pos = [5, 15, 20]
        vi.sync_crosshair(QPointF(20, 10), 1); app.processEvents()
        hud = vi.lbl_hud.text()
        check('123' in hud, f"HUD 报出该体素 HU（得「{hud}」）")
        check(('肝' in hud) or ('Liver' in hud) or ('5' in hud),
              f"HUD 报出所在器官（得「{hud}」）")

        # 联动关闭时只更新 HUD，不改光标
        vi.btn_mpr.setChecked(False)
        pos = list(vi.current_3d_pos)
        vi.sync_crosshair(QPointF(1, 1), 1); app.processEvents()
        check(list(vi.current_3d_pos) == pos, "联动关闭时悬停只刷 HUD，不移动 3D 光标")

        # 重建/对比模式下不应响应
        vi.recon_mode_active = True
        vi.sync_crosshair(QPointF(5, 5), 1); app.processEvents()
        check(list(vi.current_3d_pos) == pos, "重建实验室模式下十字线同步被守卫")
        vi.recon_mode_active = False
    finally:
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_panel_scroll(app):
    """器官多时右侧面板必须能滚动，而不是把内容压没。

    实测缺陷：18 个器官时临床 Tab 内容需约 1540px，笔记本上可用高度只有 ~775px。
    Qt 在空间不足时不会溢出，而是**压缩可伸缩控件**——器官定量标签被压到 0px 高，
    状态栏写着「检出 18 个器官」，下面一条数据都读不到。这比截断更隐蔽：用户
    不会意识到还有内容存在。
    """
    print("[右侧面板滚动与底部固定]")
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QScrollArea
    vi = None
    try:
        vi = m.MedicalViewer(); vi.setFixedHeight(900); vi.resize(1600, 900); vi.show()
        app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 30, 128, 128
        vi.volume_hu = np.full((Z, H, W), 40.0, np.float32)
        vi.dicom_datasets = [type('D', (), {'PatientID': 'X', 'SeriesInstanceUID': '1',
                                            'StudyDate': '20240101', 'PixelSpacing': [1.5, 1.5],
                                            'SliceThickness': 1.5})() for _ in range(Z)]
        _mark_supported_capabilities(vi, 1.5)
        vi.slider_slice.setRange(0, Z - 1); vi.current_3d_pos = [15, 64, 64]
        mk = np.zeros((Z, H, W), np.uint8)
        for i, lab in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]):
            r, c = divmod(i, 6)
            mk[10:20, 10 + r * 30:28 + r * 30, 8 + c * 20:24 + c * 20] = lab
        vi.volume_mask = mk
        vi.volume_conf = np.full((Z, H, W), 240, np.uint8)
        vi._update_organ_stats()
        # QScrollArea 的 widgetResizable 要走完一个布局周期才把高度分配下去，
        # 而 processEvents() 只处理已排队的事件、不推进时间。不等就会量到中间态
        # （实测 28px），把「布局还没算完」误判成「内容被压缩」。
        _t0 = time.time()
        while time.time() - _t0 < 0.5:
            app.processEvents(); time.sleep(0.01)

        n = len(vi._organ_stats)
        check(n >= 15, f"构造出 {n} 个器官的定量（模拟真实推理规模）")
        need = vi.lbl_ai_stats.sizeHint().height()
        got = vi.lbl_ai_stats.height()
        check(got >= need * 0.95,
              f"定量标签拿到足够高度：需 {need}px 实得 {got}px（压缩过就读不到数据）")

        areas = vi.right_panel.findChildren(QScrollArea)
        check(len(areas) >= 2, f"两个 Tab 都有滚动区（{len(areas)} 个）")
        for a_ in areas:
            check(a_.horizontalScrollBarPolicy() == _Qt.ScrollBarAlwaysOff,
                  "  水平滚动条关闭——面板宽度固定，出现横向滚动只说明布局错了")

        # 破坏性操作固定在底部：滚动时位置不变，且始终可见
        y1 = vi.btn_reset.mapTo(vi.right_panel, vi.btn_reset.rect().bottomLeft()).y()
        for a_ in areas:
            a_.verticalScrollBar().setValue(a_.verticalScrollBar().maximum())
        app.processEvents()
        y2 = vi.btn_reset.mapTo(vi.right_panel, vi.btn_reset.rect().bottomLeft()).y()
        check(y1 == y2, f"滚到底后「重置工作区」不移动（y={y1}→{y2}）")
        check(0 < y2 <= vi.right_panel.height(),
              f"始终在可视区内（底边 {y2} ≤ 面板 {vi.right_panel.height()}）")
    finally:
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_compare_entry(app):
    """对比模式的入口 toggle_compare：五条分支此前全部零覆盖。

    现有 test_compare 走的是内部方法，绕过了这个入口，于是「没有主序列就点对比」
    「选目录时按取消」「选到一个没有 DICOM 的目录」这三条防御路径从未被走过——
    而它们恰恰是用户最容易碰到的。
    """
    print("[对比模式入口 toggle_compare]")
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    saved_i, saved_w, saved_d = QMessageBox.information, QMessageBox.warning, QFileDialog.getExistingDirectory
    box = {'info': 0, 'warn': 0}
    QMessageBox.information = staticmethod(lambda *a, **k: box.__setitem__('info', box['info'] + 1))
    QMessageBox.warning = staticmethod(lambda *a, **k: box.__setitem__('warn', box['warn'] + 1))
    vi, tmp = None, tempfile.mkdtemp()
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()

        # 1) 还没有主序列就点「加载对比序列」
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("无主序列时不该弹目录选择框")))
        vi.toggle_compare(); app.processEvents()
        check(box['info'] == 1 and not vi.compare_mode_active,
              "无主序列时提示并返回，不弹目录框")

        # 造一个主序列（合成 DICOM，走完整加载路径）
        d_main = os.path.join(tmp, 'main'); os.makedirs(d_main)
        uid_m = generate_uid()
        for k in range(4):
            _write_min_dcm(os.path.join(d_main, f'{k}.dcm'), (16, 16), uid_m, ipp_z=k * 2.0,
                           inst=k + 1, pid='CMP_MAIN')
        vi.load_data(d_main); app.processEvents()
        check(vi.volume_hu is not None, f"主序列已加载 {vi.volume_hu.shape}")
        if vi.ai_thread: vi.ai_thread.cancel()

        # 2) 弹出目录框但用户按了取消
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
        vi.toggle_compare(); app.processEvents()
        check(not vi.compare_mode_active and vi.compare_volume is None,
              "用户取消目录选择 → 不进入对比模式，状态不变")

        # 3) 选到一个没有 DICOM 的目录
        d_empty = os.path.join(tmp, 'empty'); os.makedirs(d_empty)
        with open(os.path.join(d_empty, 'readme.txt'), 'w') as fh:
            fh.write('not dicom')
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: d_empty)
        box['warn'] = 0
        vi.toggle_compare(); app.processEvents()
        check(box['warn'] == 1 and not vi.compare_mode_active,
              "目录里没有可读 DICOM → 警告并返回")

        # 4) 正常加载既往序列
        d_prev = os.path.join(tmp, 'prev'); os.makedirs(d_prev)
        uid_p = generate_uid()
        for k in range(4):
            _write_min_dcm(os.path.join(d_prev, f'{k}.dcm'), (16, 16), uid_p, ipp_z=k * 2.0,
                           inst=k + 1, pid='CMP_PREV')
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: d_prev)
        vi.toggle_compare(); app.processEvents()
        check(vi.compare_mode_active and vi.compare_volume is not None,
              f"进入对比模式，既往序列 {None if vi.compare_volume is None else vi.compare_volume.shape}")
        check(vi.chk_register.isEnabled(), "对比模式下「配准」复选框启用")
        check(len(vi.dicom_datasets) == 4 and vi.volume_hu.shape[0] == 4,
              "读取既往序列没有污染主序列")

        # 5) 再点一次 = 退出
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("退出时不该再弹目录框")))
        vi.toggle_compare(); app.processEvents()
        check(not vi.compare_mode_active, "再次点击退出对比模式")
        check(not vi.chk_register.isEnabled(), "退出后「配准」复选框随之禁用")

        # 重建实验室下不应进入对比
        vi.recon_mode_active = True
        box['info'] = 0
        vi.toggle_compare(); app.processEvents()
        check(box['info'] == 1 and not vi.compare_mode_active,
              "重建实验室模式下拒绝进入对比，并给出提示")
        vi.recon_mode_active = False
    finally:
        QMessageBox.information, QMessageBox.warning = saved_i, saved_w
        QFileDialog.getExistingDirectory = saved_d
        shutil.rmtree(tmp, ignore_errors=True)
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_crop_and_legend(app):
    """截取工具、标注 CRUD、图例显隐、定量面板的置信度显示——annotation_lab 的零覆盖区。

    写盘一律经 mock 过的 QFileDialog 重定向到临时目录：`Exported_Lesions/` 是不可恢复的
    产物目录，测试绝不能往里写（本会话的审计脚本就往那里落过一个文件，只能请用户手删）。
    """
    print("[截取/标注/图例/置信度显示]")
    import csv as _csv2
    import glob as _glob
    import shutil
    import tempfile
    import unittest.mock as _mock

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from constants import MANUAL_TRACK_LABEL
    vi, tmp = None, tempfile.mkdtemp()
    saved_q, saved_s, saved_w = QMessageBox.question, QFileDialog.getSaveFileName, QMessageBox.warning
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 6, 40, 40
        vi.volume_hu = np.full((Z, H, W), -900.0, np.float32)
        vi.volume_hu[2, 10:30, 10:30] = 60.0
        vi.dicom_datasets = [type('D', (), {'PatientID': 'CROPTEST', 'SeriesInstanceUID': '9.9',
                                            'StudyDate': '20240101', 'PixelSpacing': [1.0, 1.0],
                                            'SliceThickness': 1.0})() for _ in range(Z)]
        _mark_supported_capabilities(vi)
        vi.current_3d_pos = [2, 20, 20]
        vi.views[1]['plane'] = AXIAL
        poly = [(12, 12), (28, 12), (28, 28), (12, 28)]

        # 非 Axial 平面：截取不适用，必须直接返回
        vi.views[1]['plane'] = CORONAL
        QMessageBox.question = staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("非 Axial 平面不该弹出统计框")))
        vi.handle_crop_requested(1, poly)
        check(True, "非 Axial 平面时截取直接返回，不弹框")
        vi.views[1]['plane'] = AXIAL

        asked = {'n': 0, 'msg': ''}

        def _q(_p, _t, msg='', *a, **k):
            asked['n'] += 1; asked['msg'] = msg
            return QMessageBox.No

        QMessageBox.question = staticmethod(_q)
        vi.handle_crop_requested(1, poly)
        check(asked['n'] == 1, "Axial 上截取弹出统计框")
        check('mm' in asked['msg'] and 'HU' in asked['msg'],
              f"统计框给出面积与均值 HU（「{asked['msg'][:40].replace(chr(10), ' / ')}」）")
        check(not _glob.glob(os.path.join(tmp, '*.png')), "选「否」时不写任何文件")

        out_png = os.path.join(tmp, "crop.png")
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_png, "PNG (*.png)"))
        vi.handle_crop_requested(1, poly)
        check(os.path.exists(out_png), "选「是」时写出裁剪 PNG")
        log = os.path.join(tmp, "export_log.csv")
        check(os.path.exists(log), "同目录追加 export_log.csv")
        if os.path.exists(log):
            with open(log, encoding='utf-8-sig') as f:
                last = [r for r in _csv2.reader(f) if r][-1]
            check(len(last) >= 4 and last[1] == '3',
                  f"日志记录切片号与面积/均值（{last}）")

        # 日志写入失败必须提示而不是静默
        warned = {'n': 0}
        QMessageBox.warning = staticmethod(lambda *a, **k: warned.__setitem__('n', warned['n'] + 1))
        with _mock.patch('builtins.open', side_effect=OSError("disk full")):
            vi.handle_crop_requested(1, poly)
        check(warned['n'] >= 1, "日志写入失败时弹出警告，不静默吞掉")

        # 标注 CRUD
        vi.global_annotations = {'all': []}
        vi.chk_global_scope.setChecked(False)
        # 故意传数字 id：渲染层把 id 塞进 setToolTip（只收 str），删除时又从 toolTip
        # 取回比对，故入口必须规范成 str，否则该标注既画不出来也删不掉
        vi.handle_annotation_added({'id': 7, 'type': 'ruler', 'p1': (1, 1), 'p2': (5, 5)})
        check(any(a.get('id') == '7' for a in vi.global_annotations.get(2, [])),
              "新标注归入当前切片，且 id 被规范成字符串")
        vi.chk_global_scope.setChecked(True)
        vi.handle_annotation_added({'id': 8, 'type': 'ruler', 'p1': (2, 2), 'p2': (6, 6)})
        check(any(a.get('id') == '8' for a in vi.global_annotations.get('all', [])),
              "勾选穿透后新标注归入 all（id 同样规范为字符串）")
        vi.update_display(); app.processEvents()
        check(True, "数字 id 的标注能正常渲染（规范化前会抛 TypeError 被吞掉）")
        vi.handle_annotation_deleted('7')      # 删除侧拿到的是 toolTip 字符串
        check(not any(str(a.get('id')) == '7' for x in vi.global_annotations.values() for a in x),
              "按 toolTip 字符串删除生效——规范化前这里对不上，删不掉")

        # 图例显隐
        vi.volume_mask = np.zeros((Z, H, W), np.uint8); vi.volume_mask[2, 12:24, 12:24] = 5
        vi._hidden_organs.clear()
        vi._toggle_organ("organ:5")
        check(5 in vi._hidden_organs, "点击图例隐藏该器官")
        vi._toggle_organ("organ:5")
        check(5 not in vi._hidden_organs, "再次点击恢复显示")
        vi._toggle_organ("organ:abc"); vi._toggle_organ("nocolon")
        check(True, "图例 href 畸形时安全忽略")

        # 定量面板的置信度显示：有 conf / 无 conf / 部分手工改动
        vi.volume_conf = None
        vi._update_organ_stats(); app.processEvents()
        t_noconf = vi.lbl_ai_stats.text()
        check('conf' not in t_noconf, "无置信度时面板不显示 conf 字样")
        vi.volume_conf = np.full((Z, H, W), 230, np.uint8)
        vi._update_organ_stats(); app.processEvents()
        check('conf' in vi.lbl_ai_stats.text(), "有置信度时面板显示 conf")
        vi.volume_conf[2, 12:18, 12:24] = 0          # 一半标成哨兵＝手工改过
        vi._update_organ_stats(); app.processEvents()
        t_part = vi.lbl_ai_stats.text()
        check(('模型判定' in t_part) or ('model' in t_part),
              "部分体素被手工改动时标出模型判定占比")
        vi.volume_mask[2, 12:24, 12:24] = MANUAL_TRACK_LABEL
        vi.volume_conf[vi.volume_mask == MANUAL_TRACK_LABEL] = 0
        vi._update_organ_stats(); app.processEvents()
        check('conf' not in vi.lbl_ai_stats.text(),
              "整层都是手动追踪时不报置信度（模型对它没有判断）")
    finally:
        QMessageBox.question, QFileDialog.getSaveFileName = saved_q, saved_s
        QMessageBox.warning = saved_w
        shutil.rmtree(tmp, ignore_errors=True)
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_model_card_fallback():
    """模型说明卡在实验产物缺失/损坏时的回退：如实说「未提供」，绝不编数字。

    这是卡片最该被信任的性质——它整篇的立足点就是「每个数字都来自实验产物」。
    产物不在场时若还能印出一个好看的数，整张卡片的可信度就没了。
    把 _RESULTS 指向空目录与坏文件来走这些路径（此前零覆盖）。
    """
    print("[模型说明卡：产物缺失时的回退]")
    import shutil
    import tempfile

    import model_card
    saved = model_card._RESULTS
    tmp = tempfile.mkdtemp()
    try:
        model_card._RESULTS = tmp                     # 空目录：所有读取都应落空
        for en in (False, True):
            txt = model_card.build_model_card(en)
            check(len(txt) > 200, f"{'英文' if en else '中文'}卡片仍能生成（不因缺产物而崩）")
            check(('not found' in txt) or ('未在' in txt) or ('未找到' in txt),
                  "  明说验证结果未提供")
            check(('has not been measured' in txt) or ('尚未在本机测量' in txt),
                  "  spacing 一节退回「未测量」表述")
            import re as _re
            nums = _re.findall(r'\b0\.\d{3,4}\b', txt)
            check(not nums, f"  没有任何 Dice 数字被凭空印出（发现 {nums}）")

        # 损坏的 CSV：读取端必须吞掉而不是把异常抛到 UI
        for name in ("seg_dice.csv", "seg_spacing.csv", "seg_multi.csv",
                     "seg_spacing_fix_multi.csv", "seg_spacing_fix.csv"):
            with open(os.path.join(tmp, name), 'w', encoding='utf-8') as f:
                f.write("这不是 CSV\x00\n乱码,,,\n")
        with open(os.path.join(tmp, "seg3d_teacher_summary.json"), 'w', encoding='utf-8') as f:
            f.write("{ 不是合法 JSON")
        txt = model_card.build_model_card(False)
        check(len(txt) > 200, "产物损坏时卡片照常生成，不把异常抛到界面")
        check(len(model_card.card_title(False)) > 0, "标题不受产物状态影响")
    finally:
        model_card._RESULTS = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_matrix_recon_ui(app):
    """DMR / ART / SIRT 的 UI 调度：四视图分配、标题与 RMSE、链式源图、异常安全退出。

    **只测调度，不测数值**——数值正确性已由 test_recon_numerics 在纯函数层覆盖。
    系统矩阵在此被替换成手工小矩阵：真构建要 O(n²) 次 Radon 且走 multiprocessing，
    本机磁盘缓存里那份 32² 矩阵有 23MB，CI 环境没有它，让测试依赖本地缓存或现算
    几分钟都不可接受。这一层此前零覆盖，而它正是重建实验室最核心的教学功能。
    """
    print("[矩阵/迭代重建的 UI 调度]")
    import unittest.mock as _mock
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        vi.tabs.setCurrentIndex(1); app.processEvents()

        # 无源图时两个入口都必须安全退出（此前靠 volume_hu 判空，改模体后由 _prepare 统一兜底）
        vi.run_dmr(); vi.run_art_sirt(); app.processEvents()
        check(True, "无源图时 DMR / ART 安全空转，不崩")

        vi.btn_phantom.click(); app.processEvents()      # 内置模体，不需要任何 DICOM
        vi.cb_matrix_size.setCurrentIndex(0)             # 取最小矩阵尺寸
        img_small, sino, theta, n = vi._prepare_small_image_and_sinogram()
        check(img_small is not None and img_small.shape == (n, n),
              f"模体经 _prepare 得到 {n}×{n} 小图与 {None if sino is None else sino.shape} 弦图")

        rs = np.random.RandomState(0)
        fake_A = rs.rand(sino.size, n * n).astype(np.float32) * 0.01

        with _mock.patch.object(type(vi), '_build_system_matrix', lambda s, nn, th: fake_A):
            vi.run_dmr(); app.processEvents()
            check(vi._last_recon_img is not None and vi._last_recon_img.shape == (n, n),
                  "DMR 结果存入 _last_recon_img，供「生成弦图」链式再投影")
            t3 = vi.views[3]['title_label'].text()
            check('RMSE' in t3, f"V3 标题带 RMSE（得「{t3}」）")
            check(f"{n}x{n}" in vi.views[4]['title_label'].text(), "V4 标题标出重建尺寸")
            check('DMR' in vi.lbl_time.text() or '矩阵' in vi.lbl_time.text(),
                  f"耗时栏标明算法（得「{vi.lbl_time.text()}」）")
            for vid in (1, 2, 3, 4):
                pm = vi.views[vid]['view'].image_item.pixmap()
                check(not pm.isNull(), f"V{vid} 已渲染（{pm.width()}×{pm.height()}）")

            for meth in ('ART', 'SIRT'):
                idx = vi.cb_art_method.findText(meth)
                if idx < 0:
                    continue
                vi.cb_art_method.setCurrentIndex(idx)
                vi.cb_art_iter.setCurrentIndex(0)
                vi._last_recon_img = None
                vi.run_art_sirt(); app.processEvents()
                check(vi._last_recon_img is not None, f"{meth} 跑通并留下重建结果")
                check(meth in vi.views[4]['title_label'].text() or 'RMSE' in vi.views[3]['title_label'].text(),
                      f"{meth} 更新了视图标题")

        # 系统矩阵构建失败（返回 None）时必须安全退出，而不是拿 None 去算
        with _mock.patch.object(type(vi), '_build_system_matrix', lambda s, nn, th: None):
            vi.run_dmr(); vi.run_art_sirt(); app.processEvents()
            check(True, "系统矩阵构建失败时安全退出，不把 None 传给求解器")
    finally:
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_probe_hu(app):
    """探针读数：三平面索引映射正确，且读不出时必须清空而不是留旧值。

    实测动机：旧实现把整段包在 `try/except: pass` 里，坐标越界或 plane 取到三者之外
    时静默失败，标签继续显示**上一次**的读数——连坐标都是旧的，看上去完全像一次
    有效读数。对会被直接用于判读的 HU 值，陈旧显示比空白危险得多。
    interaction.py 是全项目覆盖率最低的一层（64%），这些路径此前无人走过。
    """
    print("[探针 HU 读数：索引映射与失效清空]")
    import unittest.mock as _mock

    from PySide6.QtCore import QPointF

    from constants import SAGITTAL, TOOL_POINTER
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 8, 32, 32
        vi.volume_hu = np.arange(Z * H * W, dtype=np.float32).reshape(Z, H, W)
        vi.dicom_datasets = [None] * Z
        _mark_supported_capabilities(vi)
        vi.current_3d_pos = [4, 16, 16]
        vi.active_tool = TOOL_POINTER
        view_t = type(vi.views[1]['view'])

        def probe(coord, plane):
            vi.views[1]['plane'] = plane
            with _mock.patch.object(view_t, 'get_real_coordinates', lambda s, p, _c=coord: _c):
                vi.measure_hu(QPointF(0, 0), 1)
            return vi.lbl_hu_value.text()

        # 鼠标 (cx,cy)=(20,5)：三平面各自的 (z,y,x) 映射，与 mpr_geometry 同一套约定
        for pl, nm, exp in ((AXIAL, 'Axial', (4, 5, 20)),
                            (CORONAL, 'Coronal', (2, 16, 20)),
                            (SAGITTAL, 'Sagittal', (2, 20, 16))):
            txt = probe((20, 5), pl)
            want = float(vi.volume_hu[exp])
            check(f"{want:.1f} HU" in txt, f"{nm} 读到体素 {exp} = {want:.1f}（得「{txt}」）")

        probe((20, 5), AXIAL)                      # 先留下一个有效读数
        check(bool(vi.lbl_hu_value.text()), "有效读数已显示")
        for coord, plane, why in (((999, 999), AXIAL, "坐标越界"),
                                  ((-1, 5), AXIAL, "负坐标"),
                                  ((20, 5), 99, "plane 不在三平面之内"),
                                  (None, AXIAL, "鼠标在图像之外")):
            check(probe(coord, plane) == "", f"{why} → 清空，不残留上一次的读数")
        check(f"{float(vi.volume_hu[(4, 5, 20)]):.1f} HU" in probe((20, 5), AXIAL),
              "回到有效区域后立即恢复读数")
    finally:
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_wheel_and_cine(app):
    """滚轮翻页与 Cine 往返：三平面各走各的轴、边界钳制、对比模式下只翻主序列。

    这两段此前是 interaction.py 里零覆盖的部分，而滚轮是用户最高频的交互。
    三平面走的是**两条不同的代码路径**（Axial 经切片滑条的信号链，冠/矢状面直接改
    current_3d_pos 再重绘），这种不对称最容易在改动中被弄坏，故逐条钉住。
    """
    print("[滚轮翻页与 Cine 往返]")
    from constants import SAGITTAL
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 10, 24, 28
        vi.volume_hu = np.zeros((Z, H, W), np.float32)
        vi.dicom_datasets = [None] * Z
        vi.slider_slice.setRange(0, Z - 1); vi.slider_slice.setValue(5)
        vi.current_3d_pos = [5, 12, 14]

        # 滚轮向上(d>0)=上一层，向下=下一层；这是各家阅片软件的通行约定
        vi.views[1]['plane'] = AXIAL
        vi.on_wheel_mpr(1, 1); app.processEvents()
        check(vi.current_3d_pos[0] == 4, f"Axial 向上滚 → z 5→{vi.current_3d_pos[0]}（期望 4）")
        vi.on_wheel_mpr(-1, 1); app.processEvents()
        check(vi.current_3d_pos[0] == 5, f"Axial 向下滚 → z 回到 {vi.current_3d_pos[0]}")

        vi.views[1]['plane'] = CORONAL
        vi.on_wheel_mpr(1, 1); app.processEvents()
        check(vi.current_3d_pos[1] == 11 and vi.current_3d_pos[0] == 5,
              f"Coronal 滚轮只动 y（{vi.current_3d_pos}），z 不受影响")
        vi.views[1]['plane'] = SAGITTAL
        vi.on_wheel_mpr(1, 1); app.processEvents()
        check(vi.current_3d_pos[2] == 13 and vi.current_3d_pos[1] == 11,
              f"Sagittal 滚轮只动 x（{vi.current_3d_pos}），y 不受影响")

        # 边界钳制：三个轴各自到头后不得越界
        vi.views[1]['plane'] = AXIAL; vi.slider_slice.setValue(0); app.processEvents()
        for _ in range(3): vi.on_wheel_mpr(1, 1)
        app.processEvents()
        check(vi.current_3d_pos[0] == 0, f"Axial 到顶后钳制在 0（得 {vi.current_3d_pos[0]}）")
        vi.slider_slice.setValue(Z - 1); app.processEvents()
        for _ in range(3): vi.on_wheel_mpr(-1, 1)
        app.processEvents()
        check(vi.current_3d_pos[0] == Z - 1, f"Axial 到底后钳制在 {Z-1}（得 {vi.current_3d_pos[0]}）")
        vi.views[1]['plane'] = CORONAL; vi.current_3d_pos[1] = 0
        for _ in range(3): vi.on_wheel_mpr(1, 1)
        check(vi.current_3d_pos[1] == 0, f"Coronal 到边界钳制在 0（得 {vi.current_3d_pos[1]}）")
        vi.current_3d_pos[1] = H - 1
        for _ in range(3): vi.on_wheel_mpr(-1, 1)
        check(vi.current_3d_pos[1] == H - 1, f"Coronal 另一端钳制在 {H-1}")

        # 对比模式：无论视图是哪个平面，滚轮都只翻主序列切片
        vi.views[1]['plane'] = SAGITTAL
        vi.compare_mode_active = True
        vi.slider_slice.setValue(5); vi.current_3d_pos = [5, 12, 14]; app.processEvents()
        bx = vi.current_3d_pos[2]
        vi.on_wheel_mpr(1, 1); app.processEvents()
        check(vi.current_3d_pos[0] == 4 and vi.current_3d_pos[2] == bx,
              f"对比模式下滚轮只翻主序列 z（{vi.current_3d_pos}），不动矢状面的 x")
        vi.compare_mode_active = False

        # 重建实验室模式下滚轮不应干扰重建流水线
        vi.recon_mode_active = True
        before = list(vi.current_3d_pos)
        vi.on_wheel_mpr(1, 1); app.processEvents()
        check(list(vi.current_3d_pos) == before, "重建实验室模式下滚轮不改动光标")
        vi.recon_mode_active = False

        # Cine 往返：到顶/到底反向，不跳变回环
        vi.slider_slice.setValue(Z - 1); vi._cine_dir = 1; app.processEvents()
        vi._cine_step(); app.processEvents()
        check(vi.slider_slice.value() == Z - 2 and vi._cine_dir == -1,
              f"到底后反向（值 {vi.slider_slice.value()}，方向 {vi._cine_dir}）")
        vi.slider_slice.setValue(0); vi._cine_dir = -1; app.processEvents()
        vi._cine_step(); app.processEvents()
        check(vi.slider_slice.value() == 1 and vi._cine_dir == 1,
              f"到顶后反向（值 {vi.slider_slice.value()}，方向 {vi._cine_dir}）")
        vi.volume_hu = None
        vi.cine_timer.start(50)
        vi._cine_step()
        check(not vi.cine_timer.isActive(), "数据被清空后 Cine 自行停止，不空转")
    finally:
        if vi is not None:
            vi.cine_timer.stop()
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_spacing_resample():
    """nnU-Net 的 spacing 重采样契约：该做时做、不该做时不做、做完必须回到原网格。

    实测依据（experiments/seg_spacing.py）：spacing 偏离训练值一倍即掉 13% Dice，
    故这一步不是可选优化。三条不做的路径各有理由，也一并钉住：
    spacing 未知（猜不得）、已足够接近（插值反而有损）、放大后会 OOM（宁可失配也别崩）。
    用假 session 顶替 ONNX，不加载 119MB 权重。
    """
    print("[spacing 重采样（nnU-Net 推理契约）]")
    import ai_engine
    saved = ai_engine._get_session

    class _Fake:
        def get_inputs(self):
            return [type('I', (), {'name': 'x'})()]

        def run(self, _, feed):
            b = feed['x']
            o = np.zeros((1, 25, b.shape[2], b.shape[3], b.shape[4]), np.float32)
            o[0, 5] = 40.0
            return [o]

    ai_engine._get_session = lambda p: _Fake()
    try:
        Z, H, W = 40, 60, 60
        vol = np.full((Z, H, W), -500.0, np.float32)

        def make(sp):
            e = ai_engine.AutoAIEngineThread(vol, lambda *a: None, spacing=sp)
            return e, e._plan_resample()

        # 决策层：三条「不做」的路径
        check(make(None)[1] is None, "spacing 未知 → 不重采样（不基于猜测做插值）")
        check(make((1.5, 1.52, 1.48))[1] is None, "已在 1.5mm 的 5% 内 → 不重采样（免去无谓插值损失）")
        check(make((0.0, 1.0, 1.0))[1] is None, "spacing 含非法值 → 不重采样")
        # 上限保护：直接把阈值临时压到测试体积之下，测的是判定逻辑本身而非某个具体数值。
        # 真实场景中它很少触发——重采样到固定 1.5mm 后体素数只由扫描 FOV 决定
        # （胸腹 CT 约 400mm³ → 恒约 19M），与原始 spacing 无关；能超限的是全身长扫描。
        sv = ai_engine._MAX_RESAMPLED_VOXELS
        ai_engine._MAX_RESAMPLED_VOXELS = 1000
        try:
            check(make((3.0, 3.0, 3.0))[1] is None, "放大后超出体素上限 → 跳过重采样而不是 OOM")
        finally:
            ai_engine._MAX_RESAMPLED_VOXELS = sv

        # RIDER 的真实几何：0.712891mm in-plane / 1.25mm 层厚，应当触发且是缩小
        e, plan = make((1.25, 0.712891, 0.712891))
        check(plan is not None, "RIDER 几何触发重采样")
        f, shp = plan
        check(int(np.prod(shp)) < Z * H * W,
              f"细 spacing 重采样是缩小：{Z*H*W} → {int(np.prod(shp))} 体素（更快更省内存）")
        exp_h = round(H * 0.712891 / 1.5)
        check(shp[1] == exp_h, f"缩放按物理尺寸算（H {H}→{shp[1]}，期望 {exp_h}）")

        # 端到端：输出必须回到原网格，否则回调的 shape 校验会丢弃整次推理
        e._run_body()
        check(e.confidence is not None and e.confidence.shape == (Z, H, W),
              f"置信度回到原网格 {None if e.confidence is None else e.confidence.shape}")
        check(e.resampled_from is not None and e.resampled_from[0] == (Z, H, W),
              "引擎记录了本次确实做过重采样，供 UI 如实告知")

        got = {}
        e2 = ai_engine.AutoAIEngineThread(vol, lambda m, t: got.update(mask=m),
                                          spacing=(1.25, 0.712891, 0.712891))
        e2._run_body()
        m = got.get('mask')
        check(m is not None and m.shape == (Z, H, W),
              f"标签图回到原网格 {None if m is None else m.shape}")
        check(m is not None and bool((m == 5).all()), "标签值在往返缩放中未被插值破坏")

        # 代价必须可见：边界在 1.5mm 网格上决定，映射回细分辨率后是阶梯状。
        # 用一个球形结构实测台阶长度，钉住「这不是纯赚」这个事实。
        class _Ball:
            def get_inputs(self):
                return [type('I', (), {'name': 'x'})()]

            def run(self, _, feed):
                b = feed['x']
                d, h, w = b.shape[2], b.shape[3], b.shape[4]
                zz, yy, xx = np.ogrid[:d, :h, :w]
                ball = ((zz - d / 2) ** 2 + (yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(d, h, w) / 3) ** 2
                o = np.zeros((1, 25, d, h, w), np.float32)
                o[0, 0] = 10.0; o[0, 5][ball] = 40.0
                return [o]

        ai_engine._get_session = lambda p: _Ball()
        got2 = {}
        e3 = ai_engine.AutoAIEngineThread(vol, lambda m_, t: got2.update(m=m_),
                                          spacing=(0.75, 0.75, 0.75))   # 缩放正好 2 倍
        e3._run_body()
        mm = got2['m']
        mid = mm[mm.shape[0] // 2]
        rows = np.where((mid == 5).any(1))[0]
        if len(rows) > 3:
            firsts = [np.where(mid[r] == 5)[0][0] for r in rows]
            from itertools import groupby
            plateau = float(np.median([len(list(g)) for _, g in groupby(firsts)]))
            check(plateau >= 1.5,
                  f"边界按重采样网格量化，台阶中位 {plateau:.1f} 像素（2× 缩放 → 预期约 2）")
    finally:
        ai_engine._get_session = saved


def test_phantom():
    """内置 Shepp-Logan 模体：解析生成的正确性，以及与 skimage 参考实现的结构一致性。

    钉住取的是 **Toft 修订版**而非 1974 原版——原版病灶对比度仅 0.01，几乎不可见，
    且研究一（recon_study.py）用的就是 skimage 的修订版；两边若各用一版，
    实验室与实验报出的 RMSE / 对比度数字不可比。
    """
    print("[内置 Shepp-Logan 模体]")
    import recon as R
    for n in (32, 64, 128, 256):
        p = R.shepp_logan(n)
        check(p.shape == (n, n) and p.dtype == np.float32, f"n={n}: 形状/类型 {p.shape} {p.dtype}")
        check(0.0 <= float(p.min()) and float(p.max()) <= 1.0,
              f"n={n}: 值域 [{p.min():.2f}, {p.max():.2f}] ⊂ [0,1]，与切片归一化同口径")
        # 圆形掩码：四角必须为 0，否则误差图会在角落显示虚假大误差
        check(float(p[0, 0]) == 0.0 and float(p[0, -1]) == 0.0 and float(p[-1, -1]) == 0.0,
              f"n={n}: 圆掩码已施加（四角为 0）")
    try:
        R.shepp_logan(1)
        check(False, "尺寸过小时应抛 ValueError")
    except ValueError:
        check(True, "尺寸过小时抛 ValueError 而非产出畸形数组")

    p = R.shepp_logan(256)
    lv = np.unique(np.round(p[p > 0], 3))
    has = lambda x: bool(np.isclose(lv, x, atol=2e-3).any())  # noqa: E731  float32 尾数，需容差
    # 修订版的净灰度：脑实质 0.2、病灶 0.3、颅骨 1.0（0.1/0.4 为病灶与外圈的叠加处）
    check(has(1.0) and has(0.2) and has(0.3),
          f"呈现修订版灰度层级 {[round(float(x), 3) for x in lv]}")
    check(not has(0.02), "不是 1974 原版的 0.02 低对比度（那一版肉眼近乎不可见）")

    try:
        from scipy import ndimage as _nd
        from skimage.data import shepp_logan_phantom
        ref = _nd.zoom(shepp_logan_phantom().astype(np.float32), 256 / 400, order=1)
        ref = np.clip(ref, 0, 1) * R._circle_mask(256)
        a, b = p - p.mean(), ref - ref.mean()
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        # 剩余差异来自参考实现是 400² 位图再插值，本实现为解析生成（边界更锐）
        check(ncc > 0.94, f"与 skimage 参考的归一化互相关 NCC={ncc:.4f} > 0.94")
    except ImportError:
        check(True, "skimage 参考不可用，跳过互相关比对")


def test_phantom_recon_flow(app):
    """空载启动即可跑通重建：不导入任何 DICOM，模体 → 弦图 → BP/FBP/DFR。

    重建实验室原先必须先导入数据才能用，空载打开是死的。DMR/ART 不在此测——
    它们要建系统矩阵（分钟级 + 子进程），另有专门用例覆盖数值正确性。
    """
    print("[空载模体重建链路]")
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        check(vi.volume_hu is None, "空载启动，无任何数据")
        vi.tabs.setCurrentIndex(1); app.processEvents()
        check(vi.recon_mode_active, "无数据也能进入重建实验室")
        vi.generate_sinogram(); app.processEvents()
        check(vi.current_sinogram is None, "无源图时生成弦图安全空转，不崩")

        vi.btn_phantom.click(); app.processEvents()
        check(vi._phantom_img is not None, "载入内置模体")
        t1 = vi.views[1]['title_label'].text()
        check(('模体' in t1) or ('Phantom' in t1),
              f"V1 标题随之改为模体（得「{t1}」）——否则界面挂着「真实切片」在说谎")
        src, lab = vi._recon_source_slice()
        check(src is not None and lab in ("模体", "Phantom"), f"重建源切到模体（{lab}）")
        vi.generate_sinogram(); app.processEvents()
        check(vi.current_sinogram is not None and np.isfinite(vi.current_sinogram).all(),
              f"弦图生成且全为有限值 {None if vi.current_sinogram is None else vi.current_sinogram.shape}")
        for nm in ('btn_bp', 'btn_fbp', 'btn_dfr'):
            check(getattr(vi, nm).isEnabled(), f"{nm} 已启用")
        for nm, fn in (('BP', vi.run_bp), ('FBP', vi.run_fbp), ('DFR', vi.run_dfr)):
            fn(); app.processEvents()
            check(True, f"{nm} 在模体上跑通")

        # 空载下切 Tab 往返：update_display 因无数据提前返回，走不到标题修正路径，
        # 于是 V1 会挂着「真实切片」却显示模体（实测踩到）
        vi.tabs.setCurrentIndex(0); app.processEvents()
        vi.tabs.setCurrentIndex(1); app.processEvents()
        t2 = vi.views[1]['title_label'].text()
        check(('模体' in t2) or ('Phantom' in t2),
              f"切 Tab 往返后 V1 仍标为模体（得「{t2}」）")

        vi.btn_phantom.click(); app.processEvents()
        check(vi._phantom_img is None, "卸下模体")
        check(vi.current_sinogram is None,
              "卸下时清掉基于模体算出的弦图（否则模体弦图会配上真实原图）")
        check(not vi.btn_bp.isEnabled(), "重建按钮随之禁用，不留可点的死路")
    finally:
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_confidence_map():
    """置信度：softmax 最大类概率的正确性、量化误差、以及数学降级路径必须为 None。

    用手工构造的 logits 验证数值，不加载 organs.onnx、不跑真实推理。
    钉住的契约：给了 conf 才有 conf 列，没给则该键必须缺席——数学降级没有概率输出，
    此处若填一个默认值，下游会把它当成模型的置信度。
    """
    print("[逐体素置信度 softmax max-prob]")
    import quantify

    # 手工 logits：三个体素分别是「极确信」「两类难分」「均匀不确信」
    n_cls = 25
    lg = np.zeros((n_cls, 1, 1, 3), np.float32)
    lg[3, 0, 0, 0] = 50.0                       # 一类独大 → prob≈1
    lg[3, 0, 0, 1] = lg[7, 0, 0, 1] = 10.0      # 两类持平 → prob≈0.5
    # 第三个体素全 0 → 25 类均匀 → prob=1/25=0.04

    # 与 ai_engine 中完全一致的算法（in-place 减 max 再 exp，max-prob = 1/Σ）
    o = lg.copy()
    o -= o.max(0, keepdims=True)
    np.exp(o, out=o)
    conf = 1.0 / o.sum(0)
    exp = [1.0, 0.5, 1.0 / n_cls]
    for k, want in enumerate(exp):
        check(abs(float(conf[0, 0, k]) - want) < 1e-3,
              f"体素{k} max-prob={float(conf[0, 0, k]):.4f}（期望 {want:.4f}）")
    check(float(conf.min()) > 0 and float(conf.max()) <= 1.0 + 1e-6,
          "置信度落在 (0, 1] 内")

    u8 = (conf * 255.0).astype(np.uint8)
    back = u8.astype(np.float32) / 255.0
    check(float(np.abs(back - conf).max()) <= 1.0 / 255.0,
          f"uint8 量化误差 ≤ 1/255（实测 {float(np.abs(back - conf).max()):.5f}）")

    # 定量表：给了 conf 才有列，没给必须缺席
    vol = np.full((2, 4, 4), 50.0, np.float32)
    mk = np.zeros((2, 4, 4), np.uint8); mk[0, :2, :2] = 5
    cf = np.full((2, 4, 4), 204, np.uint8)       # 204/255 = 0.8
    names = {5: ("肝", "Liver")}
    r_no = quantify.compute_organ_stats(vol, mk, (1, 1, 1), names)
    r_yes = quantify.compute_organ_stats(vol, mk, (1, 1, 1), names, cf)
    check('mean_conf' not in r_no[0], "不传 conf：定量行里没有 mean_conf 键")
    check('mean_conf' in r_yes[0] and abs(r_yes[0]['mean_conf'] - 0.8) < 1e-2,
          f"传了 conf：mean_conf={r_yes[0].get('mean_conf', float('nan')):.3f}（期望 0.800）")
    # 形状不符必须被拒，而不是崩或算出错位的数
    r_bad = quantify.compute_organ_stats(vol, mk, (1, 1, 1), names, np.zeros((3, 4, 4), np.uint8))
    check('mean_conf' not in r_bad[0], "conf 形状不符时该列缺席，不静默错位")

    # conf==0 是「无模型置信度」哨兵（手动追踪 / 画笔改过的体素）。
    # 审计发现的真实缺陷：手动追踪层曾被报出 conf=0.80——那其实是模型对该处
    # 【原本那个器官】的置信度，与用户手画的东西毫无关系。
    from constants import MANUAL_TRACK_LABEL as MTL
    mk2 = np.zeros((2, 4, 4), np.uint8); mk2[0, :2, :2] = 5; mk2[1, :2, :2] = MTL
    cf2 = np.full((2, 4, 4), 204, np.uint8); cf2[1, :2, :2] = 0
    nm2 = {5: ("肝", "Liver"), MTL: ("手动追踪", "Manual")}
    r2 = {r['id']: r for r in quantify.compute_organ_stats(vol, mk2, (1, 1, 1), nm2, cf2)}
    check('mean_conf' not in r2[MTL], "手动追踪层不报置信度（模型对它没有判断）")
    check('mean_conf' in r2[5], "同一蒙版里的模型器官照常报置信度")

    mk3 = np.zeros((2, 4, 4), np.uint8); mk3[0] = 5
    cf3 = np.full((2, 4, 4), 204, np.uint8); cf3[0, :2, :] = 0    # 一半被画笔改过
    r3 = quantify.compute_organ_stats(vol, mk3, (1, 1, 1), {5: ("肝", "Liver")}, cf3)[0]
    check(abs(r3['conf_cover'] - 0.5) < 1e-6,
          f"conf_cover 报出模型判定体素占比（{r3['conf_cover']:.2f}，期望 0.50）")
    check(abs(r3['mean_conf'] - 0.8) < 1e-2, "均值只统计模型体素，不被哨兵 0 拉低")

    # 端到端跑一遍引擎的滑窗路径：用假 session 顶替 ONNX，不加载 119MB 权重。
    # 重点验证 in-place 改写 out 不炸（真实 ORT 输出实测 writeable=True，此处用可写数组同构）
    import ai_engine
    saved = ai_engine._get_session
    Z, H, W = 40, 30, 20                     # 非 32 倍数，同时覆盖 z/y/x 三个方向的 pad

    class _FakeSess:
        def get_inputs(self):
            return [type('I', (), {'name': 'input_image'})()]

        def run(self, _, feed):
            b = feed['input_image']
            d, h, w = b.shape[2], b.shape[3], b.shape[4]
            o = np.zeros((1, n_cls, d, h, w), np.float32)
            # 沿 x 铺三种已知的 softmax 情形，让引擎的置信度公式被真正钉住：
            # 处处 label 5 胜出（保持既有 argmax 断言成立），但胜出幅度不同。
            #   x%3==0 一类独大  -> max-prob≈1.0
            #   x%3==1 与 7 号并列 -> max-prob=0.5（并列时 argmax 取小者，仍是 5）
            #   x%3==2 幅度趋零   -> max-prob≈1/25=0.04
            o[0, 5, :, :, 0::3] = 40.0
            o[0, 5, :, :, 1::3] = 40.0
            o[0, 7, :, :, 1::3] = 40.0
            o[0, 5, :, :, 2::3] = 1e-6
            return [o]

    ai_engine._get_session = lambda p: _FakeSess()
    try:
        eng = ai_engine.AutoAIEngineThread(np.zeros((Z, H, W), np.float32), lambda *a: None)
        seg = eng._run_onnx_multiorgan(np.zeros((Z, H, W), np.float32))
        check(seg is not None and seg.shape == (Z, H, W), f"滑窗输出形状 {None if seg is None else seg.shape}")
        check(eng.confidence is not None and eng.confidence.shape == (Z, H, W),
              "引擎把 confidence 作为实例属性带出，形状与标签图一致")
        check(eng.confidence.dtype == np.uint8, f"confidence 为 uint8（{eng.confidence.dtype}）")
        # 三档取值直接断言引擎产出的 uint8，而不是测试自己再算一遍 softmax。
        # 旧版在测试里复刻了 `1.0 / out.sum(0)`，于是把 ai_engine 的公式改成
        # `1.0 / out.sum(0) ** 2` 时全部 495 项仍然通过——形同虚设。
        got = [int(eng.confidence[Z // 2, H // 2, x]) for x in (0, 1, 2)]
        want = [255, 127, 10]                # 一类独大 / 二类并列 / 均匀
        check(got == want,
              f"置信度三档与 softmax max-prob 一致：实得 {got}，期望 {want}"
              "（1.0 / 0.5 / 0.04 量化到 uint8）")
        check(int(eng.confidence.max()) == 255 and int(eng.confidence.min()) >= 1,
              f"置信度落在 [1,255]，0 留作哨兵（min={int(eng.confidence.min())}）")
        check(bool((seg == 5).all()), "标签图取到 argmax 所指类别")
    finally:
        ai_engine._get_session = saved


def test_zero_grade_guards(app, m):
    """四条会产出错误数字/错误影像的缺陷的回归锁。均无需真实数据。

    由来：一轮子代理审查报出九条「归零级」缺陷，而当时 498 项回归**一条都没覆盖**。
    共性是「静默给出一个看起来正常、实则错误的数」，正是最难靠肉眼发现的一类。
    """
    print("[归零级守卫：spacing 缺省 / 撤销栈 / NaN 排序 / 非 Axial 标注]")
    from PySide6.QtGui import QPixmap

    from constants import AXIAL, CORONAL

    v = m.MedicalViewer(); app.processEvents()
    if v.ai_thread:
        v.ai_thread.cancel()

    # ① set_image 的 pixel_spacing 缺省必须是「保持」而非「覆盖」。
    #    曾缺省 (1.0,1.0) 并无条件赋值，于是 compare_lab 刷新蒙版时把真实间距抹平：
    #    同样 100 像素，临床模式 69.9 mm、对比模式 99.9 mm。
    view = v.views[1]['view']
    view.pixel_spacing = (0.7, 0.7)
    view.set_image(QPixmap(8, 8))                      # 不传 spacing
    check(view.pixel_spacing == (0.7, 0.7),
          f"set_image 不传 spacing 时保持原值（得 {view.pixel_spacing}）")
    view.set_image(QPixmap(8, 8), None, (1.5, 1.5))    # 显式传才覆盖
    check(view.pixel_spacing == (1.5, 1.5),
          f"显式传入才覆盖（得 {view.pixel_spacing}）")

    # ② AI 回调整卷换蒙版必须清撤销栈。栈里是推理【开始前】的切片快照，
    #    推理期间用户可以画笔编辑；不清的话一次 Ctrl+Z 会把该层 AI 分割整层抹掉。
    v.volume_hu = np.zeros((3, 4, 4), np.float32)
    v.volume_mask = np.zeros((3, 4, 4), np.uint8)
    v._mask_undo = [(0, np.zeros((4, 4), np.uint8), None)]
    v._ai_generation = 0
    v.recon_mode_active = True          # 抑制 update_display（无 DICOM 时会越界），只验状态
    try:
        v.on_auto_ai_finished(np.ones((3, 4, 4), np.uint8), 1.0, generation=0)
    finally:
        v.recon_mode_active = False
    check(v._mask_undo == [],
          f"AI 结果落地后撤销栈被清空（剩 {len(v._mask_undo)} 条）")
    check(bool(v.volume_mask.any()), "AI 蒙版确实落地了（前一条不是因为没跑到）")

    # ③ 非 Axial 平面的标注必须被拒绝，而不是按 axial 层号错存。
    # 保持 synthetic volume / DICOM 列表长度一致；旧 fixture 留空列表，使下面合法的
    # Axial 标注在 update_display 中抛 IndexError，却被 Qt signal/slot 静默吞掉。
    v.dicom_datasets = [None] * 3
    v.volume_hu = np.zeros((3, 4, 4), np.float32)
    v.current_3d_pos = [1, 0, 0]
    v.global_annotations = {}
    v._warned_nonaxial_anno = True          # 抑制模态框，只验状态
    v.views[1]['plane'] = CORONAL
    v.views[1]['view'].annotation_added.emit(
        {'id': 'x1', 'type': 'ruler', 'p1': [0, 0], 'p2': [3, 3]})
    app.processEvents()
    check(all(not vv for vv in v.global_annotations.values()) or not v.global_annotations,
          f"冠状面标注被拒绝，未入库（现有 {v.global_annotations}）")
    v.views[1]['plane'] = AXIAL
    v.views[1]['view'].annotation_added.emit(
        {'id': 'x2', 'type': 'ruler', 'p1': [0, 0], 'p2': [3, 3]})
    app.processEvents()
    check(any(a['id'] == 'x2' for lst in v.global_annotations.values() for a in lst),
          "横断面标注正常入库（守卫未误伤）")


def test_withdrawn_claims_stay_withdrawn():
    """已撤回的结论不得回潮，且文档引用的地板值必须与产物一致。

    由来：研究一有两条结论被实测推翻——「ART 在所有剂量下最鲁棒」（实为 5:100 轮
    的算力失衡所致，各自取最优时 SIRT 全胜）与「≈180 视角后收益递减 ⇒ 剂量够了」
    （实为重建链路的离散化地板）。这两句在五个文件里出现过，改一处漏一处的风险很高，
    而且「悄悄改回去」不会被任何现有断言拦住。故锁成断言。

    纯文本 + 已入库 CSV，无 Qt / 无真实数据。
    """
    import csv as _csv
    import json as _json
    import re
    print("[已撤回结论的回潮防护]")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = ["README.md", "README.zh-CN.md", "docs/preprint_recon.md",
            "docs/technical_report.md", "experiments/README.md"]
    texts = {}
    for rel in docs:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            texts[rel] = open(p, encoding="utf-8").read()
    check(len(texts) == len(docs), f"五份文档齐备（实得 {len(texts)}）")

    # ① 两条已撤回结论不得以无限定的形式出现。
    #    【为什么从精确短语改成语义类别】原黑名单是 r"ART is the most robust" 一类的
    #    精确短语，而 technical_report.md 的实际措辞是 "(ART) is the most robust"
    #    （中间隔着右括号）、"constrained iteration is the most robust"（主语根本不
    #    出现 ART）、"achieves the lowest RMSE"。三种同义改写全部绕开黑名单，断言实
    #    测命中 0 条、一路绿灯，而那五处撤回本体从未进过该文件。精确短语锁不住改写。
    #    (a) ART 排名：可由就地限定语救活（说清是在固定迭代数下），故放行条件较宽。
    banned_rank = [r"is (the )?most robust", r"是已测方法中最鲁棒", r"各剂量最鲁棒",
                   r"achieves the lowest RMSE", r"best \(ART\)",
                   # 「ART is cleanest」写成「ART is **the** cleanest」就绕开了——
                   # 这一处正是在加固本断言的同一轮里第二次被同款改写漏掉的，故一律
                   # 用可选冠词，并把「推荐选 ART」这类祈使句也纳入。
                   r"ART is (the )?cleanest", r"choose a constrained iterative method \(ART\)",
                   r"(prefer|choose|use|select) ART\b"]
    excuse_rank = (r"withdraw|Withdraw|撤回|~~"
                   r"|under the fixed iteration counts|at the fixed iteration counts")
    #    (b) 「≈180 视角后收益递减 ⇒ 剂量够了」：这是结论本身被推翻，没有任何限定语
    #    能救，只有明写撤回才放行。放行条件**不能**包含「地板/不足以证明剂量够」之类
    #    的措辞——experiments/README.md 曾在同一行里先撤回再复述，若按那样放行，
    #    最典型的自相矛盾行反而会被自己的前半句豁免。
    banned_dose = [r"basis for [\"\u201c]?enough is enough",
                   r"[\"\u201c]diminishing-returns[\"\u201d] operating point",
                   r"[\"\u201c]enough is enough[\"\u201d] (dose|acquisition)"]
    excuse_dose = r"withdraw|Withdraw|撤回|~~|earlier revisions read"
    hits = []
    for rel, t in texts.items():
        for i, line in enumerate(t.splitlines(), 1):
            for pats, exc in ((banned_rank, excuse_rank), (banned_dose, excuse_dose)):
                if any(re.search(b, line) for b in pats) and not re.search(exc, line):
                    hits.append(f"{rel}:{i}")
                    break
    check(not hits, f"无未加撤回标注/限定的已撤回结论（违规 {len(hits)}：{hits[:5]}）")

    # ② 地板值必须与 exp_a_metric_floor.csv 的实测最小值一致
    fp = os.path.join(root, "experiments", "results", "exp_a_metric_floor.csv")
    check(os.path.exists(fp), "exp_a_metric_floor.csv 存在")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8-sig") as f:
            vals = [float(r["rmse_in_circle"]) for r in _csv.DictReader(f)]
        floor = min(vals)
        check(len(vals) >= 4, f"地板扫描至少 4 个视角档（实得 {len(vals)}）")
        # 判据不是「完全不变」（浮点与插值噪声在 1e-6 量级），而是两件事同时成立：
        #   (a) 最高三档彼此相差 < 0.1%，(b) 不再单调下降（末档不低于前一档）。
        # 后者是关键——仍在单调下降说明还没到底，那才是「剂量不足」。
        top3 = vals[-3:]
        spread = (max(top3) - min(top3)) / min(top3)
        check(spread < 1e-3, f"最高三档相对离散度 {spread:.2e} < 0.1%（已触底）")
        check(vals[-1] >= vals[-2] - 1e-9,
              f"末档不再低于前一档（{vals[-2]:.6f} → {vals[-1]:.6f}），即已停止下降")
        quoted = f"{floor:.5f}"
        where = [r for r, t in texts.items() if quoted in t]
        check(len(where) >= 2,
              f"地板值 {quoted} 至少在两份文档中被引用（实得 {len(where)}：{where}）")

    # ③ 聚类 CI 产物必须存在，且教师那档的簇数足够
    cp = os.path.join(root, "experiments", "results", "cluster_ci.json")
    check(os.path.exists(cp), "cluster_ci.json 存在（聚类口径的置信区间）")
    if os.path.exists(cp):
        cj = _json.load(open(cp, encoding="utf-8"))
        t = cj.get("seg3d_teacher_dice.csv", {})
        check(t.get("n_cases", 0) >= 10 and t.get("clustered_ci_reliable") is True,
              f"教师档簇数足够、聚类区间可用（n_cases={t.get('n_cases')}）")


def test_model_checksums():
    """models/CHECKSUMS.sha256 必须与实际文件一致，且与 ARCHITECTURE 表格逐行对应。

    由来：权重身份此前只有一句「TotalSegmentator v2」，第三方无法证明自己拿到的
    119MB 权重就是产出本仓库全部 Dice 的那一份。补了 CHECKSUMS.sha256 之后，新的
    腐烂方式是「换了权重却忘了改摘要」或「文档表格里的缩写摘要与清单对不上」。

    第一版断言只检查「文档里每个缩写摘要都能在清单里找到」，**交换表格里两个单元格
    仍然会通过**——这正是审查指出的漏洞。现在改为按表格行绑定：organs 那一行的
    摘要必须是 organs 的，recon_dl 那一行的必须是 recon_dl 的。

    数据无关：两个 .onnx 图已入库；.data 权重与 .pt checkpoint 不入库，缺席时只跳过
    其文件哈希，清单与文档的对应关系仍然照查（CHECKSUMS.sha256 本身是入库的）。
    """
    import hashlib
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(root, "models", "CHECKSUMS.sha256")
    check(os.path.exists(manifest), "models/CHECKSUMS.sha256 存在")

    entries = {}
    for line in open(manifest, encoding="utf-8").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([0-9a-f]{64})\s+(\S+)$", line)
        check(m is not None, f"清单行格式合法：{line[:60]}")
        if m:
            check(m.group(2) not in entries, f"{m.group(2)} 在清单中只出现一次")
            entries[m.group(2)] = m.group(1)

    expected = {
        "models/organs.onnx", "models/organs.onnx.data",
        "models/recon_dl_v20.onnx", "models/recon_dl_v20.onnx.data",
        "experiments/results/recon_dl_w20.pt",
        "experiments/results/seg3d_w4d3.pt", "experiments/results/seg3d_w4d3_ckpt.pt",
        "experiments/results/seg3d_w8.pt", "experiments/results/seg3d_w8d3.pt",
        "experiments/results/seg3d_w8d3_ckpt.pt",
    }
    check(set(entries) == expected,
          f"清单恰好覆盖十个模型产物（多/缺：{set(entries) ^ expected or '无'}）")
    check(len(set(entries.values())) == len(entries), "十个摘要互不相同（否则行绑定无意义）")

    hashed = 0
    for rel, digest in sorted(entries.items()):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        check(h.hexdigest() == digest, f"{rel} 实际 SHA-256 与清单一致")
        hashed += 1
    check(hashed >= 2, f"至少校验了两个入库的 .onnx 图（实际 {hashed}）")

    # —— 逐行绑定 ARCHITECTURE 的摘要表格 ——
    # 表格形如：| SHA-256 (graph, committed) | `<abbr>` | `<abbr>` |
    # 左列是 organs.onnx，右列是 recon_dl_v20.onnx；两行分别对应 .onnx 与 .onnx.data。
    arch = open(os.path.join(root, "docs", "ARCHITECTURE.md"), encoding="utf-8").read()

    def abbrev_row(label):
        """取出该行的两个缩写摘要，按 (organs 列, recon_dl 列) 返回。"""
        m = re.search(r"^\|\s*" + re.escape(label) + r"[^|]*\|([^|]*)\|([^|]*)\|",
                      arch, re.M)
        if not m:
            return None
        cells = []
        for cell in (m.group(1), m.group(2)):
            a = re.search(r"`([0-9a-f]{8})[…\.]{1,3}([0-9a-f]{6})`", cell)
            cells.append(a.groups() if a else None)
        return cells

    bindings = [
        ("SHA-256 (graph, committed)",
         ["models/organs.onnx", "models/recon_dl_v20.onnx"]),
        ("SHA-256 (`.data`, **not** committed)",
         ["models/organs.onnx.data", "models/recon_dl_v20.onnx.data"]),
    ]
    for label, files in bindings:
        cells = abbrev_row(label)
        check(cells is not None and all(cells),
              f"ARCHITECTURE 表格「{label[:22]}」行解析出两个缩写摘要（None 说明本测试定位写错）")
        if not cells or not all(cells):
            continue
        for (head, tail), rel in zip(cells, files, strict=True):
            want = entries[rel]
            check(want.startswith(head) and want.endswith(tail),
                  f"表格中 {rel} 对应的摘要 {head}…{tail} 与清单一致（交换单元格会在此失败）")


def test_performance_artifact_contract():
    """Performance run 必须留下机器、配置、耗时、峰值内存和完整模型 hash。"""
    print("[Performance provenance artifact]")
    import copy
    import hashlib
    import json
    import tempfile

    from experiments.performance_artifact import (
        build_performance_artifact,
        validate_performance_artifact,
        write_performance_artifact,
    )

    with tempfile.TemporaryDirectory() as td:
        graph = os.path.join(td, "fake.onnx")
        weights = graph + ".data"
        with open(graph, "wb") as f:
            f.write(b"graph")
        with open(weights, "wb") as f:
            f.write(b"external weights")

        artifact = build_performance_artifact(
            script="experiments/fake_bench.py",
            mode="contract-test",
            project_root=_ROOT,
            model_files=[graph, weights],
            configuration={"provider": "CPUExecutionProvider", "n_rep": 3},
            wall_time_seconds=12.5,
            peak_memory_gib=1.25,
            peak_memory_method="test fixture",
        )
        check(validate_performance_artifact(artifact) is artifact,
              "完整 performance artifact 通过 schema 校验")
        check(set(artifact) >= {"machine", "configuration", "measurements", "model"},
              "artifact 含 machine / configuration / measurements / model")
        check(artifact["measurements"]["wall_time_seconds"] == 12.5
              and artifact["measurements"]["peak_memory_gib"] == 1.25,
              "wall time 与 peak memory 使用机器可读数值")
        hashes = {x["name"]: x["sha256"] for x in artifact["model"]["files"]}
        check(hashes == {
            "fake.onnx": hashlib.sha256(b"graph").hexdigest(),
            "fake.onnx.data": hashlib.sha256(b"external weights").hexdigest(),
        }, "ONNX graph 与 external data 分别绑定 SHA-256")
        check(len(artifact["model"]["combined_sha256"]) == 64,
              "模型组合摘要存在（不会把 graph hash 冒充完整权重身份）")

        out = os.path.join(td, "run.json")
        write_performance_artifact(out, artifact)
        loaded = json.load(open(out, encoding="utf-8"))
        check(loaded == artifact, "artifact 原子写入后可无损读取")

        bad = copy.deepcopy(artifact)
        del bad["machine"]
        try:
            validate_performance_artifact(bad)
            rejected = False
        except ValueError:
            rejected = True
        check(rejected, "已知坏 artifact（缺 machine）会被拒绝")


def test_doc_code_consistency():
    """文档不得声称与代码相反的事实，也不得写下未被实验支持的等价性判断。

    这条测试的由来：曾经在同一个提交里，README 写着 recon_dl.py「never calls
    torch.manual_seed」，而那一行正是该提交加进去的——先写描述旧状态的文档、再改
    代码、然后没有回头核对。通读发现不了这种矛盾，因为两份文件不会被同时读到。
    锁成断言之后，改一边忘另一边就会当场失败。

    只查能机械判定的部分：关键词与代码事实的对应关系。它不能证明语义一致，也不
    声称能——语义仍要人读。纯文本比对，不依赖 Qt 与真实数据，进 SKIP_REAL_DATA 子集。
    """
    print("[文档与代码一致性 doc/code consistency]")
    import re

    def rd(rel):
        with open(os.path.join(_ROOT, rel), encoding='utf-8') as f:
            return f.read()

    import ast

    def calls_manual_seed(src):
        """AST 判定是否真的调用了 torch.manual_seed(...)。

        不用字符串搜索：注释、docstring、乃至本测试自己引用这个名字的地方都含有
        同样的字面量，一旦有人把调用删掉只留注释，字符串搜索会继续报「有」。
        """
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == 'manual_seed'
                        and isinstance(f.value, ast.Name) and f.value.id == 'torch'):
                    return True
        return False

    recon_dl = rd('experiments/recon_dl.py')
    seg3d_train = rd('experiments/seg3d_train.py')
    docs = {n: rd(n) for n in ('README.md', 'README.zh-CN.md', 'experiments/README.md',
                               'experiments/recon_dl.py', 'docs/technical_report.md',
                               'CHANGELOG.md')}

    # ① 文档关于 torch.manual_seed 的说法必须与代码事实一致——【两个方向都要拦】。
    #    初版只拦了「代码有、文档说没有」，而注释里却写着「反之亦然」，于是这个
    #    专门用来防「注释说了代码没做的事」的测试，自己就是那样一处。
    dl_seeded = calls_manual_seed(recon_dl)
    denies = [n for n, t in docs.items()
              if re.search(r'never calls?\s+`?torch\.manual_seed|从未调用\s*`?torch\.manual_seed', t)]
    # 【覆盖范围有限，如实说明】这里匹配的是【目前文档里实际用过的】几种措辞，
    # 不是「任何声称已加入种子的说法」。换一种写法就可能漏——所以本测试只声称
    # 「覆盖已知措辞的双向检查」，不声称锁死了全部文档表述。
    affirms = [n for n, t in docs.items()
               if re.search(r'gained\s+`?torch\.manual_seed|加入\s*`?torch\.manual_seed'
                            r'|now takes `seed=0` and pins|RNG was pinned|pinned only after'
                            r'|RNG 是在.*才固定', t)]   # 不含 AST-detected：
    # 那是在介绍检查机制，不是在声称调用当前存在——把它算作肯定陈述，会让
    # 「将来合法删除 seed 并同步了所有事实文档」的情形反被这条断言判为违规。
    check(not (dl_seeded and denies),
          f"recon_dl 有 manual_seed 时无文档声称它没有（违规: {denies or '无'}）")
    # 只在断言真被触发时才列文件名：前件为假时列出来会让一条 PASS 看着像有问题。
    _bad = affirms if not dl_seeded else []
    check(not _bad, "recon_dl 无 manual_seed 时无文档声称已加入"
                    + (f"（违规: {_bad}）" if _bad else "（当前代码确有调用，此向不适用）"))

    # ①b 变异自检：把真实调用注释掉之后，AST 判定必须翻转为 False，而字符串搜索
    #     仍会报 True。没有这一条，「AST 比字符串搜索强」就只是注释里的一句声称。
    mutated = recon_dl.replace('    torch.manual_seed(seed)\n',
                               '    # torch.manual_seed(seed)\n')
    check(mutated != recon_dl and not calls_manual_seed(mutated)
          and 'torch.manual_seed' in mutated,
          "变异自检：注释掉调用后 AST 翻转为 False，而字符串搜索仍为 True")

    # ② 加种子后从未重跑比对过，因此不得出现「统计上等价」这类判断。
    #    否定用法也一并禁止——与其给检查器开例外，不如换一种说法。
    equiv = [n for n, t in docs.items()
             if 'statistically equivalent' in t or '统计上相当' in t]
    check(not equiv, f"无未经重跑支持的等价性措辞（违规: {equiv or '无'}）")

    # ③ seed-fixed 只能用于训练侧 RNG 确实固定、且产物也产自固定之后的那条线。
    #    研究四(seg3d_train)满足；研究三(recon_dl)的已提交产物早于补种子，不满足。
    check(calls_manual_seed(seg3d_train), "seg3d_train 确有 manual_seed 调用（研究四标 seed-fixed 才成立）")
    bad = [n for n in ('README.md', 'README.zh-CN.md')
           for ln in docs[n].split('\n')
           if 'recon_dl' in ln and ('seed-fixed' in ln or '种子固定' in ln)]
    check(not bad, f"能力表未把研究三标成 seed-fixed（违规 {len(bad)} 行）")

    # ④ packaging inventory、root 产品模块、实际 local imports 与公开架构清单同源核对。
    #    不锁 LOC：行数随正常维护频繁变化，不能让每加一行都迫使公开文档改动。
    pyproject = rd('pyproject.toml')
    module_block = re.search(r'py-modules\s*=\s*\[(.*?)\]', pyproject, re.S)
    declared = set(re.findall(r'"([a-z_0-9]+)"', module_block.group(1))) if module_block else set()
    actual = {os.path.splitext(name)[0] for name in os.listdir(_ROOT)
              if re.fullmatch(r'[a-z_0-9]+\.py', name)}

    local_imports = set()
    qt_importers = set()
    for module in actual:
        tree = ast.parse(rd(f'{module}.py'))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split('.')[0])
        local_imports.update(roots & actual)
        if 'PySide6' in roots:
            qt_importers.add(module)

    # constants.py 无 Qt，但属于常量表，不计入“compute modules”。
    qt_free_compute = actual - qt_importers - {'constants'}
    arch = rd('docs/ARCHITECTURE.md')
    block = arch.split('—— Qt-free compute modules')[1]
    block = block.split('\n', 1)[1].split('——')[0]   # 跳过分隔行自身（它首尾都有 ——）
    listed = re.findall(r'^([a-z_0-9]+)\.py\s{2,}', block, re.M)
    listed = [m for m in listed if m != 'constants']          # 常量表不算计算模块
    check(len(listed) >= 5, f"模块清单解析出 {len(listed)} 条（<5 说明是本测试的定位写错了）")
    words = {'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11}
    claimed = next((v for w, v in words.items() if f'{w} Qt-free' in arch), None)

    def inventory_errors(declared_modules, listed_modules, claimed_count):
        errors = []
        if declared_modules != actual:
            errors.append(f"pyproject delta={sorted(declared_modules ^ actual)}")
        missing_imports = local_imports - declared_modules
        if missing_imports:
            errors.append(f"local imports absent from wheel={sorted(missing_imports)}")
        if set(listed_modules) != qt_free_compute:
            errors.append(f"ARCHITECTURE delta={sorted(set(listed_modules) ^ qt_free_compute)}")
        if claimed_count != len(qt_free_compute):
            errors.append(f"claimed={claimed_count}, actual Qt-free compute={len(qt_free_compute)}")
        return errors

    inventory_bad = inventory_errors(declared, listed, claimed)
    check(not inventory_bad,
          f"pyproject/root imports/ARCHITECTURE inventory 一致（问题: {inventory_bad or '无'}）")
    check(len(declared) == 19 and len(qt_free_compute) == 10,
          f"当前 candidate inventory：{len(declared)} top-level modules / "
          f"{len(qt_free_compute)} Qt-free compute modules")

    # known-bad 自检只改 synthetic set；不触碰 tracked pyproject/Markdown。
    mutated_errors = inventory_errors(declared - {'dicom_geometry'}, listed, claimed)
    check(any('dicom_geometry' in error for error in mutated_errors),
          "known-bad inventory：synthetic 移除 dicom_geometry 会被 checker 拒绝")


def test_model_card():
    """模型说明卡：数字必须来自实验产物、局限段必须在场、双语都不夹带对方语言。

    这张卡片的意义在于主动暴露适用边界，所以「局限段存在」本身就是被测契约：
    若日后有人把它精简掉，此处即失败。不依赖 Qt 与真实数据，进 SKIP_REAL_DATA 子集。
    """
    print("[模型说明卡 model_card]")
    import json
    import re

    import model_card
    for en in (False, True):
        txt = model_card.build_model_card(en)
        cjk = bool(re.search(r'[一-鿿]', txt))
        check(cjk != en, f"{'英文' if en else '中文'}卡片语言正确（含中文={cjk}）")
        check(('Known limitations' if en else '已知局限') in txt, "  局限段在场")
        check(('spacing' in txt.lower()), "  点名 spacing 未重采样这一硬伤")
        # 样本量必须写明，但不锁死具体数字——多器官验证已从 n=1 扩到 20 例，
        # 早期那条「必须出现 n=1」的断言会在扩样本后反过来阻止如实更新。
        has_n = ('n=1' in txt) or re.search(r'\b\d+ (cases|例)', txt)
        check(bool(has_n), "  如实标注多器官验证的样本量")
        check(('diagnosis' in txt.lower() or '诊断' in txt), "  非诊断用途声明在场")
        # 卡片走 QLabel 的 RichText 渲染，只认 HTML。写成 Markdown 的 **粗体** 会原样
        # 印在界面上（截图时发现的），故此处直接禁掉星号强调。
        check('**' not in txt, "  没有 Markdown 星号混进 HTML（QLabel 会原样显示）")

    # 数字必须与产物一致，不得在卡片里硬编码
    tp = os.path.join(_ROOT, "experiments", "results", "seg3d_teacher_summary.json")
    if os.path.exists(tp):
        m = json.load(open(tp, encoding='utf-8'))['overall_mean']
        card = model_card.build_model_card(False)
        check(f"{m:.3f}" in card, f"肺叶 Dice 取自 seg3d_teacher_summary.json（{m:.3f}）")
        check(str(json.load(open(tp, encoding='utf-8'))['n_cases']) in card, "例数取自同一产物")

    # spacing 那段初版把 0.922→0.881→0.799 写死在字符串里，与本模块「不硬编码」
    # 的原则自相矛盾——重跑消融换了数值，卡片会安静地继续显示旧数字。
    # 断言方向是「卡片里的每个数字都必须能在产物里找到」，而不是「产物里每个数字都要
    # 出现在卡片上」——后者会把文案精炼当成失败，而真正的风险是卡片编造或残留旧数字。
    import csv as _csv
    RES = os.path.join(_ROOT, "experiments", "results")
    allowed = set()

    def _collect(vals):
        for v in vals:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            for fmt in ('%.2f', '%.3f', '%.4f'):
                allowed.add(fmt % fv)
            allowed.add(str(v))

    for name, keys in (("seg_spacing.csv", ('mean_dice',)),
                       ("seg_spacing_fix.csv", ('dice_direct', 'dice_engine')),
                       ("seg_dice.csv", ('dice',))):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            with open(p, encoding='utf-8-sig') as f:
                rows = list(_csv.DictReader(f))
            for k in keys:
                _collect(r.get(k) for r in rows)
            if name == "seg_dice.csv":      # 卡片报的是均值，不是逐器官值
                ds = [float(r['dice']) for r in rows if float(r['dice']) > 0]
                if ds:
                    _collect([sum(ds) / len(ds)])
    # 多例配对：卡片报的是**聚合值**（两侧均值与差值均值），CSV 里没有现成字段，
    # 故此处按同一定义重算——断言的是「卡片的数字可由产物推出」，而不是「照抄某一列」。
    pm = os.path.join(RES, "seg_spacing_fix_multi.csv")
    if os.path.exists(pm):
        with open(pm, encoding='utf-8-sig') as f:
            mr = [r for r in _csv.DictReader(f)
                  if (r.get('case') or '').strip() and not r['case'].strip().startswith('#')]
        if len(mr) >= 2:
            dr = [float(r['dice_direct']) for r in mr]
            en2 = [float(r['dice_engine']) for r in mr]
            _collect([sum(dr) / len(dr), sum(en2) / len(en2),
                      sum(en2) / len(en2) - sum(dr) / len(dr)])

    # 多例 21 器官：患者级均值可独立重算并断言相等（防硬编码）；CI 是 bootstrap 结果，
    # 无法用简单算术复现，故取模块算出的值——它本身完全来自 CSV，仍守住「不硬编码」。
    pmo = os.path.join(RES, "seg_multi.csv")
    if os.path.exists(pmo):
        with open(pmo, encoding='utf-8-sig') as f:
            cv = [float(r['mean_dice']) for r in _csv.DictReader(f)
                  if (r.get('case') or '').strip() and not r['case'].strip().startswith('#')]
        got = model_card._read_multi_organ()
        if cv and got:
            # 卡片报的是产物汇总行里的值（4 位精度），不是自己重算的全精度值——
            # 这正是「单一数据源」的意图：同一个统计量若有两套实现就会漂移
            # （初版卡片自算 bootstrap，CI 下界与实验脚本差了 0.001）。
            # 故断言两件事：汇总行本身算得对，且卡片确实用的是它。
            check(abs(got[1] - sum(cv) / len(cv)) < 5e-5,
                  f"汇总行的患者级均值与逐例重算一致（{got[1]:.4f}）")
            import math
            check(not math.isnan(got[2]) and not math.isnan(got[3]),
                  f"CI 读自产物而非现算（[{got[2]:.4f}, {got[3]:.4f}]）")
            _collect([got[1], got[2], got[3]])
            _collect(float(w[1]) for w in got[4])          # 最弱器官的 Dice
    po = os.path.join(RES, "seg_multi_per_organ.csv")
    if os.path.exists(po):
        with open(po, encoding='utf-8-sig') as f:
            for r in _csv.DictReader(f):
                _collect([r['mean_dice'], r['ci_lo'], r['ci_hi']])

    tp2 = os.path.join(RES, "seg3d_teacher_summary.json")
    if os.path.exists(tp2):
        d2 = json.load(open(tp2, encoding='utf-8'))
        _collect([d2['overall_mean'], *d2['overall_ci']])
    if allowed:
        card = model_card.build_model_card(False)
        bogus = [n for n in set(re.findall(r'\b0\.\d{3,4}\b', card)) if n not in allowed]
        check(not bogus, f"卡片上的每个 Dice 数字都能在实验产物中找到（可疑 {bogus}）")
    check(len(model_card.card_title(False)) > 0 and len(model_card.card_title(True)) > 0,
          "标题双语均非空")


def test_label_palette():
    """24 类器官调色板：覆盖完整、语义与实测真相表一致、任意两类肉眼可分。

    钉住三件曾经出错的事：
      1. 只配了 16 类，其余 7 类（含 Dice 0.985 的右肾）一起渲染成同一种暗灰；
      2. 16 条注释里 15 条与实测确证的标签语义不符（如把 5 注为「心脏」，实为肝）；
      3. 肺叶左右配反——10,11 是【左】肺却按「右肺」上暖色。
    不依赖真实数据与权重，可进 SKIP_REAL_DATA 子集。
    """
    print("[器官调色板：覆盖 / 语义 / 可分辨性]")
    import json
    import re

    import constants as C
    lp = os.path.join(_ROOT, "models", "organ_labels_candidate.json")
    truth = json.load(open(lp, encoding='utf-8'))['labels']

    missing = [i for i in range(1, 25) if i not in C.LABEL_COLORS]
    check(not missing, f"1-24 类全部配色（缺 {missing}）")

    grey = [i for i in range(1, 25) if tuple(C.LABEL_LUT[i][:3]) == (96, 96, 96)]
    check(not grey, f"LUT 中无类别落到未配色暗灰（{grey}）")
    check(C.LABEL_LUT[0].tolist() == [0, 0, 0, 0], "背景 0 在 LUT 中全透明")

    # 注释名必须与实测确证的标签表逐条一致——注释是读代码的人唯一的语义来源
    src = open(os.path.join(_ROOT, "constants.py"), encoding='utf-8').read()
    blk = src[src.index('LABEL_COLORS = {'):src.index('_UNKNOWN_COLOR')]
    bad = []
    for mm in re.finditer(r'^\s*(\d+):\s*\([\d, ]+\),\s*#\s*(\S+?)(?:（|\(|\s|$)', blk, re.M):
        i, name = int(mm.group(1)), mm.group(2)
        if name != truth.get(str(i), {}).get('name_zh'):
            bad.append((i, name, truth.get(str(i), {}).get('name_zh')))
    check(not bad, f"注释名与实测标签表逐条一致（不符 {bad}）")

    # 侧别：10,11=左肺走冷色，12,13,14=右肺走暖色（实测确证的侧别，曾判反）
    cold = all(C.LABEL_COLORS[i][2] > C.LABEL_COLORS[i][0] for i in (10, 11))
    warm = all(C.LABEL_COLORS[i][0] > C.LABEL_COLORS[i][2] for i in (12, 13, 14))
    check(cold and warm, f"左肺冷色={cold}、右肺暖色={warm}（侧别经 seg_validate 确证）")

    ks = list(C.LABEL_COLORS)
    pairs = sorted((sum((a - b) ** 2 for a, b in zip(C.LABEL_COLORS[p], C.LABEL_COLORS[q], strict=True)) ** .5, p, q)
                   for x, p in enumerate(ks) for q in ks[x + 1:])
    dmin, p1, p2 = pairs[0]
    check(dmin > 40, f"任意两类 RGB 欧氏距离 > 40（最近 {p1}↔{p2} = {dmin:.0f}）")
    # LABEL_COLORS 里登记了颜色、LUT 里却没填的标签，画出来是全透明——功能静默
    # 失效且界面无提示。构建 LUT 的循环只覆盖 1-24，区间外的特殊标签需逐个登记，
    # LUNG_FALLBACK_LABEL 由 10 改为 254 时正是这样漏过一次。
    missing = [k for k in C.LABEL_COLORS if C.LABEL_LUT[k][3] == 0]
    check(not missing, f"LABEL_COLORS 的每个标签在 LUT 中都有非零 alpha（缺 {missing}）")
    check(len(set(C.LABEL_COLORS.values())) == len(C.LABEL_COLORS), "无两类共用同一 RGB")


def test_mask_nondestructive(app):
    """蒙版破坏性操作的三道防线：3D 追踪不吃掉 AI 器官、清空需确认、两者皆可撤销。

    实测动机：磁盘缓存里的 mask 曾 100% 是手动追踪标签、24 类器官一个不剩——
    旧实现的 3D 追踪对 volume_mask 整卷赋值，一次追踪就抹掉 ~100s 的推理结果，
    且经 save_project 落盘后不可恢复。合成小体积，不触发任何 AI 推理。
    """
    print("[蒙版破坏性操作：非破坏追踪 / 确认 / 撤销]")
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QMessageBox as _QMB

    from constants import MANUAL_TRACK_LABEL
    saved_q = _QMB.question
    box = {'n': 0, 'ans': _QMB.Yes}

    def _fake_q(*a, **k):
        box['n'] += 1
        return box['ans']

    _QMB.question = staticmethod(_fake_q)
    vi = None
    try:
        vi = m.MedicalViewer(); app.processEvents()
        if vi.ai_thread: vi.ai_thread.cancel()
        Z, H, W = 10, 48, 48
        vol = np.random.RandomState(0).uniform(-1000, -900, (Z, H, W)).astype(np.float32)
        vol[3:7, 12:28, 12:28] = 60.0        # 一团均质组织，供区域增长追出连通域
        vi.volume_hu = vol
        vi.dicom_datasets = [None] * Z
        _mark_supported_capabilities(vi)
        vi.volume_mask = np.zeros((Z, H, W), np.uint8)
        vi.volume_mask[2:5, 30:40, 30:40] = 5     # 肝
        vi.volume_mask[5:8, 30:40, 5:15] = 2      # 右肾
        vi.current_3d_pos = [4, H // 2, W // 2]
        vi._mask_undo.clear()
        organs = lambda mk: int(((mk > 0) & (mk != MANUAL_TRACK_LABEL)).sum())  # noqa: E731
        n0 = organs(vi.volume_mask)

        vi.handle_3d_track_requested(1, QRectF(14, 14, 12, 12)); app.processEvents()
        n_tr = int((vi.volume_mask == MANUAL_TRACK_LABEL).sum())
        check(organs(vi.volume_mask) == n0 and n0 > 0,
              f"3D 追踪后 AI 器官体素不变（{n0} → {organs(vi.volume_mask)}）")
        check(n_tr > 0, f"追踪层确实写入（{n_tr} 体素）")

        vi.handle_3d_track_requested(1, QRectF(14, 14, 12, 12)); app.processEvents()
        check(int((vi.volume_mask == MANUAL_TRACK_LABEL).sum()) == n_tr, "重复追踪不累积追踪层")
        check(sum(1 for e in vi._mask_undo if e[0] == vi._VOL_UNDO) == 1,
              "撤销栈中整卷快照只留最近一份（整卷约 61MB，不可堆 20 份）")

        snap = vi.volume_mask.copy()
        box['n'] = 0; box['ans'] = _QMB.No
        vi.clear_mask_and_annotations(); app.processEvents()
        check(box['n'] == 1, "清空蒙版前弹出确认框")
        check(np.array_equal(vi.volume_mask, snap), "确认框选 No：蒙版一个体素未动")

        box['ans'] = _QMB.Yes
        vi.clear_mask_and_annotations(); app.processEvents()
        check(not vi.volume_mask.any(), "确认框选 Yes：蒙版清零")
        vi._undo_mask_edit(); app.processEvents()
        check(np.array_equal(vi.volume_mask, snap), "Ctrl+Z 把整卷蒙版还原回来")

        box['n'] = 0
        vi.volume_mask[:] = 0; vi.global_annotations.clear()
        vi.clear_mask_and_annotations()
        check(box['n'] == 0, "无可清内容时不弹框骚扰")
    finally:
        _QMB.question = saved_q
        if vi is not None:
            if vi.ai_thread: vi.ai_thread.cancel()
            vi.close()
        app.processEvents()


def test_mask_cache_roundtrip(app):
    """mask/annotation 只在 geometry fingerprint 一致时恢复，所有 I/O 位于临时目录。"""
    print("[mask/annotation cache fingerprint save→reload]")
    import glob
    import shutil
    import tempfile

    from pydicom.uid import generate_uid
    ed = tempfile.mkdtemp()
    pid = "RID_CACHE_TEST"
    made = []

    def _mkdir_series(uid, z=3):
        d = tempfile.mkdtemp()
        for i in range(z):
            _write_min_dcm(os.path.join(d, f"s{i}.dcm"), (16, 16), uid, ipp_z=i, inst=i + 1, pid=pid)
        return d

    uid_a, uid_b = generate_uid(), generate_uid()
    da, db, dc = _mkdir_series(uid_a), _mkdir_series(uid_b), _mkdir_series(uid_a)
    try:
        vc = m.MedicalViewer(); app.processEvents()
        vc.persistence_dir = ed
        vc._kickoff_ai = lambda: None
        if vc.ai_thread:
            vc.ai_thread.cancel()
        # 序列 A：造一个非空蒙版并保存
        vc.load_data(da); app.processEvents()
        if vc.ai_thread:
            vc.ai_thread.cancel()
        vc.volume_mask = np.zeros_like(vc.volume_hu, dtype=np.uint8)
        vc.volume_mask[0, :4, :4] = 5          # 标记为器官5，便于区分
        vc.global_annotations = {0: [{'id': 'a0', 'type': 'ruler',
                                      'p1': [1, 1], 'p2': [4, 4]}], 'all': []}
        vc.save_project()
        made = glob.glob(os.path.join(ed, f"{pid}_*"))
        check(any(f.endswith("_mask.npz") for f in made)
              and any(f.endswith("_annotations.json") for f in made),
              "save_project 仅在临时 persistence_dir 落盘 mask + annotation")

        # 重开序列 A（同 UID 同 shape）→ 应恢复
        vc.volume_mask = None
        restored_a = vc._load_saved_mask(pid)
        check(restored_a and vc.volume_mask is not None and int(vc.volume_mask[0, 0, 0]) == 5,
              "同 UID/shape/fingerprint → mask 恢复")
        vc.global_annotations = {'all': []}; vc._load_annotations_json(pid)
        check(vc.global_annotations.get(0, [{}])[0].get('id') == 'a0',
              "同 UID/shape/fingerprint → slice-indexed annotation 恢复")

        # 同 UID/shape，但 SOP→slice identity 不同：必须由 fingerprint 拒绝。
        vc.load_data(dc); app.processEvents()
        vc.volume_mask = None; vc.global_annotations = {'all': []}
        check(not vc._load_saved_mask(pid), "同 UID/shape、不同 ordered SOP fingerprint → mask 拒绝")
        vc._load_annotations_json(pid)
        check(vc.global_annotations == {'all': []},
              "同 UID/shape、不同 ordered SOP fingerprint → annotation 拒绝")

        # 切到序列 B（同 PatientID、同 shape、不同 SeriesInstanceUID）→ 必须拒绝
        vc.load_data(db); app.processEvents()
        if vc.ai_thread:
            vc.ai_thread.cancel()
        vc.volume_mask = None
        restored_b = vc._load_saved_mask(pid)
        check(not restored_b and vc.volume_mask is None,
              "切到同患者另一序列（同 shape 不同 UID）→ 拒绝套用旧蒙版（核心回归）")

        # legacy 无 fingerprint：mask 与 annotation 都默认拒绝，不猜测旧顺序。
        np.savez_compressed(os.path.join(ed, f"{pid}_mask.npz"),
                            mask=np.ones((3, 16, 16), np.uint8), series_uid=np.array(uid_b))
        with open(os.path.join(ed, f"{pid}_annotations.json"), 'w', encoding='utf-8') as f:
            import json as _json
            _json.dump({'__meta__': {'series_uid': uid_b},
                        '0': [{'id': 'legacy', 'type': 'ruler', 'p1': [1, 1], 'p2': [2, 2]}]}, f)
        vc.volume_mask = None; vc.global_annotations = {'all': []}
        check(not vc._load_saved_mask(pid), "legacy mask 缺 fingerprint → 默认拒绝")
        vc._load_annotations_json(pid)
        check(vc.global_annotations == {'all': []}, "legacy annotation 缺 fingerprint → 默认拒绝")
    finally:
        shutil.rmtree(ed, ignore_errors=True)
        shutil.rmtree(da, ignore_errors=True); shutil.rmtree(db, ignore_errors=True)
        shutil.rmtree(dc, ignore_errors=True)


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
    test_runner_catches_qt_slot_exceptions()
    # 有真实数据（本地开发）跑全套；无数据或 CI（SKIP_REAL_DATA=1）只跑数据无关的自包含测试。
    has_data = (os.path.isdir(os.path.join(_ROOT, "肺癌"))
                and not os.environ.get("SKIP_REAL_DATA"))
    if not has_data:
        print("WARN: 无 ../肺癌 真实数据（或 SKIP_REAL_DATA=1），仅运行数据无关的自包含测试")
        # 这些测试自建合成 DICOM / 用 /nonexistent.onnx 走数学降级，不依赖真实数据或 119MB 权重
        for t in (test_ai_engine, test_noncanonical_dicom_gating,
                  test_unsupported_dicom_contract, test_missing_series_uid_contract,
                  test_load_clears_stale_hu_probe,
                  test_invalid_calibration_raw_gating, test_hu_unit_semantics_gating,
                  test_spacing_capability_gating,
                  test_compare_dicom_contract, test_deid_export_and_persistence_contract,
                  test_save_project_atomic_contract,
                  test_mixed_shape_dicom, test_recon_finite,
                  test_close_cancels_ai, test_malformed_pixels, test_empty_dicom_tags,
                  test_export_path_safety, test_dicom_sort_consistency, test_i18n_persistent,
                  test_nonfinite_dicom_tags, test_dialog_i18n_coverage,
                  test_hu_conversion, test_mask_cache_roundtrip, test_mouse_interaction,
                  test_mesh_view, test_ai_failure_visible, test_mask_nondestructive,
                  test_phantom_recon_flow, test_probe_hu, test_wheel_and_cine,
                  test_matrix_recon_ui, test_crop_and_legend, test_compare_entry, test_panel_scroll,
                  test_mpr_linkage):
            t(app)
        test_spacing_resample()  # 假 session，不加载权重
        test_patient_space_geometry_contract()
        test_phantom()        # 纯解析生成，无需 app
        test_confidence_map() # 手工 logits，不加载模型
        test_model_card()     # 纯字符串组装，无需 app / 真实数据
        test_model_card_fallback()  # 产物缺失/损坏时的回退，无需 app
        test_label_palette()  # 纯数据校验，无需 app / 真实数据
        test_quantify()      # 纯函数单测，无需 app / 真实数据
        test_quantify_high_label()
        test_lung_fallback()
        test_followup()
        test_registration()
        test_projection()
        test_mesh3d()
        test_undo_restores_confidence()
        test_model_card_bad_fields()
        test_stale_ai_cannot_overwrite_restored_mask()
        test_anisotropic_pixel_spacing(app)
        test_mpr_geometry()
        test_dicom_landmark_orientation(app)
        test_mask_cache_guard()
        test_geometry_fingerprint_contract()
        test_recon_numerics()          # 重建数值正确性：解析模体，无 Qt / 真实数据
        test_asdpocs_numerics()        # ASD-POCS/TV 正则化：合成小系统，无 Qt / 真实数据
        test_dl_recon_guard()
        test_recon_pipeline_helpers()
        test_sampling_density()       # 纯 recon.make_theta，无 Qt / 真实数据
        test_zero_grade_guards(app, m)  # 归零级守卫：合成数据，无真实数据依赖
        test_withdrawn_claims_stay_withdrawn()  # 撤回结论防回潮：纯文本 + 已入库 CSV
        test_model_checksums()        # 权重摘要清单与文档一致：纯文本 + 已入库 .onnx
        test_performance_artifact_contract()  # 性能产物 provenance 合约：纯 stdlib + 临时文件
        test_doc_code_consistency()   # 文档与代码一致性：纯文本，无 Qt / 真实数据
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
        test_sampling_density()
        test_compare(v, app)
        test_cine_keyboard(v, app)
        test_compliance(v, app)
        test_edge_cases(v, app)
        test_noncanonical_dicom_gating(app)
        test_unsupported_dicom_contract(app)
        test_missing_series_uid_contract(app)
        test_load_clears_stale_hu_probe(app)
        test_invalid_calibration_raw_gating(app)
        test_hu_unit_semantics_gating(app)
        test_spacing_capability_gating(app)
        test_compare_dicom_contract(app)
        test_deid_export_and_persistence_contract(app)
        test_save_project_atomic_contract(app)
        test_mixed_shape_dicom(app)
        test_legend_consistency(v, app)
        test_recon_finite(app)
        test_malformed_annotations(v, app)
        test_close_cancels_ai(app)
        test_ai_failure_visible(app)
        test_malformed_pixels(app)
        test_empty_dicom_tags(app)
        test_nonfinite_dicom_tags(app)
        test_export_path_safety(app)
        test_dicom_sort_consistency(app)
        test_dialog_i18n_coverage(app)
        test_i18n_persistent(app)
        test_projection_ui(v, app)
        test_dl_recon_ui(v, app)
        test_mesh3d_ui(v, app)
        test_mesh_view(app)
        test_hu_conversion(app)
        test_mask_cache_roundtrip(app)
        test_mask_nondestructive(app)
        test_mouse_interaction(app)
        test_label_palette()
        test_model_card()
        test_model_card_fallback()
        test_confidence_map()
        test_probe_hu(app)
        test_wheel_and_cine(app)
        test_matrix_recon_ui(app)
        test_crop_and_legend(app)
        test_compare_entry(app)
        test_panel_scroll(app)
        test_mpr_linkage(app)
        test_spacing_resample()
        test_patient_space_geometry_contract()
        test_phantom()
        test_phantom_recon_flow(app)
        test_quantify()
        test_quantify_high_label()
        test_lung_fallback()
        test_followup()
        test_registration()
        test_projection()
        test_mesh3d()
        test_undo_restores_confidence()
        test_model_card_bad_fields()
        test_stale_ai_cannot_overwrite_restored_mask()
        test_anisotropic_pixel_spacing(app)
        test_mpr_geometry()
        test_dicom_landmark_orientation(app)
        test_mask_cache_guard()
        test_geometry_fingerprint_contract()
        test_recon_numerics()          # 重建数值正确性：解析模体，无 Qt / 真实数据
        test_asdpocs_numerics()        # ASD-POCS/TV 正则化：合成小系统，无 Qt / 真实数据
        test_dl_recon_guard()
        test_recon_pipeline_helpers()
        test_zero_grade_guards(app, m)  # 归零级守卫：合成数据，无真实数据依赖
        test_withdrawn_claims_stay_withdrawn()  # 撤回结论防回潮：纯文本 + 已入库 CSV
        test_model_checksums()        # 权重摘要清单与文档一致：纯文本 + 已入库 .onnx
        test_performance_artifact_contract()  # 性能产物 provenance 合约：纯 stdlib + 临时文件
        test_doc_code_consistency()   # 文档与代码一致性：纯文本，无 Qt / 真实数据
    print("\n" + ("全部通过" if not _FAILS else f"{len(_FAILS)} 项失败: " + "; ".join(_FAILS)))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main_run())
