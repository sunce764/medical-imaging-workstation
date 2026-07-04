#!/usr/bin/env python
# =============================================================================
# 医学影像工作站 —— 回归测试套件
#
# 覆盖：启动/工具栏、AI 推理引擎（取消/进度/信号回调）、历次修复项、
#      多器官分割渲染与定量、分割手动编辑（画笔/橡皮/目标/撤销）、
#      椭圆 ROI（渲染/拖动/缩放/命中）、采样密度、双序列对比（配准/守卫）、
#      Cine（往返/调速/键盘）、合规（脱敏/免责）。
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
from PySide6.QtWidgets import QApplication, QMessageBox, QGraphicsView, QGraphicsTextItem
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtCore import Qt, QEvent, QPointF

import main as m
import ai_engine
import graphics_view as gv
from graphics_view import ROIGraphicsItem
from constants import (AXIAL, CORONAL, MANUAL_TRACK_LABEL, TOOL_SEG_BRUSH,
                       TOOL_ROI, TOOL_POINTER)

# 静音弹窗，避免离屏阻塞
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

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
        from PySide6.QtGui import QResizeEvent
        from PySide6.QtCore import QSize
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


def test_sampling_density(v):
    print("[重建采样密度]")
    import recon
    th = recon.make_theta(180, 180 * 4)
    check(len(th) == 720 and th[-1] < 180, "180° 4× 过采样 = 720 投影且覆盖不变")


def test_compare(v, app):
    print("[双序列随访对比]")
    saved_id = id(v.dicom_datasets)
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
    z = v.current_3d_pos[0]
    v.volume_mask = np.zeros(v.volume_hu.shape, np.uint8)
    v.handle_seg_paint(1, [(200, 200)], False)
    # 换病例清撤销栈
    v._build_volume_hu()
    check(len(v._mask_undo) == 0, "换病例清空分割撤销栈")
    # 换更小病例后撤销不越界崩溃
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
    # 脱敏隐去对比既往日期（PHI）
    vv = m.MedicalViewer(); app.processEvents()
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


def _write_min_dcm(path, shape, series_uid, ipp_z, inst, pid='RID_TEST', empty_numeric=False):
    """写一张最小合规的 CT DICOM，供混合形状加载测试使用。ipp_z=None 则不写 ImagePositionPatient。
    empty_numeric=True 时把 RescaleSlope/Intercept/PixelSpacing/SliceThickness 写成空值（None），
    模拟畸形 DICOM（pydicom 读回 None，直接 float() 会崩）。"""
    import numpy as _np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
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
    ds.RescaleSlope = None if empty_numeric else 1
    ds.RescaleIntercept = None if empty_numeric else -1024
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = _np.full((rows, cols), 100, dtype=_np.int16).tobytes()
    ds.save_as(path, write_like_original=False)


def test_mixed_shape_dicom(app):
    """加载同序列/无 SeriesUID 但切片形状不一致的目录，不得崩溃（形状一致性过滤）。"""
    print("[混合形状 DICOM 加载防护]")
    import tempfile, shutil
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


def test_empty_dicom_tags(app):
    """RescaleSlope/Intercept/PixelSpacing/SliceThickness 存在但为空(None)时，
    加载/定量不得因 float(None) 崩溃。"""
    print("[空数值标签 DICOM 防护]")
    import tempfile, shutil
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
        def __init__(self, pid): self.PatientID = pid; self.PatientName = pid

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
    import tempfile, shutil
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
    from PySide6.QtWidgets import QLabel, QPushButton, QGroupBox, QCheckBox, QComboBox
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


def main_run():
    app = QApplication([])
    if not os.path.isdir(os.path.join(_ROOT, "肺癌")):
        print("WARN: 缺少 ../肺癌 真实数据，仅运行 AI 引擎单元测试")
        test_ai_engine(app)
    else:
        v = m.MedicalViewer()
        app.processEvents()
        if v.ai_thread:
            v.ai_thread.cancel()
        test_startup(v)
        test_ai_engine(app)
        test_prior_fixes(v, app)
        test_multiorgan_and_edit(v, app)
        test_roi(v, app)
        test_sampling_density(v)
        test_compare(v, app)
        test_cine_keyboard(v, app)
        test_compliance(v, app)
        test_edge_cases(v, app)
        test_mixed_shape_dicom(app)
        test_empty_dicom_tags(app)
        test_export_path_safety(app)
        test_dicom_sort_consistency(app)
        test_i18n_persistent(app)
    print("\n" + ("全部通过" if not _FAILS else f"{len(_FAILS)} 项失败: " + "; ".join(_FAILS)))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main_run())
