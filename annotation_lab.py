# =============================================================================
# 标注与分割 Mixin
# 负责：标注（卡尺/画笔/ROI）CRUD 与渲染、分割蒙版手动编辑（画笔/橡皮/3D 追踪/
#       撤销）、截取统计、器官定量与图例、标注/蒙版工程持久化。
#
# 设计：以 Mixin 形式并入 MedicalViewer。方法通过 self 访问主窗口的 UI 控件与
#       状态（self.views / self.volume_hu / self.volume_mask / self.global_annotations /
#       self._organ_stats / self.organ_names 等）及留在 main.py 的共享方法
#       （_read_dicom_dir / _dcm_float / _safe_name / _export_tag / update_display）。
#       状态（volume_mask / global_annotations / _mask_undo / _hidden_organs /
#       _organ_stats）在 MedicalViewer.__init__ 中初始化；_render_clinical_plane
#       对 _render_annotations、update_display 对 _update_legend 的调用留在 main。
# =============================================================================

import csv
import json
import math
import os
import tempfile
from datetime import datetime

import numpy as np
import scipy.ndimage as ndimage
from PySide6.QtCore import QLineF, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

import mesh3d
import model_card
import quantify
from constants import AXIAL, LABEL_LUT, MANUAL_TRACK_LABEL
from dicom_geometry import series_fingerprint
from graphics_view import ROIGraphicsItem


class MeshView(QLabel):
    """可用鼠标拖动旋转的三维预览控件。

    横向拖动改方位角、纵向拖动改俯仰角，灵敏度 0.5°/px（实测这个值在 360px 视图上
    拖过半屏正好转半圈，手感接近常见的三维查看器）。俯仰角夹在 ±89°：到 ±90° 时
    视线与旋转轴共线，方位角失去意义（万向节锁），画面会在拖动中突然翻转。

    自身只负责「把像素位移换算成角度并发信号」，不碰网格与渲染——渲染策略
    （拖动降质、松手提质）由弹窗持有，因为只有它知道两套网格。
    """
    rotated = Signal(float, float)   # (azimuth, elevation)，已累积的绝对角度
    settled = Signal()               # 松开鼠标：可以做高质量重渲染了

    def __init__(self, azimuth=30.0, elevation=20.0, parent=None):
        super().__init__(parent)
        self.azimuth, self.elevation = float(azimuth), float(elevation)
        self._last = None
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, ev):
        self._last = ev.position(); self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._last is None: return
        p = ev.position(); dx = p.x() - self._last.x(); dy = p.y() - self._last.y()
        self._last = p
        self.azimuth = (self.azimuth + dx * 0.5) % 360.0
        self.elevation = max(-89.0, min(89.0, self.elevation - dy * 0.5))
        self.rotated.emit(self.azimuth, self.elevation)

    def mouseReleaseEvent(self, ev):
        if self._last is None: return
        self._last = None; self.setCursor(Qt.OpenHandCursor); self.settled.emit()

    def set_angles(self, azimuth, elevation):
        """由预设视角按钮调用：直接跳到指定角度（同样走 rotated → settled 两步）。"""
        self.azimuth = float(azimuth) % 360.0
        self.elevation = max(-89.0, min(89.0, float(elevation)))
        self.rotated.emit(self.azimuth, self.elevation); self.settled.emit()


def mask_cache_matches(saved_uid, saved_shape, saved_fingerprint,
                       cur_uid, cur_shape, cur_fingerprint):
    """判断磁盘缓存的分割蒙版能否安全恢复到当前序列（纯函数，无 Qt，可独立单测）。

    只比 shape 是不够的：缓存按 PatientID 命名，而同一患者的随访/复扫序列
    （本软件的双序列对比功能正是为此设计）往往同为 512×512×N —— 只按 shape 匹配
    会把 A 序列的蒙版静默套到 B 序列上，器官定量随之给出错误体积且无任何告警。
    故要求 SeriesInstanceUID 严格相等；缓存或当前序列缺 UID 时一律拒绝：
    宁可重跑 AI，也不返回可能张冠李戴的蒙版。

    返回 (是否可恢复, 拒绝原因)；可恢复时原因为 ''。
    """
    if tuple(saved_shape) != tuple(cur_shape):
        return False, f"shape 不匹配（缓存 {tuple(saved_shape)} vs 当前 {tuple(cur_shape)}）"
    if not saved_uid:
        return False, "缓存未记录 SeriesInstanceUID（旧版本产物），无法确认是否同一序列"
    if not cur_uid:
        return False, "当前序列缺 SeriesInstanceUID，无法确认与缓存是否同源"
    if str(saved_uid) != str(cur_uid):
        return False, "SeriesInstanceUID 不同（同一患者的另一序列），拒绝套用"
    if not saved_fingerprint:
        return False, "缓存缺 geometry/order fingerprint（legacy），无法证明切片对应关系"
    if not cur_fingerprint:
        return False, "当前序列无法生成 geometry/order fingerprint，拒绝自动恢复"
    if str(saved_fingerprint) != str(cur_fingerprint):
        return False, "geometry/order fingerprint 不同，拒绝把缓存套到不同切片顺序"
    return True, ""


# AI 分割输出的轴向契约标识。修复推理左右轴之前落盘的蒙版，其 W 轴是镜像的
# （成对器官标签互换），但 SeriesInstanceUID、shape 与 geometry fingerprint
# 三项都不会因此改变——既有三项守卫识别不出它。故显式记录契约并单独判定。
MASK_AXIS_CONTRACT = 'dicom-cols-left/v2'


def mask_axis_contract_ok(saved_contract):
    """缓存蒙版的轴向契约是否与当前推理路径一致（纯函数，无 Qt，可独立单测）。

    缺失即视为修复前的产物：那批蒙版左右镜像，宁可重跑 AI 也不恢复。
    """
    if not saved_contract:
        return False, "缓存未记录轴向契约（AI 左右方向修复前的产物，蒙版左右镜像）"
    if str(saved_contract) != MASK_AXIS_CONTRACT:
        return False, (f"轴向契约不同（缓存 {saved_contract} vs 当前 {MASK_AXIS_CONTRACT}）")
    return True, ""


class AnnotationMixin:
    """标注 / 分割蒙版编辑 / 器官定量相关方法集合，混入 MedicalViewer。"""

    # =========================================================================
    # 分割蒙版编辑：3D 追踪 / 画笔 / 橡皮 / 撤销
    # =========================================================================
    def handle_3d_track_requested(self, vid, rect):
        """3D 连通域追踪：在当前 Axial 切片上框选 ROI，提取该区域的 HU 统计特征，
        然后在整个 3D 体积中找出 HU 分布相似的连通域，生成 3D 分割蒙版。

        算法原理：
          1. 计算 ROI 的 HU 中位数和标准差（中位数比均值更抗离群值）
          2. 在全体积中找出 HU 在 [med-1.5σ, med+1.5σ] 范围内的体素（类似区域增长）
          3. 对该 HU 范围内的体素做 3D 连通域标记
          4. 选取在 ROI 框内体素最多的连通域标签，即为目标结构
        """
        if (self.volume_hu is None or self.recon_mode_active or self.compare_mode_active
                or not all((getattr(self, 'hu_calibrated', False),
                            getattr(self, 'canonical_orientation', False),
                            getattr(self, 'inplane_spacing_valid', False),
                            getattr(self, 'uniform_z_geometry_valid', False)))):
            return
        if self.views[vid]['plane'] != AXIAL:
            QMessageBox.information(self, "Info" if self.is_english else "提示",
                                    "3D tracking is only available on the axial plane."
                                    if self.is_english else "目前智能追踪仅支持在 Axial 进行。")
            return
        idx = self.current_3d_pos[0]
        x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
        h, w = self.volume_hu.shape[1], self.volume_hu.shape[2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        p = QProgressDialog("Computing 3D..." if self.is_english else "正在计算 3D...", None, 0, 0, self)
        p.setWindowModality(Qt.WindowModal); p.show(); QApplication.processEvents()
        # 【失败必须可区分，且不得改动既有蒙版】原实现把整段包在 try/except: pass 里。
        # 空 ROI（np.median 对空数组返回 nan）、连通域计算抛异常、以及「ROI 内没有任何
        # 前景标签」三种结果全都表现为「点了没反应」，用户无法区分「没找到结构」与
        # 「算失败了」。现分成三条出口各给明确反馈；写蒙版的动作全部推迟到成功之后，
        # 故失败路径连 undo 都不压栈，既有 mask/confidence 逐位不变。
        tracked = None
        fail_msg = None
        e = self.is_english
        try:
            roi = self.volume_hu[idx, y1:y2, x1:x2]
            if roi.size == 0:
                fail_msg = ("The selected region is empty — draw a box with both width and "
                            "height inside the image." if e else
                            "框选区域为空——请在图像内画出同时具有宽和高的方框。")
            else:
                med, std = np.median(roi), np.std(roi)
                if not (np.isfinite(med) and np.isfinite(std)):
                    fail_msg = ("Could not compute HU statistics for the selected region."
                                if e else "无法对框选区域计算 HU 统计量。")
                else:
                    bv = ((self.volume_hu >= med - 1.5 * std)
                          & (self.volume_hu <= med + 1.5 * std))
                    lab, _ = ndimage.label(bv)
                    rl = lab[idx, y1:y2, x1:x2]
                    rl = rl[rl > 0]  # 过滤背景标签 0
                    if len(rl) == 0:
                        fail_msg = ("No connected structure was found in the selected region."
                                    if e else "框选区域内未找到连通结构。")
                    else:
                        # bincount 统计 ROI 内各标签出现次数，取最多的那个为目标
                        tracked = (lab == np.bincount(rl.flatten()).argmax())
        except Exception as exc:                      # 计算失败：如实报错，不动蒙版
            fail_msg = (f"3D tracking failed: {type(exc).__name__}: {exc}" if e else
                        f"3D 追踪计算失败：{type(exc).__name__}: {exc}")
        p.close()
        if fail_msg is not None:
            QMessageBox.warning(self, "3D Tracking" if e else "智能追踪", fail_msg)
            return                                    # 未写蒙版，不必刷新定量与显示
        # —— 以下为成功路径，此前不曾改动任何状态 ——
        if self.volume_mask is None:
            self.volume_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
        self._push_volume_undo()
        # 【只动追踪层，不碰 AI 器官】旧实现在此处整卷赋值，一次追踪就把 ~100s 推理
        # 出的 24 类器官全部抹掉，且经 save_project 落盘后再也恢复不回来（实测：缓存
        # mask 里 100% 体素为 255，器官一个不剩）。现改为：先清掉上一次的追踪结果避免
        # 多次追踪累积，再写入本次；1-24 号器官标签原样保留。
        self.volume_mask[self.volume_mask == MANUAL_TRACK_LABEL] = 0
        self.volume_mask[tracked] = MANUAL_TRACK_LABEL
        # 追踪是用户画的，模型对它没有判断：把这些体素的置信度清成哨兵 0，
        # 否则定量表会拿「模型对该处原本器官的置信度」冒充追踪结果的置信度
        if getattr(self, 'volume_conf', None) is not None \
                and self.volume_conf.shape == self.volume_mask.shape:
            self.volume_conf[tracked] = 0
        self._update_organ_stats()  # 追踪已改写蒙版，定量面板同步刷新
        self.update_display()

    def handle_seg_paint(self, vid, points, is_erase):
        """分割手动修正：把画笔/橡皮轨迹写入当前 Axial 切片的 volume_mask。
        画笔补画为手动标注层(MANUAL_TRACK_LABEL)，橡皮把覆盖处清零（可擦除 AI 误分割）。
        用 QPainter 圆头粗线栅格化轨迹，与 handle_crop 的多边形栅格化同一套做法。
        """
        if self.volume_hu is None or self.recon_mode_active or self.compare_mode_active:
            return
        if self.views[vid]['plane'] != AXIAL or not points:
            return
        if self.volume_mask is None:
            self.volume_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
        z = self.current_3d_pos[0]
        self._push_mask_undo(z)   # 编辑前存快照，支持 Ctrl+Z 撤销
        h, w = self.volume_hu.shape[1], self.volume_hu.shape[2]
        r = max(1, self.views[vid]['view'].brush_radius)
        qi = QImage(w, h, QImage.Format_Grayscale8); qi.fill(Qt.black)
        painter = QPainter(qi)
        pen = QPen(Qt.white, r * 2); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        if len(points) == 1:
            painter.drawPoint(QPointF(points[0][0], points[0][1]))
        else:
            path = QPainterPath(QPointF(points[0][0], points[0][1]))
            for px, py in points[1:]:
                path.lineTo(QPointF(px, py))
            painter.drawPath(path)
        painter.end()
        ma = np.array(qi.constBits(), dtype=np.uint8).reshape((h, qi.bytesPerLine()))[:, :w]
        brush = ma > 0
        # 补画写入所选目标器官标签（修正计入该器官定量）；橡皮清零
        label = 0 if is_erase else int(self.cb_paint_target.currentData() or MANUAL_TRACK_LABEL)
        self.volume_mask[z][brush] = label
        # 手工改过的体素同样清成哨兵 0：其原值是模型对改动前那个标签的置信度
        if getattr(self, 'volume_conf', None) is not None \
                and self.volume_conf.shape == self.volume_mask.shape:
            self.volume_conf[z][brush] = 0
        self._update_organ_stats()
        self.update_display()

    _VOL_UNDO = 'VOL'         # 整卷快照的槽位标记（区别于逐切片的整数切片号）
    _VOL_UNDO_CLEAR = 'VOLC'  # 「清空蒙版」专用槽位——与上者分开保留，理由见 _push_volume_undo

    def _push_mask_undo(self, z):
        """把当前切片蒙版压入撤销栈（上限 20 步，含切片号以便精确回退）。

        置信度必须一起存：画笔与追踪会把改动体素的 conf 清成哨兵 0，而 quantify 用
        conf==0 剔除非模型体素。只还原 mask 的话，撤销后那个器官的 conf_cover 会
        永久 < 1——定量面板于是给一个 100% 来自模型的器官标上「模型判定 XX%」。
        撤销的语义是回到原状态，显示出来的数字也在内。
        """
        cf = None if self.volume_conf is None else self.volume_conf[z].copy()
        self._mask_undo.append((z, self.volume_mask[z].copy(), cf))
        if len(self._mask_undo) > 20:
            self._mask_undo.pop(0)

    def _push_volume_undo(self, slot=None, adopt=False):
        """整卷级操作（3D 追踪 / 清空蒙版）前存一份整卷快照。

        与逐切片快照走同一个栈，但同一槽位只保留最近一份：一份 (Z,H,W) uint8 在
        233×512² 下约 61MB，若像切片那样堆 20 份会吃掉 1.2GB。

        【清空单独占一个槽位】原先所有整卷操作共用一个槽位，于是「清空蒙版」存下的
        那份会被之后任意一次 3D 追踪顶掉——而清空的确认框刚刚写着「可用 Ctrl+Z
        还原蒙版」，被顶掉之后那 ~100 秒的推理产物就真的回不来了。清空是这里破坏性
        最大的一步，它的快照不该被一次普通编辑挤走。

        adopt=True 时直接接管传入的数组而不 copy：清空的调用方本来就要丢弃旧蒙版，
        移交给撤销栈是零成本的，不必为「保留两份整卷快照」多付一份内存。
        """
        if self.volume_mask is None:
            return
        slot = slot or self._VOL_UNDO
        cf = self.volume_conf
        if adopt:
            mask_snap, conf_snap = self.volume_mask, cf
        else:
            mask_snap, conf_snap = self.volume_mask.copy(), (None if cf is None else cf.copy())
        self._mask_undo = [e for e in self._mask_undo if e[0] != slot]
        self._mask_undo.append((slot, mask_snap, conf_snap))
        if len(self._mask_undo) > 20:
            self._mask_undo.pop(0)

    def _undo_mask_edit(self):
        """撤销最近一次分割编辑：整卷快照整卷还原，切片快照只还原该切片。"""
        if not self._mask_undo or self.volume_mask is None:
            return
        z, snap, conf_snap = self._mask_undo.pop()
        if z in (self._VOL_UNDO, self._VOL_UNDO_CLEAR):
            # 换病例后旧快照的形状可能与当前体积不符，形状不合则丢弃不还原
            if snap.shape != self.volume_mask.shape:
                return
            self.volume_mask = snap
            if z == self._VOL_UNDO_CLEAR:
                # 用户撤销了全局清空；后续保存应持久化恢复后的非零 mask，而非 empty intent。
                self._mask_cache_clear_requested = False
            # conf 与 mask 同源同快照：要么一起回退，要么都不动。形状不符时置 None
            # 而不是留着旧的——留着会让 quantify 拿错网格的哨兵去剔体素。
            if conf_snap is not None and conf_snap.shape == snap.shape:
                self.volume_conf = conf_snap
            elif conf_snap is None:
                self.volume_conf = None
        # z 越界保护：换病例后旧切片号可能超出新蒙版层数
        elif z < self.volume_mask.shape[0] and snap.shape == self.volume_mask[z].shape:
            self.volume_mask[z] = snap
            if (conf_snap is not None and self.volume_conf is not None
                    and conf_snap.shape == self.volume_conf[z].shape):
                self.volume_conf[z] = conf_snap
        else:
            return
        self._update_organ_stats()
        if not self.recon_mode_active:
            self.update_display()

    # =========================================================================
    # 截取工具（多边形 ROI 统计 + 可选导出）
    # =========================================================================
    def handle_crop_requested(self, vid, pts):
        """截取工具：对多边形 ROI 区域统计 HU 值，可选保存裁剪图像和 CSV 报告。

        步骤：
          1. 用 QPainter 将多边形栅格化为白色掩码图（白=ROI内，黑=ROI外）
          2. 将掩码转换为 NumPy 数组，提取 ROI 内的 HU 值
          3. 计算面积（像素数 × 像素间距²）和平均 HU
          4. 弹框确认，用户选择是否保存裁剪图像和 CSV 记录
        """
        if (self.recon_mode_active or self.compare_mode_active or self.views[vid]['plane'] != AXIAL
                or not getattr(self, 'hu_calibrated', False)
                or not getattr(self, 'inplane_spacing_valid', False)):
            return
        idx = self.current_3d_pos[0]
        ds = self.dicom_datasets[idx]
        hu = self.volume_hu[idx]
        # 列间距缺省回退到行间距而非 1.0：畸形 DICOM 的 PixelSpacing=[0.7, None] 下，
        # 写死 1.0 会让面积/体积整体偏大（0.7 时 +42.9%）。与 main.py 同一口径。
        _psr = self._dcm_float(ds, 'PixelSpacing', 0.0, idx=0)
        sp = (_psr, self._dcm_float(ds, 'PixelSpacing', 0.0, idx=1))
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
            _msg = (f"Area: {area:.2f} mm²\nMean: {np.mean(rh):.1f} HU\nSave?" if self.is_english
                    else f"面积: {area:.2f} mm²\n均值: {np.mean(rh):.1f} HU\n是否保存？")
            if QMessageBox.question(self, "Stats" if self.is_english else "统计", _msg) == QMessageBox.Yes:
                # 软组织窗归一化：-1250~250 HU 映射到 0~255（保存为 PNG）
                img = np.clip(hu, -1250, 250)
                img = ((img + 1250) / 1500 * 255).astype(np.uint8)
                fn = f"{self._export_tag()}_S{idx+1}_{datetime.now().strftime('%H%M%S')}.png"
                ed = getattr(self, 'export_dir',
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "Exported_Lesions"))
                os.makedirs(ed, exist_ok=True)
                default_path = self._unique_export_path(ed, fn)
                s_p, _ = QFileDialog.getSaveFileName(self, "Save", default_path, "PNG (*.png)")
                if s_p:
                    # img*bm：将 ROI 外的像素清零，保留病灶区域
                    QImage((img * bm).data, w, h, w, QImage.Format_Grayscale8).copy().save(s_p)
                    try:
                        with open(os.path.join(os.path.dirname(s_p), "export_log.csv"), 'a',
                                  newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow([os.path.basename(s_p), idx + 1, round(area, 2), round(np.mean(rh), 2)])
                    except OSError as e:
                        QMessageBox.warning(self, "Export Warning" if self.is_english else "导出警告",
                                            (f"Image saved but log write failed:\n{e}" if self.is_english
                                             else f"图像已保存，但日志写入失败：\n{e}"))

    # =========================================================================
    # 标注 CRUD
    # =========================================================================
    def handle_annotation_added(self, data):
        """将新增标注持久化到内存数据结构，并刷新显示。
        根据 chk_global_scope 决定标注归属：
          - 勾选"穿透所有切片"→ 存入 global_annotations['all']，所有切片可见
          - 未勾选 → 存入 global_annotations[当前切片索引]，仅该切片可见
        """
        if self.recon_mode_active or self.compare_mode_active:
            return
        # 标注体系是【按 axial 层号】存、且 _render_annotations 只在 AXIAL 平面调用。
        # 在冠/矢状面画出来的标注会被存到当前 axial 层号下、按 axial 的 spacing 换算
        # 毫米、再画到另一个视图的不同解剖上——实测拖动中显示 60.0 mm、落库后变
        # 20.0 mm。与其静默产出错误数字，不如在入口拒绝并说明。
        src = self.sender()
        plane = next((vd['plane'] for vd in self.views.values()
                      if vd['view'] is src), AXIAL)
        if plane != AXIAL:
            if not getattr(self, '_warned_nonaxial_anno', False):
                self._warned_nonaxial_anno = True
                e = self.is_english
                QMessageBox.information(
                    self, "Axial only" if e else "仅支持横断面",
                    ("Annotations are stored and rendered per axial slice, so they can only be "
                     "drawn in an Axial view. The measurement you just made was discarded rather "
                     "than saved against the wrong slice. Switch a view to Axial and try again.")
                    if e else
                    ("标注按横断面（Axial）层号存储与绘制，因此只能在 Axial 视图中标注。"
                     "刚才那一笔已被丢弃，而不是错存到别的层上。请把某个视图切到 Axial 后重试。"))
            return
        tk = 'all' if self.chk_global_scope.isChecked() else self.current_3d_pos[0]
        if tk not in self.global_annotations:
            self.global_annotations[tk] = []
        # id 在整条链路上都被当作字符串：渲染时进 setToolTip（只收 str），删除时又从
        # toolTip 取回来比对（annotation_deleted 是 Signal(str)）。一个数字 id 会让
        # setToolTip 抛 TypeError 被渲染层的 except 吞掉——标注既画不出来也删不掉。
        # 故在入口一律规范成 str，而不是在渲染处打补丁。
        if isinstance(data, dict) and 'id' in data:
            data['id'] = str(data['id'])
        self.global_annotations[tk].append(data)
        self.update_display()

    def handle_annotation_deleted(self, aid):
        """按 UUID 从所有切片的标注列表中删除指定标注。
        遍历所有键是因为用户可能在不知情的情况下删除了一个全局标注。
        """
        if self.recon_mode_active or self.compare_mode_active:
            return
        for k in self.global_annotations:
            self.global_annotations[k] = [a for a in self.global_annotations[k] if a['id'] != aid]
        self.update_display()

    def clear_mask_and_annotations(self):
        """清空【当前切片】的标注，并把【整卷】分割蒙版重置为全零。

        两者粒度本就不同（标注按切片、蒙版整卷），旧实现对此不置一词，用户无从
        得知一次点击会波及全部切片；蒙版中若含 AI 器官，清掉意味着 ~100s 的推理
        作废且不可逆。故此处：先算清代价并要求确认，再压入整卷快照（Ctrl+Z 可还原）。
        无可清时直接返回，不弹框骚扰、也不改动任何状态。
        """
        idx = self.current_3d_pos[0]
        n_anno = len(self.global_annotations.get(idx, []))
        has_mask = self.volume_mask is not None and bool(self.volume_mask.any())
        if not n_anno and not has_mask:
            return
        if has_mask:
            organ = (self.volume_mask > 0) & (self.volume_mask != MANUAL_TRACK_LABEL)
            n_organ = int(np.unique(self.volume_mask[organ]).size)
            zs = int(self.volume_mask.shape[0])
            if self.is_english:
                msg = (f"This clears {n_anno} annotation(s) on the current slice AND the "
                       f"segmentation mask on ALL {zs} slices.")
                if n_organ:
                    msg += (f"\n\n{n_organ} AI-segmented organ(s) will be lost; "
                            f"re-running inference takes about 100 s on CPU.")
                msg += "\n\nCtrl+Z restores the mask."
            else:
                msg = f"将清除当前切片的 {n_anno} 条标注，以及【全部 {zs} 层】的分割蒙版。"
                if n_organ:
                    msg += f"\n\n其中含 AI 分割的 {n_organ} 个器官，清除后需重新推理（CPU 约 100 秒）。"
                msg += "\n\n可用 Ctrl+Z 还原蒙版。"
            if QMessageBox.question(self, "Confirm" if self.is_english else "确认清空", msg,
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) != QMessageBox.Yes:
                return
            # 用专属槽位，免得之后一次 3D 追踪把它顶掉——确认框刚承诺过 Ctrl+Z 可还原。
            # adopt=True：旧蒙版与旧 conf 本来就要被丢弃，移交给撤销栈是零成本的。
            self._invalidate_running_ai()  # 先作废旧代次，避免后台结果在清空后复活
            self._push_volume_undo(slot=self._VOL_UNDO_CLEAR, adopt=True)
            self.volume_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
            self.volume_conf = None
            self._mask_cache_clear_requested = True
        if idx in self.global_annotations:
            self.global_annotations[idx] = []
        self._update_organ_stats()  # 蒙版已清，定量面板同步清空
        if not self.recon_mode_active:
            self.update_display()

    # =========================================================================
    # 标注 / 蒙版持久化
    # =========================================================================
    @staticmethod
    def _valid_anno(a):
        """校验单条标注结构完整。用于加载 JSON 时过滤畸形/旧版本/被篡改的条目——
        _render_annotations 会在每次刷新时硬取 type/p1/p2/points/rect，缺字段或类型
        不符会让整个显示刷新崩溃（等于阅片被卡死），故在入口就挡掉不合规条目。"""
        if not isinstance(a, dict) or 'id' not in a:
            return False
        def _pair(p):
            return isinstance(p, (list, tuple)) and len(p) >= 2 \
                and all(isinstance(c, (int, float)) for c in p[:2])
        t = a.get('type')
        if t == 'ruler':
            return _pair(a.get('p1')) and _pair(a.get('p2'))
        if t == 'path':
            pts = a.get('points')
            return isinstance(pts, (list, tuple)) and len(pts) >= 1 and all(_pair(p) for p in pts)
        if t == 'roi':
            r = a.get('rect')
            return isinstance(r, (list, tuple)) and len(r) == 4 \
                and all(isinstance(c, (int, float)) for c in r)
        return False

    def _load_annotations_json(self, pid):
        """尝试加载同 PatientID 命名的注解 JSON 文件，恢复历史标注。
        文件不存在/损坏静默跳过；结构畸形的单条标注被过滤（不带崩后续渲染）。"""
        ed = getattr(self, 'persistence_dir',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"))
        af = os.path.join(ed, f"{self._safe_name(pid)}_annotations.json")
        if not os.path.exists(af):
            return
        try:
            with open(af, encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return
            meta = raw.pop('__meta__', None)
            saved_uid = (meta or {}).get('series_uid', '') if isinstance(meta, dict) else ''
            saved_fingerprint = ((meta or {}).get('geometry_fingerprint', '')
                                 if isinstance(meta, dict) else '')
            cur_uid = self._current_series_uid()
            cur_fingerprint = self._current_geometry_fingerprint()
            if (not saved_uid or not cur_uid or saved_uid != cur_uid
                    or not saved_fingerprint or saved_fingerprint != cur_fingerprint):
                # 明确是另一序列：拒绝。标注坐标按切片号存，套到同形状的另一序列上
                # 会落在完全不同的解剖上，而测距/ROI 会照样给出数字。
                print("标注文件缺少或不匹配 geometry/order fingerprint，"
                      f"已跳过加载：{af}")
                e = self.is_english
                QMessageBox.information(
                    self, "Annotations not loaded" if e else "标注未加载",
                    ("The saved annotations lack a matching geometry/order fingerprint and "
                     "were not loaded; their slice indices cannot be proven safe. The file "
                     "was left untouched.") if e else
                    ("已保存标注缺少匹配的 geometry/order fingerprint，无法证明切片号对应"
                     "关系，故未加载；原文件保留在磁盘上，未改动。"))
                return
            for k, v in raw.items():
                # JSON 键只能是字符串，数字键需要转回 int
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                annos = v if isinstance(v, list) else []
                valid = [a for a in annos if self._valid_anno(a)]
                for a in valid:
                    a['id'] = str(a['id'])   # 同上：外部编辑过的 JSON 常见数字 id
                if len(valid) != len(annos):
                    print(f"标注键 {k!r}: 跳过 {len(annos) - len(valid)} 条畸形/旧版本条目")
                self.global_annotations[key] = valid
        except Exception as e:
            print(f"Warning: failed to load annotations from {af}: {e}")

    def _current_series_uid(self):
        """当前序列的 SeriesInstanceUID；无数据或畸形 DICOM 缺该标签时返回 ''。"""
        if not self.dicom_datasets:
            return ''
        return str(getattr(self.dicom_datasets[0], 'SeriesInstanceUID', '') or '')

    def _current_geometry_fingerprint(self):
        """当前有序 volume 的稳定 geometry/order SHA-256；无法证明时为空。"""
        if self.volume_hu is None:
            return ''
        return series_fingerprint(self.dicom_datasets, self.volume_hu.shape)

    def _load_saved_mask(self, pid):
        """尝试加载上次保存的 AI 分割标签图(.npz)。自动恢复要求 SeriesInstanceUID、
        mask/volume shape 与 geometry/order fingerprint 三者均匹配；legacy cache 缺少
        fingerprint 时 fail closed。判定逻辑在纯函数 mask_cache_matches（无 Qt，可独立
        单测），避免同一患者的随访/复扫、切片重排或几何变化静默复用错误蒙版。
        """
        ed = getattr(self, 'persistence_dir',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"))
        fp = os.path.join(ed, f"{self._safe_name(pid)}_mask.npz")
        if not os.path.exists(fp):
            return False
        try:
            z = np.load(fp)
            m = z['mask']
            saved_uid = str(z['series_uid'].item()) if 'series_uid' in z.files else ''
            saved_fingerprint = (str(z['geometry_fingerprint'].item())
                                 if 'geometry_fingerprint' in z.files else '')
            saved_contract = (str(z['axis_contract'].item())
                              if 'axis_contract' in z.files else '')
            ok, why = mask_axis_contract_ok(saved_contract)
            if ok:
                ok, why = mask_cache_matches(saved_uid, m.shape, saved_fingerprint,
                                             self._current_series_uid(), self.volume_hu.shape,
                                             self._current_geometry_fingerprint())
            if not ok:
                print(f"跳过磁盘缓存的分割蒙版：{why}；将重新运行 AI 分割。")
                return False
            self.volume_mask = m.astype(np.uint8)
            self._mask_cache_clear_requested = False
            return True
        except Exception as e:
            print(f"Warning: failed to load saved mask: {e}")
        return False

    def save_project(self):
        """将当前所有标注保存为 JSON 文件（以 PatientID 命名），方便下次加载时自动恢复。
        JSON 键必须为字符串（JSON 规范），整数切片索引在此序列化为字符串，加载时再转回 int。

        单个目标文件通过同目录临时文件 + ``os.replace`` 原子替换；JSON 与 NPZ 是两个
        独立目标，因此不声称跨文件事务原子性。任何前置条件、序列化或替换失败均返回
        False，且不会显示成功提示。
        """
        if not self.dicom_datasets:
            QMessageBox.warning(self, "Save Failed" if self.is_english else "保存失败",
                                ("No DICOM series is loaded."
                                 if self.is_english else "尚未加载 DICOM 序列。"))
            return False

        series_uid = self._current_series_uid().strip()
        geometry_fingerprint = self._current_geometry_fingerprint().strip()
        if not series_uid or not geometry_fingerprint:
            missing = []
            if not series_uid:
                missing.append("SeriesInstanceUID")
            if not geometry_fingerprint:
                missing.append("geometry fingerprint")
            QMessageBox.warning(
                self, "Save Failed" if self.is_english else "保存失败",
                (("Project persistence requires verified " + " and ".join(missing) + ".")
                 if self.is_english else
                 ("工程持久化缺少可验证的 " + " 和 ".join(missing) + "，未写入任何文件。")))
            return False

        mask_present = self.volume_mask is not None
        # 任何 ndarray（包括全零）都先验 shape；不能因为 np.any=False 就绕过 payload 校验，
        # 让 wrong-shape zero 先覆盖 JSON、再留下无法安全恢复的旧 NPZ。
        if mask_present and (self.volume_hu is None
                             or tuple(self.volume_mask.shape) != tuple(self.volume_hu.shape)):
            QMessageBox.warning(
                self, "Save Failed" if self.is_english else "保存失败",
                ("The segmentation mask shape does not match the current volume."
                 if self.is_english else "分割蒙版 shape 与当前 volume 不一致，未写入任何文件。"))
            return False
        has_nonzero_mask = mask_present and bool(np.any(self.volume_mask))
        explicit_empty = (mask_present and not has_nonzero_mask
                          and getattr(self, '_mask_cache_clear_requested', False))
        # fresh/AI-pending placeholder zero 不写 NPZ；用户明确清空则写带 provenance 的零 NPZ，
        # 使重开后仍为空且跳过 AI，防止旧非零 cache 复活。
        write_mask = bool(has_nonzero_mask or explicit_empty)

        pid = self._safe_name(str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown')))
        ed = getattr(self, 'persistence_dir',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"))
        json_target = os.path.join(ed, f"{pid}_annotations.json")
        npz_target = os.path.join(ed, f"{pid}_mask.npz")
        payload = {'__meta__': {
                       'series_uid': series_uid,
                       'geometry_fingerprint': geometry_fingerprint,
                   },
                   **{str(k): v for k, v in self.global_annotations.items()}}
        temp_paths = []
        replaced = []
        replacement_target = None
        try:
            os.makedirs(ed, exist_ok=True)

            # 先完整序列化所有临时文件；任一写入失败时，既有目标文件均保持不变。
            json_fd, json_temp = tempfile.mkstemp(
                prefix=f".{pid}_annotations.", suffix=".tmp", dir=ed)
            temp_paths.append(json_temp)
            with os.fdopen(json_fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4)
                f.flush(); os.fsync(f.fileno())

            if write_mask:
                npz_fd, npz_temp = tempfile.mkstemp(
                    prefix=f".{pid}_mask.", suffix=".tmp", dir=ed)
                temp_paths.append(npz_temp)
                with os.fdopen(npz_fd, 'wb') as f:
                    np.savez_compressed(f, mask=self.volume_mask,
                                        series_uid=np.array(series_uid),
                                        geometry_fingerprint=np.array(geometry_fingerprint),
                                        axis_contract=np.array(MASK_AXIS_CONTRACT))
                    f.flush(); os.fsync(f.fileno())

            replacement_target = os.path.basename(json_target)
            os.replace(json_temp, json_target); replaced.append(replacement_target)
            if write_mask:
                replacement_target = os.path.basename(npz_target)
                os.replace(npz_temp, npz_target); replaced.append(replacement_target)

            if write_mask:
                # 只有所有目标替换均成功才消费 intent；失败路径保留 True，允许用户重试。
                self._mask_cache_clear_requested = False

            if self.anonymize:
                QMessageBox.warning(
                    self, "Internal project identifiers" if self.is_english else "内部工程仍含标识",
                    ("De-ID does not anonymize internal project persistence. The JSON/NPZ cache "
                     "retains PatientID/SeriesInstanceUID and geometry provenance for safe reload."
                     if self.is_english else
                     "脱敏开关不会匿名化内部工程持久化。为安全恢复与 provenance，JSON/NPZ 缓存"
                     "仍保留 PatientID/SeriesInstanceUID 和几何标识。"))
            QMessageBox.information(self, "Success" if self.is_english else "成功",
                                    "Project saved." if self.is_english else "标注工程已保存。")
            return True
        except Exception as e:
            partial = ""
            if replacement_target is not None:
                joined = ", ".join(replaced)
                completed = joined or "none"
                partial = ((f"\nReplacement failed for {replacement_target}; already replaced: "
                            f"{completed}. Cross-file atomicity is not provided.")
                           if self.is_english else
                           (f"\n替换失败目标：{replacement_target}；已替换目标："
                            f"{joined or '无'}。本操作不提供跨文件原子性。"))
            QMessageBox.warning(self, "Save Failed" if self.is_english else "保存失败",
                                ((f"Failed to save project:\n{e}{partial}") if self.is_english
                                 else f"标注工程保存失败：\n{e}{partial}"))
            return False
        finally:
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except OSError:
                    pass

    # =========================================================================
    # 器官定量 / 图例
    # =========================================================================
    def _compute_organ_stats(self):
        """器官定量薄包装：读取 self 的体积/蒙版/spacing 后调用纯函数
        quantify.compute_organ_stats（无 Qt，可独立单测）。返回按体积降序的列表。"""
        if (self.volume_mask is None or self.volume_hu is None or not self.dicom_datasets
                or not all((getattr(self, 'hu_calibrated', False),
                            getattr(self, 'canonical_orientation', False),
                            getattr(self, 'inplane_spacing_valid', False),
                            getattr(self, 'uniform_z_geometry_valid', False)))):
            return []
        ds = self.dicom_datasets[0]
        _psr = self._dcm_float(ds, 'PixelSpacing', 0.0, idx=0)
        spacing = (_psr, self._dcm_float(ds, 'PixelSpacing', 0.0, idx=1),
                   self._slice_spacing())
        if not all(value > 0 for value in spacing):
            return []
        # volume_conf 只在 ONNX 路径产出；手工编辑过蒙版后形状仍一致，故置信度沿用
        # 原推理结果——注意画笔改过的体素其置信度并非模型对新标签的置信度，
        # 这一点在定量面板的提示文案里已写明。
        return quantify.compute_organ_stats(self.volume_hu, self.volume_mask, spacing,
                                            self.organ_names, getattr(self, 'volume_conf', None))

    def _update_organ_stats(self):
        """刷新器官定量面板；无分割结果时清空并禁用导出按钮。"""
        self._organ_stats = self._compute_organ_stats()
        self._refresh_paint_target()   # 无论有无器官都刷新画笔目标下拉
        if not self._organ_stats:
            self.lbl_ai_stats.setText("")
            self.btn_export_stats.setEnabled(False)
            self.btn_mesh3d.setEnabled(False)
            return
        e = self.is_english
        lines = []
        for r in self._organ_stats:
            r_, g_, b_ = (int(LABEL_LUT[r['id']][0]), int(LABEL_LUT[r['id']][1]), int(LABEL_LUT[r['id']][2]))
            nm = r['name_en'] if e else r['name_zh']
            # 与椭圆 ROI 同口径给出 mean±SD：只报均值无法反映区域内密度离散程度
            txt = (f'<span style="color:#{r_:02X}{g_:02X}{b_:02X};">■</span> '
                   f"{nm}: {r['volume_ml']:.1f} mL / {r['mean_hu']:.0f}±{r['sd_hu']:.0f} HU")
            if 'mean_conf' in r:
                # 低置信标红：模型自己都不确信的器官，读数不该和高置信的一样呈现。
                # 阈值 0.9 取自 softmax 最大类概率的经验分界，仅作视觉提示，非诊断阈值。
                c = r['mean_conf']
                col = '#E67E22' if c < 0.9 else '#7F8C8D'
                extra = ''
                # 覆盖率明显不足 1 时必须标出：该器官已被大量手工改动，
                # 此时的 conf 只代表剩下那部分模型体素，不是整个器官的置信度
                if r.get('conf_cover', 1.0) < 0.98:
                    extra = (f" · {'model' if e else '模型判定'} {100*r['conf_cover']:.0f}%")
                txt += (f' <span style="color:{col};font-size:10px;">'
                        f"conf {c:.2f}/p5 {r['p5_conf']:.2f}{extra}</span>")
            lines.append(txt)
        # 置信度的口径说明做成悬停提示而非常驻文字：它是查一次就记住的静态解释，
        # 常驻会占掉两行并被面板宽度裁断（实测截图里就被裁成「多为边界体…」），
        # 而它旁边每一行都是随数据变化的实时读数，两者不该抢同样的版面。
        self.lbl_ai_stats.setToolTip(
            "conf = softmax max-probability; p5 = 5th percentile, mostly boundary voxels.\n"
            "Voxels edited by hand or written by 3D tracking are excluded — their stored\n"
            "value describes the label that was there before the edit."
            if e else
            "conf = 模型 softmax 最大类概率；p5 = 5% 分位，多为边界体素。\n"
            "画笔改过或 3D 追踪写入的体素不计入——它们的原值描述的是改动前那个标签。")
        self.lbl_ai_stats.setText("<br>".join(lines))
        self.btn_export_stats.setEnabled(True)
        self.btn_mesh3d.setEnabled(True)

    def show_model_card(self):
        """弹出模型说明卡：出处如何被推断出来、实测到什么程度、有哪些已知局限。

        内容全部由 model_card 从已跑出的实验产物现读现算，不在 UI 层硬编码任何数字——
        实验重跑后卡片自动跟着变，避免界面上的指标与 results/ 里的产物各说各话。
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(model_card.card_title(self.is_english))
        dlg.resize(560, 520)
        lay = QVBoxLayout(dlg)
        body = QLabel(model_card.build_model_card(self.is_english))
        body.setWordWrap(True); body.setTextFormat(Qt.RichText)
        body.setAlignment(Qt.AlignTop)
        body.setStyleSheet("font-size: 12px; line-height: 150%;")
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(body)
        lay.addWidget(sc)
        btn = QPushButton("Close" if self.is_english else "关闭")
        btn.clicked.connect(dlg.accept); lay.addWidget(btn)
        dlg.exec()

    def show_mesh3d(self):
        """对当前画笔目标所指器官做三维表面重建，弹窗展示四视角预览 + 形状特征 + STL 导出。

        取 cb_paint_target 的选中项作为对象，与画笔编辑保持同一"当前器官"语义，
        不再单设一个下拉——多一个状态就多一处可能不同步。
        """
        if (self.volume_mask is None or not self._organ_stats
                or not all((getattr(self, 'canonical_orientation', False),
                            getattr(self, 'inplane_spacing_valid', False),
                            getattr(self, 'uniform_z_geometry_valid', False)))):
            return
        lid = self.cb_paint_target.currentData()
        e = self.is_english
        if lid is None or not (self.volume_mask == lid).any():
            QMessageBox.information(self, "3D", "Selected target has no voxels." if e
                                    else "所选目标在当前蒙版中没有体素。")
            return
        ds = self.dicom_datasets[0] if self.dicom_datasets else None
        # extract_surface 的 spacing 契约是 (行间距, 列间距, 层厚)，两者不可混用同一个值：
        # 面内各向异性时，网格的体积/表面积/球形度与导出 STL 的尺寸会整体错，而数量级
        # 仍然对得上，肉眼看不出来。此前这里传的是 (ps, ps, st)。
        ps_row = self._dcm_float(ds, 'PixelSpacing', 0.0, idx=0) if ds is not None else 0.0
        ps_col = self._dcm_float(ds, 'PixelSpacing', 0.0, idx=1) if ds is not None else 0.0
        # z 尺度取层间距而非层厚（重叠重建下二者可差一倍，网格会被拉伸/压扁）
        st = self._slice_spacing() if ds is not None else 0.0
        if not all(value > 0 for value in (ps_row, ps_col, st)):
            return
        # marching cubes 在 512² 体积上 step=1 约 1.4s、step=2 约 0.11s（实测），
        # 故取 2：这是交互预览，不是几何精算；耗时与精度的取舍在 mesh3d 模块注释里说明。
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            verts, faces = mesh3d.extract_surface(self.volume_mask, lid, (ps_row, ps_col, st), step=2)
            stats = mesh3d.mesh_shape_stats(verts, faces)
        finally:
            QApplication.restoreOverrideCursor()
        if len(faces) == 0:
            QMessageBox.information(self, "3D", "Too few voxels to build a surface." if e
                                    else "体素过少，无法构成表面。")
            return
        self._show_mesh_dialog(lid, verts, faces, stats)

    def _show_mesh_dialog(self, lid, verts, faces, stats):
        """三维预览弹窗：可鼠标拖动旋转的渲染视图 + 预设视角 + 形状特征 + STL 导出。"""
        e = self.is_english
        nm = next((r['name_en'] if e else r['name_zh'] for r in self._organ_stats if r['id'] == lid),
                  f"label {lid}")
        rgb = (int(LABEL_LUT[lid][0]), int(LABEL_LUT[lid][1]), int(LABEL_LUT[lid][2]))
        dlg = QDialog(self)
        dlg.setWindowTitle(f"3D · {nm}")
        lay = QVBoxLayout(dlg)

        # 拖动降质、松手提质：实测完整网格渲染约 100–140ms/帧，拖动时会明显顿挫；
        # 再减一档面到 grid=16（约 2000 面、~55ms）拖起来才跟手。松开鼠标后立刻用
        # 完整网格重渲染一帧，所以静止时看到的始终是全精度画面。
        # 这只影响预览显示——形状特征与 STL 导出一律用完整网格 verts/faces。
        dv, df = mesh3d.decimate_vertex_clustering(verts, faces, grid=16)
        SZ = 360
        view = MeshView(azimuth=30.0, elevation=20.0)
        view.setFixedSize(SZ, SZ)
        view.setStyleSheet("background:#0D1117; border:1px solid #30363D;")
        lb_ang = QLabel(); lb_ang.setStyleSheet("color:#8B949E; font-size:10px;")

        def paint(v, f):
            arr = np.ascontiguousarray(mesh3d.render_mesh(v, f, size=SZ, azimuth=view.azimuth,
                                                          elevation=view.elevation, rgb=rgb))
            h, w = arr.shape[:2]
            view.setPixmap(QPixmap.fromImage(
                QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()))
            lb_ang.setText(("Azimuth %.0f° · Elevation %.0f° — drag to rotate" if e else
                            "方位角 %.0f° · 俯仰角 %.0f° —— 按住拖动可旋转")
                           % (view.azimuth, view.elevation))

        # 防重入：鼠标移动事件比一帧渲染密得多，不设闸门会积压成越拖越卡的事件队列。
        # 丢弃渲染中到达的中间帧不影响正确性——下一帧用的是控件里最新的绝对角度。
        busy = {'v': False}
        def on_rotate(_az, _el):
            if busy['v']: return
            busy['v'] = True
            try: paint(dv, df)
            finally: busy['v'] = False
        view.rotated.connect(on_rotate)
        view.settled.connect(lambda: paint(verts, faces))

        row = QHBoxLayout(); row.addWidget(view); row.addStretch()
        col = QVBoxLayout()
        col.addWidget(QLabel("View" if e else "视角"))
        for label_zh, label_en, az, el in (("前", "Ant", 90, 0), ("后", "Post", 270, 0),
                                           ("左", "Left", 180, 0), ("右", "Right", 0, 0),
                                           ("上", "Sup", 90, 89), ("斜", "Oblique", 30, 20)):
            b = QPushButton(label_en if e else label_zh); b.setFixedWidth(74)
            b.clicked.connect(lambda _=False, a=az, elv=el: view.set_angles(a, elv))
            col.addWidget(b)
        col.addStretch(); row.addLayout(col)
        lay.addLayout(row)
        lay.addWidget(lb_ang)
        paint(verts, faces)
        # 形状特征：体积可信；表面积/球形度受 marching cubes 阶梯效应系统性影响，
        # 故在界面上直接标注"相对比较用"，避免被当作绝对几何量引用。
        txt = (f"Surface {stats['surface_area_mm2'] / 100:.1f} cm² · "
               f"Volume {stats['volume_mm3'] / 1000:.1f} mL · "
               f"Sphericity {stats['sphericity']:.3f} · "
               f"{stats['n_faces']:,} faces" if e else
               f"表面积 {stats['surface_area_mm2'] / 100:.1f} cm² · "
               f"体积 {stats['volume_mm3'] / 1000:.1f} mL · "
               f"球形度 {stats['sphericity']:.3f} · "
               f"{stats['n_faces']:,} 面片")
        lb_s = QLabel(txt); lb_s.setStyleSheet("color:#C9D1D9;"); lay.addWidget(lb_s)
        note = QLabel("Pipeline: marching cubes → Taubin smoothing → decimation. On an analytic "
                      "sphere: volume within 0.1%, surface area within ~1.3%."
                      if e else
                      "流程：marching cubes → Taubin 平滑 → 减面。解析球体验算：体积误差 0.1% 以内，"
                      "表面积误差约 1.3%。")
        note.setWordWrap(True); note.setStyleSheet("color:#8B949E; font-size:10px;")
        lay.addWidget(note)
        btn = QPushButton("Export STL" if e else "导出 STL")
        btn.clicked.connect(lambda: self._export_stl(nm, verts, faces))
        lay.addWidget(btn)
        dlg.exec()

    def _export_stl(self, nm, verts, faces):
        """把网格写为 ASCII STL 到 Exported_Lesions/（文件名经 _safe_name 净化）。"""
        e = self.is_english
        ed = getattr(self, 'export_dir',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"))
        os.makedirs(ed, exist_ok=True)
        tag = self._export_tag() if self.anonymize else self._safe_name(
            str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown')))
        fp = self._unique_export_path(ed, f"{tag}_{self._safe_name(nm)}.stl")
        try:
            with open(fp, 'wb') as f:
                f.write(mesh3d.to_stl_bytes(verts, faces, self._safe_name(nm)))
            QMessageBox.information(self, "Success" if self.is_english else "成功",
                                    (f"Saved: {os.path.basename(fp)}" if self.is_english
                                     else f"已保存：{os.path.basename(fp)}"))
        except Exception as ex:
            QMessageBox.warning(self, "Export Failed" if e else "导出失败", str(ex))

    def _refresh_paint_target(self):
        """刷新画笔目标下拉：手动标注 + 当前蒙版中检出的各器官。
        让画笔可把修正直接补进指定器官（该器官定量随之更新），而非只写无名手动层。"""
        cur = self.cb_paint_target.currentData()
        e = self.is_english
        self.cb_paint_target.blockSignals(True)
        self.cb_paint_target.clear()
        self.cb_paint_target.addItem("Manual" if e else "手动标注", MANUAL_TRACK_LABEL)
        for r in self._organ_stats:
            if r['id'] == MANUAL_TRACK_LABEL:
                continue
            nm = r['name_en'] if e else r['name_zh']
            self.cb_paint_target.addItem(f"{nm} (#{r['id']})", r['id'])
        idx = self.cb_paint_target.findData(cur)
        self.cb_paint_target.setCurrentIndex(max(0, idx))
        self.cb_paint_target.blockSignals(False)

    def export_organ_stats(self):
        """将器官定量结果导出为 CSV（utf-8-sig 便于 Excel 正确显示中文）。"""
        rows = self._organ_stats or self._compute_organ_stats()
        if not rows:
            return
        tag = self._export_tag() if self.anonymize else self._safe_name(
            str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown')))
        ed = getattr(self, 'export_dir',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"))
        os.makedirs(ed, exist_ok=True)
        fp = self._unique_export_path(ed, f"{tag}_organ_stats.csv")
        try:
            with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['# AI-derived, for reference only, NOT for diagnosis; organ labels are auto-inferred /'
                            ' AI 自动推断，仅供参考，非诊断依据'])
                # 置信度列只在模型真的产出概率时才写；数学降级路径没有概率，
                # 那时整列缺席，而不是填 0 或 1 让下游误以为有这个量
                has_conf = bool(rows) and 'mean_conf' in rows[0]
                hdr = ['class_id', 'organ_zh', 'organ_en', 'voxels', 'volume_mL',
                       'mean_HU', 'SD_HU', 'median_HU', 'p5_HU', 'p95_HU', 'min_HU', 'max_HU']
                w.writerow(hdr + (['mean_conf', 'p5_conf', 'conf_cover'] if has_conf else []))
                for r in rows:
                    row = [r['id'], r['name_zh'], r['name_en'], r['voxels'],
                           f"{r['volume_ml']:.2f}", f"{r['mean_hu']:.1f}", f"{r['sd_hu']:.1f}",
                           f"{r['median_hu']:.1f}", f"{r['p5_hu']:.1f}", f"{r['p95_hu']:.1f}",
                           f"{r['min_hu']:.1f}", f"{r['max_hu']:.1f}"]
                    if has_conf:
                        # conf_cover<1 表示该器官被手工改过，读数只覆盖剩余的模型体素
                        row += [f"{r.get('mean_conf', float('nan')):.4f}",
                                f"{r.get('p5_conf', float('nan')):.4f}",
                                f"{r.get('conf_cover', float('nan')):.4f}"]
                    w.writerow(row)
            QMessageBox.information(self, "Success" if self.is_english else "成功",
                                    (f"Saved: {os.path.basename(fp)}" if self.is_english
                                     else f"已保存：{os.path.basename(fp)}"))
        except Exception as ex:
            QMessageBox.warning(self, "Export Failed" if self.is_english else "导出失败", str(ex))

    def _update_legend(self, labels):
        """刷新图例：每个器官为可点击项（色块+名称），点击切换其在蒙版叠加中的显隐。
        已隐藏的项显示为灰色删除线。"""
        if len(labels) == 0:
            self.lbl_ai_legend.setText("")
            return
        e = self.is_english
        parts = []
        for lb in labels:
            lb = int(lb)
            r, g, b = (int(LABEL_LUT[lb][0]), int(LABEL_LUT[lb][1]), int(LABEL_LUT[lb][2]))
            name = self.organ_names.get(lb, (f"类{lb}", f"cls{lb}"))[1 if e else 0]
            hidden = lb in self._hidden_organs
            swatch = "#555555" if hidden else f"#{r:02X}{g:02X}{b:02X}"
            deco = "text-decoration:line-through; color:#666;" if hidden else "color:#B0B8C4;"
            parts.append(f'<a href="toggle:{lb}" style="text-decoration:none;">'
                         f'<span style="color:{swatch};">■</span>'
                         f'<span style="{deco}"> {name}</span></a>')
        title = "Detected: " if e else "检出器官: "
        self.lbl_ai_legend.setText(title + "&nbsp;&nbsp;".join(parts))

    def _toggle_organ(self, href):
        """图例项被点击：切换该器官类别在蒙版叠加中的显隐，并重绘。"""
        try:
            lid = int(href.split(":")[1])
        except (IndexError, ValueError):
            return
        self._hidden_organs.discard(lid) if lid in self._hidden_organs else self._hidden_organs.add(lid)
        if not self.recon_mode_active:
            self.update_display()  # 重绘 overlay，并经 _update_legend 刷新图例样式

    # =========================================================================
    # 标注渲染（仅 Axial，供 _render_clinical_plane 调用）
    # =========================================================================
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
              # 逐条兜底：万一有畸形标注漏过加载期过滤，也只跳过这一条，绝不拖垮整次刷新
              try:
                if anno['type'] == 'ruler':
                    line = QGraphicsLineItem(QLineF(anno['p1'][0], anno['p1'][1], anno['p2'][0], anno['p2'][1]))
                    line.setPen(QPen(col, 2))
                    line.setToolTip(anno['id'])          # toolTip 存 UUID，Delete 键删除时用
                    line.setFlag(QGraphicsLineItem.ItemIsSelectable)
                    vdata['view'].scene.addItem(line)
                    if getattr(self, 'inplane_spacing_valid', False):
                        dist = math.sqrt(
                            ((anno['p2'][0] - anno['p1'][0]) * sp[1]) ** 2 +
                            ((anno['p2'][1] - anno['p1'][1]) * sp[0]) ** 2
                        )
                        label = f"{dist:.1f} mm"
                    else:
                        label = "distance unavailable" if self.is_english else "距离不可用"
                    txt = QGraphicsTextItem(label)
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
                elif anno['type'] == 'roi':
                    rx0, ry0, rw, rh = anno['rect']
                    # 可拖动+可缩放的 ROI；改动后经 update_display 回调重算统计并重绘
                    ell = ROIGraphicsItem(anno, self.update_display)
                    ell.set_appearance(col)
                    ell.setToolTip(anno['id'])
                    vdata['view'].scene.addItem(ell)
                    # 椭圆内 HU 统计：用 numpy 椭圆掩码取 volume_hu[z] 内部体素
                    H, W = self.volume_hu.shape[1], self.volume_hu.shape[2]
                    cx, cy = rx0 + rw / 2.0, ry0 + rh / 2.0
                    ax, ay = max(rw / 2.0, 0.5), max(rh / 2.0, 0.5)
                    yy, xx = np.ogrid[:H, :W]
                    emask = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
                    vals = self.volume_hu[z][emask]
                    if vals.size:
                        parts = []
                        if getattr(self, 'hu_calibrated', False):
                            parts += [f"{vals.mean():.0f}±{vals.std():.0f} HU",
                                      f"[{vals.min():.0f}, {vals.max():.0f}]"]
                        else:
                            parts.append("HU unavailable" if self.is_english else "HU 不可用")
                        if getattr(self, 'inplane_spacing_valid', False):
                            area = vals.size * sp[0] * sp[1]
                            parts.append(f"{area:.0f} mm²")
                        else:
                            parts.append("area unavailable" if self.is_english else "面积不可用")
                        stat = "\n".join(parts)
                        txt = QGraphicsTextItem(stat)
                        txt.setDefaultTextColor(col)
                        txt.setFont(QFont("Arial", 10, QFont.Bold))
                        # 防跑出画面：右侧放不下则移到椭圆左侧，纵向夹取在图像内
                        tx = rx0 + rw + 4
                        if tx + 95 > W:
                            tx = max(0, rx0 - 95)
                        ty = min(max(0.0, ry0), max(0.0, H - 46))
                        txt.setPos(tx, ty)
                        vdata['view'].scene.addItem(txt)
              except Exception as _e:
                  print(f"跳过畸形标注: {_e}")
                  continue
