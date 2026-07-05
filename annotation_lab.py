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
from datetime import datetime

import numpy as np
import scipy.ndimage as ndimage
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QMessageBox,
    QProgressDialog,
)

import quantify
from constants import AXIAL, LABEL_LUT, MANUAL_TRACK_LABEL
from graphics_view import ROIGraphicsItem


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
        if self.volume_hu is None or self.recon_mode_active or self.compare_mode_active:
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
                # 赋专属标签值 MANUAL_TRACK_LABEL(255)，与 AI 器官类别(1-24)区分显示
                self.volume_mask = (lab == np.bincount(rl.flatten()).argmax()).astype(np.uint8) * MANUAL_TRACK_LABEL
        except Exception:
            pass
        p.close()
        self._update_organ_stats()  # 追踪已替换蒙版，定量面板同步刷新，避免残留旧 AI 数据
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
        self._update_organ_stats()
        self.update_display()

    def _push_mask_undo(self, z):
        """把当前切片蒙版压入撤销栈（上限 20 步，含切片号以便精确回退）。"""
        self._mask_undo.append((z, self.volume_mask[z].copy()))
        if len(self._mask_undo) > 20:
            self._mask_undo.pop(0)

    def _undo_mask_edit(self):
        """撤销最近一次分割编辑，恢复对应切片的蒙版。"""
        if not self._mask_undo or self.volume_mask is None:
            return
        z, snap = self._mask_undo.pop()
        # z 越界保护：换病例后旧切片号可能超出新蒙版层数
        if z < self.volume_mask.shape[0] and snap.shape == self.volume_mask[z].shape:
            self.volume_mask[z] = snap
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
        if self.recon_mode_active or self.compare_mode_active or self.views[vid]['plane'] != AXIAL:
            return
        idx = self.current_3d_pos[0]
        ds = self.dicom_datasets[idx]
        hu = self.volume_hu[idx]
        sp = (self._dcm_float(ds, 'PixelSpacing', 1.0, idx=0), self._dcm_float(ds, 'PixelSpacing', 1.0, idx=1))
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
                fn = f"{self._export_tag()}_S{idx+1}_{datetime.now().strftime('%H%M%S')}.png"
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
                    except OSError as e:
                        QMessageBox.warning(self, "Export Warning", f"Image saved but log write failed:\n{e}")

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
        tk = 'all' if self.chk_global_scope.isChecked() else self.current_3d_pos[0]
        if tk not in self.global_annotations:
            self.global_annotations[tk] = []
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

    def clear_current_slice_annotations(self):
        """清空当前切片的标注，并将整个 3D 蒙版重置为全零。
        注意：蒙版是 3D 的，清空操作影响全部切片（AI 分割结果一并清除）。
        """
        idx = self.current_3d_pos[0]
        if idx in self.global_annotations:
            self.global_annotations[idx] = []
        if self.volume_mask is not None:
            self.volume_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
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
        af = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Exported_Lesions", f"{self._safe_name(pid)}_annotations.json")
        if not os.path.exists(af):
            return
        try:
            with open(af, encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return
            for k, v in raw.items():
                # JSON 键只能是字符串，数字键需要转回 int
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                annos = v if isinstance(v, list) else []
                valid = [a for a in annos if self._valid_anno(a)]
                if len(valid) != len(annos):
                    print(f"标注键 {k!r}: 跳过 {len(annos) - len(valid)} 条畸形/旧版本条目")
                self.global_annotations[key] = valid
        except Exception as e:
            print(f"Warning: failed to load annotations from {af}: {e}")

    def _load_saved_mask(self, pid):
        """尝试加载上次保存的 AI 分割标签图(.npz)，shape 匹配才恢复到 volume_mask。"""
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Exported_Lesions", f"{self._safe_name(pid)}_mask.npz")
        if not os.path.exists(fp):
            return False
        try:
            m = np.load(fp)['mask']
            if m.shape == self.volume_hu.shape:
                self.volume_mask = m.astype(np.uint8)
                return True
        except Exception as e:
            print(f"Warning: failed to load saved mask: {e}")
        return False

    def save_project(self):
        """将当前所有标注保存为 JSON 文件（以 PatientID 命名），方便下次加载时自动恢复。
        JSON 键必须为字符串（JSON 规范），整数切片索引在此序列化为字符串，加载时再转回 int。
        """
        if not self.dicom_datasets:
            return
        pid = self._safe_name(str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown')))
        ed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions")
        os.makedirs(ed, exist_ok=True)
        try:
            with open(os.path.join(ed, f"{pid}_annotations.json"), 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.global_annotations.items()}, f, indent=4)
            # 一并保存 AI 多器官分割标签图，下次加载可直接恢复，免掉 ~100s 重算
            if self.volume_mask is not None and np.any(self.volume_mask):
                np.savez_compressed(os.path.join(ed, f"{pid}_mask.npz"), mask=self.volume_mask)
            QMessageBox.information(self, "Success", "Project Saved.")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Failed to save project:\n{e}")

    # =========================================================================
    # 器官定量 / 图例
    # =========================================================================
    def _compute_organ_stats(self):
        """器官定量薄包装：读取 self 的体积/蒙版/spacing 后调用纯函数
        quantify.compute_organ_stats（无 Qt，可独立单测）。返回按体积降序的列表。"""
        if self.volume_mask is None or self.volume_hu is None or not self.dicom_datasets:
            return []
        ds = self.dicom_datasets[0]
        spacing = (self._dcm_float(ds, 'PixelSpacing', 1.0, idx=0),
                   self._dcm_float(ds, 'PixelSpacing', 1.0, idx=1),
                   self._dcm_float(ds, 'SliceThickness', 1.0))
        return quantify.compute_organ_stats(self.volume_hu, self.volume_mask, spacing, self.organ_names)

    def _update_organ_stats(self):
        """刷新器官定量面板；无分割结果时清空并禁用导出按钮。"""
        self._organ_stats = self._compute_organ_stats()
        self._refresh_paint_target()   # 无论有无器官都刷新画笔目标下拉
        if not self._organ_stats:
            self.lbl_ai_stats.setText("")
            self.btn_export_stats.setEnabled(False)
            return
        e = self.is_english
        lines = []
        for r in self._organ_stats:
            r_, g_, b_ = (int(LABEL_LUT[r['id']][0]), int(LABEL_LUT[r['id']][1]), int(LABEL_LUT[r['id']][2]))
            nm = r['name_en'] if e else r['name_zh']
            lines.append(f'<span style="color:#{r_:02X}{g_:02X}{b_:02X};">■</span> '
                         f"{nm}: {r['volume_ml']:.1f} mL / {r['mean_hu']:.0f} HU")
        self.lbl_ai_stats.setText("<br>".join(lines))
        self.btn_export_stats.setEnabled(True)

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
        pid = "ANON" if self.anonymize else str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown'))
        ed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions")
        os.makedirs(ed, exist_ok=True)
        fp = os.path.join(ed, f"{self._safe_name(pid)}_organ_stats.csv")
        try:
            with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['# AI-derived, for reference only, NOT for diagnosis; organ labels are auto-inferred /'
                            ' AI 自动推断，仅供参考，非诊断依据'])
                w.writerow(['class_id', 'organ_zh', 'organ_en', 'voxels', 'volume_mL', 'mean_HU'])
                for r in rows:
                    w.writerow([r['id'], r['name_zh'], r['name_en'], r['voxels'],
                                f"{r['volume_ml']:.2f}", f"{r['mean_hu']:.1f}"])
            QMessageBox.information(self, "Success", f"Saved: {os.path.basename(fp)}")
        except Exception as ex:
            QMessageBox.warning(self, "Export Failed", str(ex))

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
                        area = vals.size * sp[0] * sp[1]  # mm²（sp=行/列像素间距）
                        stat = (f"{vals.mean():.0f}±{vals.std():.0f} HU\n"
                                f"[{vals.min():.0f}, {vals.max():.0f}]\n{area:.0f} mm²")
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
