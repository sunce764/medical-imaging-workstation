# =============================================================================
# 重建实验室 Mixin
# 负责：CT 断层重建教学实验室的全部 UI 调度（投影生成 / BP / FBP / DFR /
#       DMR / ART / SIRT / ASD-POCS，重建模式进出、视图刷新）。
#
# 设计：以 Mixin 形式并入 MedicalViewer（class MedicalViewer(QMainWindow,
#       ReconLabMixin)）。这些方法通过 self 访问主窗口的 UI 控件与状态
#       （self.views / self.slider_* / self.btn_* / self.volume_hu 等），
#       以及留在 main.py 的共享方法（self.set_view_title / self.update_display /
#       self._apply_grid_visibility / self._apply_grid_sizes）。
#       纯数值计算仍在 recon.py（recon_lib），本模块只做 UI 接线与调度。
# =============================================================================

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

import recon as recon_lib
from constants import RECON_DL_MODEL, RECON_DL_VIEWS


class ReconLabMixin:
    """重建实验室相关方法集合，混入 MedicalViewer。"""

    # =========================================================================
    # 重建模式进出与视图刷新
    # =========================================================================
    def _enter_recon_mode(self):
        """进入重建实验室：记忆原布局、清空视图、切到 2x2、隐藏每视图工具栏控件。"""
        self._cancel_view_interactions()
        self._pre_recon_layout = self.combo_layout.currentIndex()
        self._recon_ref_z = None   # 强制下一次 _render_recon_reference 初始化重建流水线
        for vid in range(1, 5):
            v = self.views[vid]['view']
            v.image_item.setPixmap(QPixmap())
            v.mask_item.setPixmap(QPixmap())
            v.set_overlay({}, {})  # 清空 DICOM 叠加，避免临床态的患者信息/方位字母残留到重建图上
            v.resetTransform()
        # 切到 2x2，setSizes 在 setUpdatesEnabled(False) 下同步生效
        self._apply_grid_visibility(2)
        self._apply_grid_sizes(2)
        # 标题必须看模体状态：空载时 update_display 会因 volume_hu 为 None 直接返回，
        # 走不到 _render_recon_reference 那条修正路径，于是切 Tab 往返一次后 V1 就
        # 挂着「真实切片」显示着模体（实测踩到）。
        if self._phantom_img is not None:
            self.set_view_title(1, "V1 [Phantom · known truth]" if self.is_english
                                else "V1 [模体 · 真值已知]")
            self.display_numpy_image(1, self._phantom_img)
        else:
            self.set_view_title(1, "V1 [Ground Truth]" if self.is_english else "V1 [真实切片]")
        self._set_recon_pending_titles()
        for v in self.views.values():
            v['cb_plane'].hide(); v['preset'].hide(); v['chk_anno'].hide()
        self._sync_view_controls()
        self.update_display()

    def _exit_recon_mode(self):
        """退出重建实验室：清空弦图缓存与按钮、恢复每视图工具栏控件、还原原布局。"""
        self._invalidate_recon_results()
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp, self.btn_dl]:
            b.setEnabled(False)
        for vid, v in self.views.items():
            v['view'].image_item.setPixmap(QPixmap())
            v['view'].mask_item.setPixmap(QPixmap())
            v['view'].resetTransform()
            v['view'].setRenderHint(QPainter.SmoothPixmapTransform, True)
            v['cb_plane'].show(); v['preset'].show(); v['chk_anno'].show()
            self.set_view_title(vid, f"V{vid}")
        prev = self._pre_recon_layout
        self._apply_grid_visibility(prev)
        self._apply_grid_sizes(prev)
        self._sync_view_controls()
        self.update_display()

    def _set_recon_pending_titles(self):
        """将 V2/V3/V4 标题统一设置为"请先生成弦图"的等待提示。
        进入重建实验室、切换切片导致旧弦图失效时调用，集中处理避免散落多份字符串。
        """
        txt = "[— run projection —]" if self.is_english else "[— 请先生成弦图 —]"
        for vid in (2, 3, 4):
            self.set_view_title(vid, f"V{vid} {txt}")

    def _invalidate_recon_results(self):
        """数据来源变更时统一作废弦图及派生结果，不能只拿层号判断是否换数据。"""
        self._recon_ref_z = None
        self.current_sinogram = self.current_theta = self._last_recon_img = None
        self._cached_bp = self._cached_bp_sino = None
        for b in (self.btn_dfr, self.btn_bp, self.btn_fbp, self.btn_dl):
            b.setEnabled(False)
        if self.recon_mode_active:
            for vid in (2, 3, 4):
                view = self.views[vid]['view']
                view.set_image(QPixmap())
                view.clear_annotations()
                view.set_overlay({}, {})
            self._set_recon_pending_titles()

    def _render_recon_reference(self, z):
        """重建实验室分支：仅刷新 V1 的"真实切片"参考图，并重置 V2-V4 重建流水线状态。"""
        # 模体在场时 V1 就该是模体：否则切一次层，V1 变成真实切片而弦图仍来自模体，
        # 两者对不上却毫无提示。模体不随切片变化，也就不必重置重建流水线。
        if self._phantom_img is not None:
            self.display_numpy_image(1, self._phantom_img)
            self.set_view_title(1, "V1 [Phantom · known truth]" if self.is_english
                                else "V1 [模体 · 真值已知]")
            return
        img_gt = self._display_intensity(self.volume_hu[z])
        ww, wl = self.slider_ww.value(), self.slider_wl.value()
        # 与阅片页共用显示强度，未知单位的预览不变更底层体积。
        img_windowed = np.clip(img_gt, wl - ww / 2, wl + ww / 2)
        img_windowed = ((img_windowed - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        img_windowed = np.ascontiguousarray(img_windowed)
        h, w = img_windowed.shape
        qimg = QImage(img_windowed.data, w, h, w, QImage.Format_Grayscale8).copy()
        # V1 回到源轴位切片，不能沿用进入重建前冠/矢状面的 (z, in-plane) 间距。
        ds = self.dicom_datasets[z]
        spacing = (tuple(self._dcm_float(ds, 'PixelSpacing', 1.0, idx=i) for i in (0, 1))
                   if self.inplane_spacing_valid else (1.0, 1.0))
        self.views[1]['view'].set_image(QPixmap.fromImage(qimg), pixel_spacing=spacing)
        self.views[1]['view'].clear_annotations()
        self.set_view_title(1, "V1 [Ground Truth]" if self.is_english else "V1 [真实切片]")
        # 仅当切片真正改变时才重置重建流水线；调窗等其他刷新不应清掉已生成的弦图/重建结果
        if self._recon_ref_z != z:
            self._invalidate_recon_results()
            self._recon_ref_z = z
            for vid in [2, 3, 4]:
                self.views[vid]['view'].image_item.setPixmap(QPixmap())
            self.current_sinogram = None
            self._last_recon_img = None   # 换切片清链式源图，下次"生成弦图"从新层原图开始
            self._cached_bp = None; self._cached_bp_sino = None
            for b in [self.btn_dfr, self.btn_bp, self.btn_fbp, self.btn_dl]:
                b.setEnabled(False)
            self._set_recon_pending_titles()

    # =========================================================================
    # UI 参数读取
    # =========================================================================
    def _get_n_angles(self):
        """读取当前 UI 选中的投影角度数（60/120/180/360），180° 为默认值。"""
        if self.rad_60.isChecked():    return 60
        if self.rad_120.isChecked():   return 120
        if self.rad_360.isChecked():   return 360
        return 180

    def _get_angle_oversample(self):
        """读取采样密度倍率（1/2/4）。在同一角度范围内，倍率越高投影越密、重建质量越好。"""
        return [1, 2, 4][max(0, self.combo_oversample.currentIndex())]

    # =========================================================================
    # 数值图像显示
    # =========================================================================
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

    # =========================================================================
    # 投影生成 + 解析重建（BP / FBP / DFR）
    # =========================================================================
    PHANTOM_N = 256   # 模体分辨率：解析生成故可任选，256 兼顾 Radon 速度与细节

    def _sync_matrix_buttons(self):
        """按「是否存在重建源图」同步 DMR / 迭代重建按钮的可用性。

        判据必须与 _recon_source_slice 的前提一致：模体在场，或已载入 DICOM 体数据。
        三处调用点（载入数据、重置、模体开关）共用本方法，避免各自维护一份判据。
        """
        has_src = (getattr(self, '_phantom_img', None) is not None
                   or getattr(self, 'volume_hu', None) is not None)
        for b in (self.btn_gen_sino, self.btn_dmr, self.btn_art):
            b.setEnabled(has_src)

    def _recon_source_slice(self):
        """返回重建链路的源图（float32，归一化 [0,1]）与来源标签。

        内置模体优先于真实切片：重建实验室原先必须先导入 DICOM 才能用，空载启动时
        整个实验室是死的。模体让它自成闭环——且模体是**已知真值**，误差图这一栏
        才真正有意义（对真实切片，"真值"其实只是原图，本身也含噪声与重建痕迹）。

        无源可用时返回 (None, None)，由各调用方自行 return。
        """
        if self._phantom_img is not None:
            return self._phantom_img, ("Phantom" if self.is_english else "模体")
        if not self.dicom_datasets or self.volume_hu is None:
            return None, None
        img = self.volume_hu[self.current_3d_pos[0]]
        denom = img.max() - img.min()
        if not np.isfinite(denom) or denom <= 0:
            # 均匀切片（空白定位像、全空气层）：归一化没有意义，重建出来必是全黑，
            # 而耗时栏照样报成功。直接判为不可重建，由调用方给提示。
            return None, None
        norm = (img - img.min()) / denom
        norm = norm.astype(np.float32)
        # 与 recon.prepare_small_image 同一处理：radon(circle=True) 只作用于内切圆，
        # 圆外像素根本不进弦图。不掩膜的话 V1 展示的「真值」含有重建永远拿不到的
        # 内容，误差图便把这部分算成算法误差——实测圆内 FBP 峰值差达显示范围的 26.9%，
        # 而 skimage 的警告只进 stderr，界面上看不到。
        norm = norm * recon_lib._circle_mask(norm.shape[0]) if norm.shape[0] == norm.shape[1] \
            else norm
        return norm, ("Origin" if self.is_english else "原图")

    def toggle_phantom(self):
        """载入/卸下内置 Shepp-Logan 模体。

        卸下时把重建链路的中间状态一并清掉：弦图与上一次重建结果都是基于模体算的，
        留着会让「模体的弦图」配上「真实数据的原图」，是最容易骗过自己的那类不一致。
        """
        on = self._phantom_img is None
        self._phantom_img = recon_lib.shepp_logan(self.PHANTOM_N) if on else None
        self._invalidate_recon_results()
        self.current_sinogram = None
        self.current_theta = None
        self._last_recon_img = None
        # BP 缓存同样作废。它用 `is` 比对弦图对象，换源后本就不会命中，但另两处
        # 重置点（退出重建模式、换切片）都清了它，这里不清就成了唯一的例外，
        # 且白白占着一张 256² 中间结果。
        self._cached_bp = None; self._cached_bp_sino = None
        for vid in (2, 3, 4):
            v = self.views[vid]['view']
            v.image_item.setPixmap(QPixmap()); v.mask_item.setPixmap(QPixmap())
        for b in (self.btn_bp, self.btn_fbp, self.btn_dfr, self.btn_dl):
            b.setEnabled(False)
        # 【模体也是合法源图】DMR/迭代重建的源由 _recon_source_slice 决定，它**优先**
        # 取内置模体、其次才是 DICOM 切片。但这两个按钮的可用性此前只看 volume_hu，
        # 于是空载载入模体后源图明明可用、按钮却仍是灰的——「无需导入数据即可使用
        # 完整重建实验室」这句公开说明在真实 UI 上不成立。既有测试直接调方法，
        # 绕过了按钮可达性，因此一直是绿的。
        self._sync_matrix_buttons()
        e = self.is_english
        self.btn_phantom.setText(("Unload Phantom" if on else "Load Shepp-Logan Phantom") if e
                                 else ("卸下模体" if on else "载入 Shepp-Logan 模体"))
        if on:
            self.display_numpy_image(1, self._phantom_img)
            # 标题必须同步改掉：进入重建实验室时 V1 被标为「真实切片」，模体在场时
            # 仍挂着那个标题等于界面在说谎（截图时发现——只读代码看不出来）
            self.set_view_title(1, "V1 [Phantom · known truth]" if e else "V1 [模体 · 真值已知]")
            self.lbl_time.setText("Phantom loaded (known ground truth)" if e
                                  else "已载入模体（真值已知）")
        else:
            self.lbl_time.setText("")
            self.set_view_title(1, "V1 [Ground Truth]" if e else "V1 [真实切片]")
            if self.volume_hu is None:
                self.views[1]['view'].set_image(QPixmap())
                self.views[1]['view'].clear_annotations()
            self.update_display()

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
        # 来源选择：有重建结果时对重建图做 Radon，用完清空（下次回到原图/模体）
        if self._last_recon_img is not None:
            img_src = self._last_recon_img
            src_label = "重建图" if not self.is_english else "Recon"
            self._last_recon_img = None   # 消费后清空，下次按钮回到源图路径
        else:
            img_src, src_label = self._recon_source_slice()
            if img_src is None:
                QMessageBox.information(self, "No source" if self.is_english else "无可重建源图",
                                        "Load a phantom or select a non-uniform image slice."
                                        if self.is_english else "请载入模体，或选择具有灰度变化的图像切片。")
                return
        ar = self._get_n_angles()
        self.current_theta = recon_lib.make_theta(ar, ar * self._get_angle_oversample())

        start_t = time.perf_counter()
        self.current_sinogram = recon_lib.compute_sinogram(img_src, self.current_theta)
        elapsed = (time.perf_counter() - start_t) * 1000
        self.lbl_time.setText(f"Radon [{src_label}]: {elapsed:.1f} ms")
        self.display_numpy_image(2, self.current_sinogram.T)
        self.set_view_title(2, f"V2 [Sinogram - {src_label}]")
        for b in [self.btn_dfr, self.btn_bp, self.btn_fbp]:
            b.setEnabled(True)
        # DL 按钮额外要求模型与 onnxruntime 就绪——缺任一就一直禁用，不给用户
        # 一个点了报错的按钮
        self.btn_dl.setEnabled(bool(getattr(self, '_dl_model_ready', False)))
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

    def run_dl_recon(self):
        """学习式后处理重建：把 ramp-FBP 的条纹伪影交给 CNN 去除（研究三产物）。

        V3 显示作为输入的 ramp-FBP，V4 显示网络输出，便于直接对比"改了什么"。

        三处刻意的诚实处理，都写在界面上而非只写在文档里：
          1) 输入**强制**用 ramp 而非下拉框里选的滤波器——模型以 ramp 为输入训练，
             喂 hann 会偏，而这种偏差在图上看不出来，只会让结果悄悄变差；
          2) 当前视角数与模型训练视角不一致时，标题明确标注"视角不匹配"——
             模型只在 20 视角下训练过，用在别处效果打折，不能装作通用；
          3) 缺模型或缺 onnxruntime 时按钮本就禁用，不会走到这里。
        """
        if self.current_sinogram is None or not getattr(self, '_dl_model_ready', False):
            return
        self._fit_recon_views(smooth=True)
        # 强制 ramp：模型的训练输入就是 ramp-FBP
        _, fbp_in = recon_lib.compute_fbp(self.current_sinogram, self.current_theta, 'ramp')
        try:
            out, ms = recon_lib.compute_dl_recon(fbp_in, RECON_DL_MODEL)
        except Exception as ex:                       # 推理失败要说出来，不能静默留着旧图
            QMessageBox.warning(self, "DL Recon Failed" if self.is_english else "深度学习重建失败",
                                str(ex))
            return
        n_now = len(self.current_theta) if self.current_theta is not None else 0
        mism = n_now != RECON_DL_VIEWS
        e = self.is_english
        self.lbl_time.setText(f"DL recon time: {ms:.1f} ms" if e else f"深度学习重建耗时: {ms:.1f} ms")
        self.display_numpy_image(3, fbp_in)
        self.set_view_title(3, "V3 [FBP ramp — network input]" if e else "V3 [FBP ramp — 网络输入]")
        self.display_numpy_image(4, out)
        tag = ((f" ⚠ view mismatch: model trained at {RECON_DL_VIEWS}, now {n_now}" if e
                else f" ⚠ 视角不匹配：模型训练于 {RECON_DL_VIEWS} 视角，当前 {n_now}") if mism else "")
        self.set_view_title(4, ("V4 [CNN post-processing]" if e else "V4 [CNN 后处理重建]") + tag)

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
        # V4 显示：重建图像（朝向已在 compute_dfr 内校正，直接 abs 即可）
        self.display_numpy_image(4, np.abs(recon_dfr))
        self.set_view_title(4, "V4 [Direct Fourier DFR]" if self.is_english else "V4 [直接傅里叶重建 DFR]")

    # =========================================================================
    # 矩阵重建准备 + 系统矩阵构建 + 视图刷新
    # =========================================================================
    def _prepare_small_image_and_sinogram(self):
        """为 DMR/ART 准备小尺寸图像及其弦图。

        步骤：
          1. 取当前切片的 HU 值，归一化到 [0, 1]
          2. 用双三次插值（ndimage.zoom）缩小到 n×n（n 由 UI 下拉框选择：16/32/64）
          3. 施加圆形掩码（radon(circle=True) 只假定圆外为零并发警告，不代为置零；
             掩码是为了满足该前提并与 iradon 的输出支撑对齐）
             关键：若不施加此掩码，V1（原图）角落有值而 V4（重建）角落为0，
             误差图会在角落显示虚假的大误差，迷惑用户误判算法质量
          4. 对小图做 Radon 变换生成弦图，投影角由 UI 的角度范围(60/120/180/360°)
             与采样密度(1×/2×/4×)共同决定：投影数 = 角度范围 × 密度倍率

        返回：(img_small, sinogram, theta, n)
        """
        img_norm, _ = self._recon_source_slice()
        if img_norm is None:
            return None, None, None, None
        n = int(self.cb_matrix_size.currentText().split('×')[0])
        ar = self._get_n_angles()
        img_small, sinogram, theta = recon_lib.prepare_small_image(img_norm, n, ar, ar * self._get_angle_oversample())
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

        try:
            A, key = recon_lib.build_system_matrix(
                n, theta, self._cached_A, self._cached_A_key, progress_cb=_progress
            )
        except Exception as e:
            prog.close()   # 关键：异常时也要关模态框，否则 UI 卡死
            QMessageBox.warning(self, "Matrix Build Failed" if self.is_english else "系统矩阵构建失败", str(e))
            return None
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
        # 不再检查 volume_hu：源图可以是内置模体，_prepare 内部统一判空
        img_small, sinogram, theta, n = self._prepare_small_image_and_sinogram()
        if img_small is None:
            return
        A = self._build_system_matrix(n, theta)
        if A is None:
            return   # 系统矩阵构建失败已提示，安全退出
        p_vec = sinogram.flatten().astype(np.float32)
        # lstsq 同步阻塞主线程，用忙碌对话框提示，避免 UI 看起来像卡死
        pd = QProgressDialog("Solving A·x=p (lstsq)..." if self.is_english else "正在求解 A·x=p (最小二乘)...",
                             None, 0, 0, self)
        pd.setWindowModality(Qt.WindowModal); pd.show(); QApplication.processEvents()
        img_recon, t_ms = recon_lib.compute_dmr(A, p_vec, n)
        pd.close()
        self._last_recon_img = img_recon   # 供"生成弦图"按钮对重建结果做正向投影
        error_map = np.abs(img_small - img_recon)

        self.display_numpy_image(1, recon_lib.upscale_recon(img_small, n))
        self.display_numpy_image(2, sinogram.T)
        self.display_numpy_image(3, recon_lib.upscale_recon(error_map, n))
        self.display_numpy_image(4, recon_lib.upscale_recon(img_recon, n))
        self._fit_recon_views(smooth=False)

        # RMSE 只在内切圆内统计，与 experiments/recon_study.roi_metrics 同口径。
        # 必要性：矩阵法（DMR/ART/SIRT）与解析法不同，圆外解并不为零
        # （iradon 显式置零，最小二乘不会），按全幅统计会把圆外那部分算成
        # 算法误差。改动理由是口径一致而非幅度：本实验室不加噪声，满秩档两种口径
        # 差别极小（实测 60 视角全幅 0.0149 / 圆内 0.0154，90 视角两者均≈0），
        # 但真值本身已被掩到圆内，误差就该在同一支撑上统计。
        _cm = recon_lib._circle_mask(n) > 0
        rmse = float(np.sqrt(np.mean(error_map[_cm] ** 2)))
        self.set_view_title(1, f"V1 [Orig {n}x{n}]" if self.is_english else f"V1 [原始 {n}x{n}]")
        self.set_view_title(2, "V2 [Sinogram]" if self.is_english else "V2 [投影弦图]")
        self.set_view_title(3, f"V3 [Error RMSE={rmse:.4f}]")
        self.set_view_title(4, f"V4 [DMR {n}x{n}]")
        self.lbl_time.setText(f"DMR lstsq: {t_ms:.1f} ms" if self.is_english else f"直接矩阵重建耗时: {t_ms:.1f} ms")

    # =========================================================================
    # ART / SIRT / ASD-POCS 迭代重建
    # =========================================================================
    # 各方法的迭代档位与默认值。**别合并成一张通用表**——见 ui_builder 里的说明：
    # ASD-POCS 每轮 = 一次 ART 扫掠 + n_grad 次 TV 最速下降，收敛慢一个量级，
    # 用 ART 的 10/20/50 去跑它会得到比 FBP 还差的结果（按本实验室默认路径实测：
    # 32×32、180°×1×、无噪，10/20 轮 RMSE 0.1460/0.1336 对 FBP 的 0.0995）。
    ITER_OPTIONS = {
        'ART':      (["10", "20", "50"], 1),
        'SIRT':     (["10", "20", "50"], 1),
        # 50 是实测第一个胜过 FBP 的档（0.0672 对 FBP 0.0995）。耗时与轮数成正比，
        # 默认 32×32 下约 8.8 ms/轮（300 轮 ≈2.7 s），64×64 下约 37 ms/轮——
        # 绝对秒数随机器与负载变化，此处只作量级参考。
        'ASD-POCS': (["50", "100", "150", "300"], 2),
    }

    def _sync_art_iter_options(self, method):
        """把迭代次数下拉框换成当前方法适用的档位，尽量保留用户已选的值。"""
        items, default_idx = self.ITER_OPTIONS.get(method, self.ITER_OPTIONS['ART'])
        prev = self.cb_art_iter.currentText()
        self.cb_art_iter.blockSignals(True)
        self.cb_art_iter.clear()
        self.cb_art_iter.addItems(items)
        self.cb_art_iter.setCurrentIndex(items.index(prev) if prev in items else default_idx)
        self.cb_art_iter.blockSignals(False)

    def run_art_sirt(self):
        """ART / SIRT / ASD-POCS 迭代重建——通过逐步修正逼近方程组的解。

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

        ASD-POCS（TV 正则化的压缩感知重建，Sidky & Pan 2008）：
          每轮 = 一次带松弛的 ART 扫掠（投影到数据一致集与非负集）+ n_grad 次
          全变差最速下降。相比前两者多了一个先验：真实断层图像的梯度是稀疏的。
          特点：稀疏角/低剂量下明显优于 ART/SIRT，但收敛慢一个量级——迭代次数
          档位因此与前两者不同（见 ITER_OPTIONS），用 ART 的轮数跑它会更差。

        三种方法共同特点：
          - x = clip(x, 0) 每轮强制非负约束（HU值不存在负像素强度）
          - 支持中途取消（wasCanceled），取消后显示当前迭代的中间结果
          - 视图分配与 DMR 相同：V1=原图, V2=弦图, V3=误差, V4=重建
        """
        # 不再检查 volume_hu：源图可以是内置模体，_prepare 内部统一判空
        img_small, sinogram, theta, n = self._prepare_small_image_and_sinogram()
        if img_small is None:
            return
        A = self._build_system_matrix(n, theta)
        if A is None:
            return   # 系统矩阵构建失败已提示，安全退出
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

        # done[0] 记实际完成的迭代数：两个求解器都在 cancel 时 break，但只返回
        # (图, 耗时)，调用方无从得知跑了几轮。中途取消后若照 n_iter 报告，标题与
        # 耗时会同时失真（实测 3 轮的结果被标成 50it，RMSE 差 17 倍、耗时差 20 倍），
        # 而这正是本实验室要教的对比量。progress_cb 每完成一轮调用一次，故 last+1。
        done = [0]

        def _progress(it):
            done[0] = it + 1
            prog_iter.setValue(it + 1)
            QApplication.processEvents()

        solver = {'ART': recon_lib.compute_art,
                  'SIRT': recon_lib.compute_sirt,
                  'ASD-POCS': recon_lib.compute_asdpocs}.get(method, recon_lib.compute_sirt)
        img_recon, t_ms = solver(
            A, p_vec, n, n_iter, cancel_check=_cancel, progress_cb=_progress)

        prog_iter.close()
        n_done = done[0]                   # 实际完成轮数；未取消时等于 n_iter

        self._last_recon_img = img_recon   # 供"生成弦图"按钮对重建结果做正向投影
        error_map = np.abs(img_small - img_recon)
        self.display_numpy_image(1, recon_lib.upscale_recon(img_small, n))
        self.display_numpy_image(2, sinogram.T)
        self.display_numpy_image(3, recon_lib.upscale_recon(error_map, n))
        self.display_numpy_image(4, recon_lib.upscale_recon(img_recon, n))
        self._fit_recon_views(smooth=False)
        # RMSE 只在内切圆内统计，与 experiments/recon_study.roi_metrics 同口径。
        # 必要性：矩阵法（DMR/ART/SIRT）与解析法不同，圆外解并不为零
        # （iradon 显式置零，最小二乘不会），按全幅统计会把圆外那部分算成
        # 算法误差。改动理由是口径一致而非幅度：本实验室不加噪声，满秩档两种口径
        # 差别极小（实测 60 视角全幅 0.0149 / 圆内 0.0154，90 视角两者均≈0），
        # 但真值本身已被掩到圆内，误差就该在同一支撑上统计。
        _cm = recon_lib._circle_mask(n) > 0
        rmse = float(np.sqrt(np.mean(error_map[_cm] ** 2)))
        self.set_view_title(1, f"V1 [Orig {n}x{n}]" if self.is_english else f"V1 [原始 {n}x{n}]")
        self.set_view_title(2, "V2 [Sinogram]" if self.is_english else "V2 [投影弦图]")
        self.set_view_title(3, f"V3 [Error RMSE={rmse:.4f}]")
        _cancelled = n_done < n_iter
        _tag = f"{n_done}/{n_iter}it" if _cancelled else f"{n_iter}it"
        self.set_view_title(4, f"V4 [{method} {_tag} {n}x{n}]"
                            + (" (cancelled)" if _cancelled and self.is_english else
                               "（已取消）" if _cancelled else ""))
        _sfx = ((f"  ← cancelled, not the {n_iter}-iteration result" if self.is_english
                 else f"  ← 已取消，非 {n_iter} 次迭代的结果") if _cancelled else "")
        self.lbl_time.setText((f"{method} ({_tag}): {t_ms:.1f} ms" if self.is_english
                               else f"{method} ({_tag} 迭代)耗时: {t_ms:.1f} ms") + _sfx)
