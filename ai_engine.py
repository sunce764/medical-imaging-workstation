# =============================================================================
# AI 推理引擎模块
# 负责：异步后台 AI 多器官分割推理（organs.onnx，25 类）
# 设计：纯 Python daemon 线程，避免继承 QThread 的析构崩溃风险；
#       完成/进度经 Qt 信号（QueuedConnection）投递回主线程更新 UI
# =============================================================================

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import numpy as np
import shiboken6  # 随 PySide6 一同安装（PySide6_Essentials 硬依赖 shiboken6==6.11.0），非新增依赖
from PySide6.QtCore import QObject, Signal

import segmentation
from constants import MODEL_PATH


class _AISignals(QObject):
    """跨线程回调载体。在主线程创建，子线程 emit 时 Qt 以 QueuedConnection 自动投递到
    主线程事件循环——这是从 threading.Thread 安全更新 Qt UI 的正确方式。
    （不能用 QTimer.singleShot：它依附调用它的子线程，而子线程无 Qt 事件循环，回调不 fire。）"""
    finished = Signal(object, float)   # (label_map, elapsed_ms)
    progress = Signal(int, int)        # (done_slices, total_slices)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# 模块级 InferenceSession 缓存：模型加载（含 119MB 外部权重）有固定开销，
# 按 model_path 缓存复用，避免每次推理（尤其快速切换数据）重复加载。
# onnxruntime 的 session.run 是线程安全的，可被多个后台线程共享。
_SESSION_CACHE = {}


def _get_session(model_path: str):
    sess = _SESSION_CACHE.get(model_path)
    if sess is None:
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # 关内存池，显著降低分块推理的峰值内存
        sess = ort.InferenceSession(model_path, sess_options=so,
                                    providers=['CPUExecutionProvider'])
        _SESSION_CACHE[model_path] = sess
    return sess


class AutoAIEngineThread:
    """在后台线程中执行 AI 多器官分割推理，完成后通过 Qt 信号安全回调主线程。"""

    def __init__(self, volume_hu: np.ndarray,
                 callback: Callable[[np.ndarray, float], None],
                 model_path: str = MODEL_PATH,
                 progress_callback: Callable[[int, int], None] | None = None) -> None:
        # volume_hu: 完整的 3D HU 值体素数组，shape=(Z, H, W)，float32
        # callback: 推理完成后调用，签名为 callback(label_map, elapsed_ms)
        #           label_map 为 uint8 多类标签图（0=背景，1-24=器官类别）
        # model_path: ONNX 模型文件绝对路径，默认 constants.MODEL_PATH（models/organs.onnx）
        # progress_callback: 可选，签名 progress_callback(done_slices, total_slices)，
        #                    在滑窗推理过程中经 Qt 信号（QueuedConnection）投递到主线程，
        #                    用于更新进度显示
        self.volume_hu = volume_hu
        self.callback = callback
        self.model_path = model_path
        self.progress_callback = progress_callback
        self._thread = None
        # 信号对象在此（主线程）创建，其槽即在主线程执行；子线程 emit 自动排队投递。
        # 【刻意不设 parent，勿"优化"】实测（PySide6 6.11.0，15×15 对照）：不设 parent 时
        # 宿主 widget 销毁后信号源仍存活（isValid=True），emit 正常，0/15 异常；一旦设
        # parent=viewer，Qt 会在宿主析构时连带删除本对象，后台线程的 emit 必抛
        # RuntimeError('Signal source has been deleted') —— 15/15 必现，且把 Python 层
        # 异常升级为 C++ 跨线程 use-after-free。生存期已由后台线程栈帧持有本引擎保证。
        self._signals = _AISignals()
        self._signals.finished.connect(lambda m, t: self.callback(m, t))
        if progress_callback is not None:
            self._signals.progress.connect(lambda d, t: self.progress_callback(d, t))
        # 单次推理约 8.8GB 内存、~100s 且不可中途中断 onnxruntime.run。快速切换数据时
        # 若不作废旧线程，多个推理会并发叠加导致内存翻倍甚至 OOM。cancel() 置位后，
        # 滑窗循环在下一个 z 块边界提前退出并放弃回调，及时释放内存。
        self._cancelled = False

    def start(self) -> None:
        # daemon=True 确保主窗口关闭后不会因后台线程阻塞进程退出
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """作废本次推理：滑窗循环会在下一个 z 块边界停止，且不再触发完成回调。"""
        self._cancelled = True

    def isRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _emit_safe(self, signal, *args) -> bool:
        """向主线程投递信号；若信号源已被拆卸则返回 False（不抛异常）。

        QApplication 拆卸 / 进程退出时，_AISignals 的 C++ 侧可能已被销毁，此后任何
        emit 都抛 RuntimeError('Signal source has been deleted')。那是退出期竞态，
        不是推理失败——不加守卫的话它会被 _run 的 except 吞成「ONNX 推理失败」（日志说假话），
        或从 finished.emit 处直接冲出线程打印 traceback。
        投递机制仍是 Qt 信号（QueuedConnection），此处只加一道前置存活校验。
        """
        if not shiboken6.isValid(self._signals):
            return False
        try:
            signal.emit(*args)
            return True
        except RuntimeError:
            return False   # isValid 与 emit 之间宿主被拆卸的 TOCTOU 窗口，同样按拆卸处理

    def _run(self) -> None:
        """推理主体，运行在后台线程，严禁在此处操作任何 Qt 对象（非线程安全）。"""
        start_t = time.perf_counter()

        # HU 值归一化到 [0, 1]：肺窗范围 -1000~400 HU 覆盖空气到软组织
        # 超出范围的 HU 值（骨骼 > 400）通过 clip 截断，防止影响网络输入分布
        norm_vol = np.clip(self.volume_hu, -1000, 400)
        norm_vol = (norm_vol - (-1000)) / (400 - (-1000))
        norm_vol = norm_vol.astype(np.float32)

        final_mask = None

        # === 路径1：真实 ONNX 多器官分割推理 ===
        if HAS_ONNX and os.path.exists(self.model_path):
            try:
                final_mask = self._run_onnx_multiorgan(norm_vol)
            except RuntimeError as e:
                # Qt 生命周期竞态，不是模型问题：进度 emit 在 QApplication 拆卸后会抛
                # RuntimeError('Signal source has been deleted')。实测 onnxruntime 1.23.2 的
                # 14 个异常类型全部直接继承 Exception、无一继承 RuntimeError，故此分支
                # 不会误吞真正的推理失败。日志必须说实话：这不是「ONNX 推理失败」。
                print(f"AI 推理在应用拆卸期间中断（非模型故障）: {e}")
                self._cancelled = True   # 宿主已在拆卸，无需再做数学降级，让线程尽早退出
            except Exception as e:
                print(f"ONNX 推理失败，降级为数学算法: {e}")

        # 推理被作废（用户已切换到新数据）：直接退出，不回调、不做数学降级，释放内存
        if self._cancelled:
            return

        # === 路径2：纯数学算法降级（无模型文件或推理失败时自动启用）===
        # 逻辑已抽到 segmentation.segment_lungs_fallback（无 Qt，可独立单测）：
        # 阈值取低密度空气 → 3D 连通域 → 剔除体表边界相连的体外空气 → 剩余内部空气取
        # 最大(及≥其5%的次大)连通域为双肺。
        if final_mask is None:
            final_mask = segmentation.segment_lungs_fallback(self.volume_hu)

        if self._cancelled:
            return  # 数学降级期间也可能被作废，退出前再确认一次
        final_mask = final_mask.astype(np.uint8)  # 统一为 uint8 标签图，供调色板 LUT 索引
        end_t = time.perf_counter()
        # 经信号跨线程投递到主线程（QueuedConnection），安全更新 Qt UI。
        # 走 _emit_safe：拆卸期若信号源已销毁，此处原本会抛 RuntimeError 冲出 _run，
        # 由 threading 打印「Exception in thread」traceback（此 emit 不在任何 try 内）。
        self._emit_safe(self._signals.finished, final_mask, (end_t - start_t) * 1000)

    def _run_onnx_multiorgan(self, norm_vol: np.ndarray) -> np.ndarray | None:
        """ONNX 多器官分割推理，返回 uint8 标签图（0=背景，1-24=器官类别）。

        关键约束（均由对 organs.onnx 的实测确定）：
          - 输入 (1,1,D,H,W)，每个空间维必须 pad 到 32 的倍数；
          - 必须整幅 xy 送入，做中心裁剪会破坏全局上下文导致器官被误判为背景；
          - 输出 (1,25,D,H,W) 为 logits，取 argmax(axis=1) 得类别，而非阈值二值化；
          - 整卷一次推理输出约 6.7GB 会 OOM，故沿 z 分块滑窗（DZ=32）+ 关闭
            CPU 内存池，把峰值压到约 8.8GB。
        """
        Z, H, W = norm_vol.shape
        session = _get_session(self.model_path)  # 复用缓存的会话，避免重复加载模型
        input_name = session.get_inputs()[0].name

        ph, pw = (-H) % 32, (-W) % 32  # xy 方向对齐到 32 的倍数所需的填充
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        DZ = 32
        for z0 in range(0, Z, DZ):
            if self._cancelled:
                return None  # 已作废：停止推理，让 _run 放弃回调并释放内存
            z1 = min(z0 + DZ, Z)
            blk = norm_vol[z0:z1]
            pd = (-blk.shape[0]) % 32  # z 方向（尤其末块）对齐到 32
            if pd or ph or pw:
                blk = np.pad(blk, ((0, pd), (0, ph), (0, pw)), mode='constant')
            out = session.run(None, {input_name: blk[np.newaxis, np.newaxis].astype(np.float32)})[0][0]
            lab = out.argmax(0).astype(np.uint8)  # (D',H',W')
            seg[z0:z1] = lab[:z1 - z0, :H, :W]     # 裁掉 pad 部分写回原尺寸
            del out, lab
            if self.progress_callback is not None:
                # 经信号跨线程投递到主线程更新进度显示（子线程禁止直接操作 Qt）
                self._emit_safe(self._signals.progress, z1, Z)
        return seg
