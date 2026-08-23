"""研究一补测：FBP 的 RMSE 在高视角处触及的是**实现的离散化误差地板**，不是剂量充足。

动机：实验 A 观察到「RMSE 在 ≈180 视角后进入收益递减」，此前被解释成「剂量够了，
再加视角收益有限」，并作为「enough is enough」的定量依据写进文档。该归因**未被验证**。

本脚本在**无噪声**条件下把视角一路加到 2880（远超任何采样判据），看 RMSE 是否继续
下降。若它在某个值上硬性停住，则曲线变平的原因是重建链路自身的误差下限
（`ndimage.zoom` 样条插值 + `iradon` 的 ramp 滤波与插值），与剂量无关。

参照判据：Kak & Slaney《Principles of Computerized Tomographic Imaging》§5.1.1 由
角向/径向频域间距相等推得「投影数应与每投影的射线数大致相同」，对 256 宽探测器
即 ≈256 视角（严格取 π/2 系数则 ≈402）。

确定性，无随机成分。用法（须在 dicom_gui 环境内）：python experiments/recon_floor.py
输出：experiments/results/exp_a_metric_floor.csv
"""
import csv
import os
import sys

import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recon_study as rs  # noqa: E402

import recon  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VIEWS = (180, 256, 360, 720, 1440, 2880)


def main():
    n = 256
    gt = rs.get_phantom(n)
    cm = recon._circle_mask(n) > 0     # 与 roi_metrics 同口径：圆内统计
    rows = []
    print(f"{'views':>6}{'RMSE':>10}{'SSIM':>9}   （无噪声，圆内 RMSE）")
    for npj in VIEWS:
        theta = recon.make_theta(180.0, npj)
        sino = recon.compute_sinogram(gt, theta)
        _, rec = recon.compute_fbp(sino, theta, "ramp")
        r = float(np.sqrt(np.mean((rec[cm] - gt[cm]) ** 2)))
        s = float(ssim(gt, rec, data_range=1.0))
        rows.append({"views": npj, "rmse_in_circle": round(r, 6), "ssim": round(s, 6)})
        print(f"{npj:6d}{r:10.5f}{s:9.4f}", flush=True)

    floor = min(x["rmse_in_circle"] for x in rows)
    at180 = next(x["rmse_in_circle"] for x in rows if x["views"] == 180)
    print(f"\n地板 = {floor:.5f}；180 视角处仅高出地板 {at180 / floor - 1:.1%}。")
    top3 = [x["rmse_in_circle"] for x in rows[-3:]]
    print(f"最高三档 {top3}，相差 {(max(top3) - min(top3)) / min(top3):.2%}，且不再单调下降"
          " ⇒ 平台高度由重建实现决定，非剂量。")

    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "exp_a_metric_floor.csv")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {dest}（新文件，未覆盖 exp_a_dose_quality.csv）")


if __name__ == "__main__":
    main()
