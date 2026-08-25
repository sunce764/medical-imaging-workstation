# =============================================================================
# 研究四（推理基准）：分割模型在本机的推理成本，以及一条被否掉的加速捷径
# ---------------------------------------------------------------------------
# 两个问题，一次量清楚：
#   1. 推理时间由什么决定？（模型 FLOPs × 输入体素数，与训练数据量无关）
#   2. Mac 上换 CoreML provider 能不能白捡加速？——**实测答案是不能**
#
# 【为什么要把一个否定结果写进产物】
#   `onnxruntime` 在 macOS 上确实提供 CoreMLExecutionProvider，而项目各处写死了
#   CPUExecutionProvider——看上去像是一处遗漏的优化。实测下来 CoreML 反而更慢
#   （本机 5.682 vs 5.237 s/块，慢 8.5%，见 results/seg3d_bench.csv）：
#   3D 卷积在 Apple Neural Engine 上支持有限，大部分算子回退 CPU，反而多出调度与
#   数据传输开销。把它记下来，后来的人（包括我自己）不必再试一遍。
#
# 【判据必须同时看速度与正确性】
#   本脚本初版只比对了两种 provider 的输出一致性，然后打印「加速可用」——
#   而当时 CoreML 明明更慢。判据不完整会让结论反向，这里显式地两项都判。
#
# 用法：
#   python experiments/seg3d_bench.py                    # 教师模型（organs.onnx）
#   python experiments/seg3d_bench.py --model <x.onnx>   # 换模型（学生模型同样适用）
# 产出：results/seg3d_bench.csv + timestamped machine-readable provenance JSON
# =============================================================================

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from performance_artifact import (
    artifact_timestamp_slug,
    build_performance_artifact,
    process_peak_memory_gib,  # noqa: E402
    write_performance_artifact,
)

from constants import MODEL_PATH  # noqa: E402

RESULTS = os.path.join(HERE, "results")
# 沿 z 的分块高度：与 ai_engine 的滑窗一致（模型有 5 次下采样，须为 32 的倍数）
DZ = 32


def _session(model, providers):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False       # 与产品一致：压峰值内存，代价是略慢
    return ort.InferenceSession(model, sess_options=so, providers=providers)


def bench_providers(model, hw=(320, 320), n_rep=3):
    """同一块输入，比较 CPU 与 CoreML 的速度**和**输出一致性。

    两项都必须判：只比一致性会把「等价但更慢」误报成「加速可用」。
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    x = np.random.RandomState(0).rand(1, 1, DZ, *hw).astype(np.float32)
    rows, outs = [], {}
    for prov in (["CPUExecutionProvider"],
                 ["CoreMLExecutionProvider", "CPUExecutionProvider"]):
        tag = prov[0].replace("ExecutionProvider", "")
        if prov[0] not in avail:
            print(f"  {tag:<8} 本机不可用，跳过"); continue
        try:
            sess = _session(model, prov)
            iname = sess.get_inputs()[0].name
            sess.run(None, {iname: x})                 # 预热，排除首次建图开销
            ts = []
            for _ in range(n_rep):
                t = time.perf_counter(); o = sess.run(None, {iname: x})[0]
                ts.append(time.perf_counter() - t)
            outs[tag] = o
            actual = sess.get_providers()[0].replace("ExecutionProvider", "")
            rows.append(dict(provider=tag, actual=actual, sec_mean=round(float(np.mean(ts)), 3),
                             sec_std=round(float(np.std(ts)), 3)))
            print(f"  {tag:<8} {np.mean(ts):6.2f} ± {np.std(ts):.2f} s/块   实际 {actual}")
        except Exception as ex:
            print(f"  {tag:<8} 失败: {type(ex).__name__}: {str(ex)[:70]}")
    if len(outs) == 2:
        a, b = outs["CPU"], outs["CoreML"]
        agree = float((a[0].argmax(0) == b[0].argmax(0)).mean()) * 100
        sc, sm = (next(r['sec_mean'] for r in rows if r['provider'] == p) for p in ("CPU", "CoreML"))
        faster = sm < sc * 0.95           # 至少快 5% 才算有意义的加速
        print(f"\n  标签一致率 {agree:.3f}%   CoreML/CPU 耗时比 {sm/sc:.2f}")
        # 判据两项都要过：等价 **且** 更快，才谈得上「可用的加速」
        if agree > 99.9 and faster:
            print("  → CoreML 等价且更快，值得启用")
        elif agree > 99.9:
            print(f"  → CoreML 结果等价但**慢 {(sm/sc-1)*100:.0f}%**，不值得启用（本机实测）")
        else:
            print("  → CoreML 结果不等价，不能用")
    return rows


def bench_sizes(model, sizes=((32, 160, 160), (32, 224, 224), (32, 320, 320), (32, 384, 384)),
                n_rep=3):
    """推理时间随输入体素数如何变化——回答「训练数据量影响推理速度吗」这个常见误解。

    训练数据量只决定训练时长；推理时间由模型 FLOPs 与输入体素数决定，与前者无关。
    """
    sess = _session(model, ["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rows = []
    print(f"  {'输入':<20}{'体素数':>12}{'秒/块':>9}{'μs/体素':>11}")
    for shp in sizes:
        x = np.random.RandomState(0).rand(1, 1, *shp).astype(np.float32)
        sess.run(None, {iname: x})
        ts = []
        for _ in range(n_rep):
            t = time.perf_counter(); sess.run(None, {iname: x}); ts.append(time.perf_counter() - t)
        vox = int(np.prod(shp))
        s = float(np.mean(ts))
        rows.append(dict(shape="x".join(map(str, shp)), voxels=vox,
                         sec_mean=round(s, 3), us_per_voxel=round(s / vox * 1e6, 4)))
        print(f"  {str(shp):<20}{vox:>12,}{s:>9.2f}{s/vox*1e6:>11.4f}")
    # 过原点拟合的优度用「相对残差」评估，不用 np.corrcoef——后者度量的是含截距的
    # 线性相关，对过原点模型并不对应：一条明显不过原点的直线也能给出 r≈1。
    v = np.array([r['voxels'] for r in rows], float)
    t = np.array([r['sec_mean'] for r in rows], float)
    # 过原点的线性拟合与相关系数：接近 1 说明「时间 ∝ 体素数」成立
    k = float((v * t).sum() / (v * v).sum())
    resid = float(np.max(np.abs(t - k * v) / t))      # 过原点模型的最大相对残差
    r = float(np.corrcoef(v, t)[0, 1])                # 保留供参考，判据不用它
    print(f"\n  线性拟合 t ≈ {k*1e6:.4f} μs × 体素数   最大相对残差 {resid*100:.1f}%"
          f"（参考 r={r:.4f}）")
    print(f"  → 推理时间与输入体素数{'近似线性' if resid < 0.10 else '偏离线性'}；"
          f"与训练集大小无关")
    return rows, k, resid


def main():
    run_started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL_PATH)
    a = ap.parse_args()
    if not os.path.exists(a.model):
        print(f"  缺少模型 {a.model}"); return 1
    import onnx
    m = onnx.load(a.model, load_external_data=False)
    npar = sum(int(np.prod(list(i.dims))) for i in m.graph.initializer if i.dims)
    print(f"  模型 {os.path.basename(a.model)}：{len(m.graph.node)} 算子，"
          f"权重元素 {npar/1e6:.1f}M\n")

    provider_hw = (320, 320)
    sizes = ((32, 160, 160), (32, 224, 224), (32, 320, 320), (32, 384, 384))
    n_rep = 3
    print("  === Provider 对比（单块 32×320×320）===")
    prov_rows = bench_providers(a.model, hw=provider_hw, n_rep=n_rep)
    print("\n  === 推理时间 vs 输入体素数 ===")
    size_rows, k, resid = bench_sizes(a.model, sizes=sizes, n_rep=n_rep)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "seg3d_bench.csv"), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value", "unit"])
        w.writerow(["model", "params_million", round(npar / 1e6, 2), "M"])
        w.writerow(["model", "onnx_nodes", len(m.graph.node), "count"])
        # std 必须落盘：n_rep 只有 3 次，没有离散度就无法判断 provider 间的差异
        # 是否超出测量噪声——「慢 8.5%」若落在 ±5% 的抖动里，那个结论就不成立。
        for r in prov_rows:
            w.writerow(["provider", r['provider'], r['sec_mean'], "s/block"])
            w.writerow(["provider", r['provider'] + "_std", r['sec_std'], "s/block"])
            w.writerow(["provider", r['provider'] + "_n_rep", n_rep, "count"])
        for r in size_rows:
            w.writerow(["size", r['shape'], r['sec_mean'], "s/block"])
        w.writerow(["fit", "us_per_voxel", round(k * 1e6, 4), "us"])
        w.writerow(["fit", "max_rel_residual", round(resid, 4), "ratio"])

    peak_gib, peak_method = process_peak_memory_gib()
    # `--model` may point at a custom ONNX whose external-data blob is not named
    # `<graph>.data`; bind every location declared by the graph, not a filename guess.
    external_locations = {
        entry.value
        for tensor in m.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    model_files = [a.model] + [
        os.path.join(os.path.dirname(a.model), location)
        for location in sorted(external_locations)
    ]
    artifact = build_performance_artifact(
        script="experiments/seg3d_bench.py",
        mode="provider-and-input-size-benchmark",
        project_root=os.path.dirname(HERE),
        model_files=model_files,
        configuration={
            "provider_input_shape": [1, 1, DZ, *provider_hw],
            "size_inputs": [list(shape) for shape in sizes],
            "n_rep": n_rep,
            "warmup_runs_per_configuration": 1,
            "random_seed": 0,
            "cpu_mem_arena_enabled": False,
        },
        wall_time_seconds=time.perf_counter() - run_started,
        peak_memory_gib=peak_gib,
        peak_memory_method=peak_method,
        dependency_packages=("numpy", "onnx", "onnxruntime"),
        extra_measurements={
            "provider_results": prov_rows,
            "size_results": size_rows,
            "fit_us_per_voxel": k * 1e6,
            "fit_max_relative_residual": resid,
        },
    )
    artifact_name = f"seg3d_bench_{artifact_timestamp_slug(artifact)}.json"
    write_performance_artifact(os.path.join(RESULTS, artifact_name), artifact)
    print(f"    → results/seg3d_bench.csv / results/{artifact_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
