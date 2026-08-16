# =============================================================================
# 研究四（数据管线）：为 3D 器官分割自训练准备带真值的多例公开数据
# ---------------------------------------------------------------------------
# 数据源：TotalSegmentator-CT-Lite（HuggingFace，CC BY-4.0，1228 例带真值标注）。
#        与 seg_validate.py 同一数据集，但这里取多例并做患者级划分。
#
# 【相对 seg_validate.py 修正的一处复现性缺陷】
#   那个脚本用 /resolve/main 抓数据——main 是【可变分支引用】，上游一更新就再也拿不到
#   当初那份，而当时也没记校验和。本脚本改用**固定 commit**，并把每个文件的 SHA256
#   写进清单，任何人都能验证自己拿到的是否逐字节相同。
#
# 【为什么必须用带真值的数据，而不是教师模型的输出】
#   用 TotalSegmentator 的预测当训练标签是可以的（知识蒸馏），但**验证集绝不能用它**：
#   那样测出的 Dice 是「学生模仿教师有多像」，不是「分割有多准」。教师自身相对真值
#   的 Dice 也不是 1.0，误差会被静默吞掉，最后报出一个看着漂亮但没有意义的数字。
#
# 用法：
#   python experiments/seg3d_data.py fetch [N]     # 抽 N 例（默认 100）到本地缓存
#   python experiments/seg3d_data.py verify        # 按清单校验本地文件完整性
#   python experiments/seg3d_data.py split         # 患者级划分并打印统计
#
# 数据落在 experiments/.seg3d_cache/（已 gitignore，不入库）。
# =============================================================================

import hashlib
import json
import os
import sys
import time

REPO = "YongchengYAO/TotalSegmentator-CT-Lite"
# 固定 commit：上游更新不影响本实验的可复现性
REVISION = "6f14b84ecf8ad7592fd7ad06c57c26d34ee61067"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".seg3d_cache")
MANIFEST = os.path.join(HERE, "results", "seg3d_manifest.json")

# 划分比例与种子。患者级划分——同一例的任何切片都不会跨集出现。
SPLIT_SEED = 0
SPLIT = {'train': 0.70, 'val': 0.10, 'test': 0.20}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _list_cases(n):
    """取前 n 个病例 ID。按名称排序后截取——确定性选取，换台机器拿到的是同一批。"""
    from remotezip import RemoteZip
    with RemoteZip(f"{BASE}/Masks.zip") as zm:
        return sorted(x[len("Masks/"):-len(".nii.gz")]
                      for x in zm.namelist() if x.endswith(".nii.gz"))[:n]


def fetch(n=100, batch=8, retries=4):
    """按 HTTP Range 从远端 zip 里只抽需要的那 n 例，不下载 22GB 整包。

    【为什么要分批重连】HuggingFace 的 CDN 直链是**预签名 URL**（内含 Expires），
    一条 RemoteZip 连接开太久必然遇到签名过期或 SSL EOF——实测一口气拉 100 例
    在第 11 例上就断了。故每 batch 例重开一次连接，并对整批做有限重试。
    已下载的文件会被跳过，所以中断后重跑即为断点续传。
    """
    from remotezip import RemoteZip
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    t0 = time.perf_counter()
    cases = _list_cases(n)
    print(f"  从 {REPO}@{REVISION[:8]} 抽取 {len(cases)} 例（每 {batch} 例重开连接）")
    done, failed = [], []
    for s in range(0, len(cases), batch):
        chunk = cases[s:s + batch]
        # 「存在」不等于「完好」：加 .part 原子改名之前的那次中断留下过 0 字节残骸，
        # 只判 exists 会把它当成已下载而跳过，直到 nibabel 读取时才炸（实测：s0013_msk）。
        todo = [c for c in chunk if not all(
            os.path.exists(q) and os.path.getsize(q) > 0
            for q in (os.path.join(CACHE, f"{c}_img.nii.gz"),
                      os.path.join(CACHE, f"{c}_msk.nii.gz")))]
        for attempt in range(1, retries + 1):
            if not todo:
                break
            try:
                with RemoteZip(f"{BASE}/Masks.zip") as zm, RemoteZip(f"{BASE}/Images.zip") as zi:
                    for cid in list(todo):
                        # 先写临时文件再改名：中途断连不会留下半截文件冒充「已下载」
                        for sub, z, suf in (("Images", zi, "img"), ("Masks", zm, "msk")):
                            dst = os.path.join(CACHE, f"{cid}_{suf}.nii.gz")
                            tmp = dst + ".part"
                            with open(tmp, 'wb') as f:
                                f.write(z.read(f"{sub}/{cid}.nii.gz"))
                            os.replace(tmp, dst)
                        todo.remove(cid)
            except Exception as ex:
                if attempt == retries:
                    failed.extend(todo)
                    print(f"    ✗ {todo} 重试 {retries} 次仍失败: {type(ex).__name__}")
                else:
                    time.sleep(2 * attempt)
        done.extend([c for c in chunk if c not in failed])
        mb = sum(os.path.getsize(os.path.join(CACHE, f))
                 for f in os.listdir(CACHE) if f.endswith('.nii.gz')) / 1e6
        print(f"    {min(s+batch, len(cases))}/{len(cases)}  累计 {mb:.0f} MB  "
              f"{time.perf_counter()-t0:.0f}s"); sys.stdout.flush()

    rows = []
    for cid in done:
        pi = os.path.join(CACHE, f"{cid}_img.nii.gz")
        pm = os.path.join(CACHE, f"{cid}_msk.nii.gz")
        if os.path.exists(pi) and os.path.exists(pm):
            rows.append(dict(case=cid, img_sha256=_sha256(pi), msk_sha256=_sha256(pm),
                             img_bytes=os.path.getsize(pi), msk_bytes=os.path.getsize(pm)))
    with open(MANIFEST, 'w') as f:
        json.dump({'repo': REPO, 'revision': REVISION, 'n': len(rows), 'cases': rows}, f, indent=1)
    print(f"  成功 {len(rows)} 例，失败 {len(failed)} 例"
          + (f"：{failed}" if failed else "") + "；清单已写入 results/seg3d_manifest.json")
    return rows


def verify(deep=True):
    """校验本地缓存。

    【为什么不能只比 SHA256】清单本身就是由这些文件生成的，自己验自己必然通过——
    一个 0 字节的残骸也有 SHA256，照样写进清单、照样"一致"。实测正是如此：
    校验报告"完好 100"，而 s0013_msk 是空文件（清单里也老老实实记着 0 字节），
    直到 nibabel 读取时才炸。
    故 deep=True 时额外验证：文件非空 + 能被 nibabel 解析 + 影像与标注形状一致。
    """
    if not os.path.exists(MANIFEST):
        print("  无清单，请先 fetch"); return False
    man = json.load(open(MANIFEST))
    missing, mismatch, corrupt = [], [], []
    for r in man['cases']:
        pi = os.path.join(CACHE, f"{r['case']}_img.nii.gz")
        pm = os.path.join(CACHE, f"{r['case']}_msk.nii.gz")
        if not (os.path.exists(pi) and os.path.exists(pm)):
            missing.append(r['case']); continue
        if _sha256(pi) != r['img_sha256'] or _sha256(pm) != r['msk_sha256']:
            mismatch.append(r['case']); continue
        if deep:
            try:
                if os.path.getsize(pi) == 0 or os.path.getsize(pm) == 0:
                    raise ValueError("空文件")
                import nibabel as nib
                si, sm = nib.load(pi).shape, nib.load(pm).shape
                if si != sm:
                    raise ValueError(f"影像 {si} 与标注 {sm} 形状不一致")
            except Exception as ex:
                corrupt.append((r['case'], str(ex)[:40]))
    n_ok = man['n'] - len(missing) - len(mismatch) - len(corrupt)
    print(f"  清单 {man['n']} 例（{man['repo']}@{man['revision'][:8]}）")
    print(f"  缺失 {len(missing)}  校验和不符 {len(mismatch)}  内容损坏 {len(corrupt)}  完好 {n_ok}")
    for tag, lst in (("缺失", missing), ("不符", mismatch), ("损坏", corrupt)):
        if lst:
            print(f"    ✗ {tag}: {lst[:5]}")
    return not (missing or mismatch or corrupt)


def split():
    """患者级划分。返回 {'train': [...], 'val': [...], 'test': [...]}。

    按病例 ID 划分而非按切片——同一例的相邻切片高度相关，按切片划分会让
    训练集与测试集共享同一个解剖结构，Dice 虚高而毫无意义。
    """
    import numpy as np
    man = json.load(open(MANIFEST))
    cases = sorted(r['case'] for r in man['cases'])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(cases))
    n_tr = int(len(cases) * SPLIT['train'])
    n_va = int(len(cases) * SPLIT['val'])
    out = {'train': [cases[i] for i in idx[:n_tr]],
           'val': [cases[i] for i in idx[n_tr:n_tr + n_va]],
           'test': [cases[i] for i in idx[n_tr + n_va:]]}
    # 划分自证：三集必须两两不交，且并集等于全集
    a, b, c = (set(out[k]) for k in ('train', 'val', 'test'))
    assert not (a & b) and not (a & c) and not (b & c), "划分有交叠"
    assert a | b | c == set(cases), "划分未覆盖全集"
    for k, v in out.items():
        print(f"  {k:>5}: {len(v):>3} 例   {v[:4]}{' …' if len(v) > 4 else ''}")
    print(f"  三集两两不交、并集=全集（seed={SPLIT_SEED}，患者级）")
    return out


def main():
    cmd = (sys.argv[1] if len(sys.argv) > 1 else 'fetch').lower()
    if cmd == 'fetch':
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        verify(); split()
    elif cmd == 'verify':
        return 0 if verify() else 1
    elif cmd == 'split':
        split()
    else:
        print(__doc__ or "用法见文件头"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
