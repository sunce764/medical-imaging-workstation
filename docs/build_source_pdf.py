"""生成软件著作权登记用《源代码》PDF（docs/source_code_zh.pdf）。

按软著规范：每页不少于 50 行；源代码总量超过 60 页时，提交前 30 页 + 后 30 页
（本软件应用代码约 4,100 行 ≈ 82 页，故取前 30 + 后 30，中间依规定略去）。
代码为软件真实源码，未删改语法，仅按每页 50 行分页并加页眉/页码。

依赖：Python 库 markdown 不需要；仅需 Google Chrome（无头打印）。
用法：  python docs/build_source_pdf.py
"""
import html
import os
import shutil
import subprocess
import sys

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
BUILD_HTML = os.path.join(DOCS, "_source_build.html")
PDF = os.path.join(DOCS, "source_code_zh.pdf")
TITLE = "医学影像工作站 Pro + 重建实验室 V1.0"

# 拼接顺序：入口 main.py 在前，UI 层 Mixin，随后引擎/视图，最后纯计算模块与常量。
ORDER = ["main.py", "ui_builder.py", "interaction.py", "recon_lab.py", "compare_lab.py",
         "annotation_lab.py", "ai_engine.py", "graphics_view.py", "recon.py",
         "quantify.py", "segmentation.py", "mpr_geometry.py", "constants.py"]

LINES_PER_PAGE = 50

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 1.2cm 1.2cm; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "SF Mono", Menlo, Consolas, monospace; }
.page { break-after: page; }
.page:last-child { break-after: auto; }
.hd { font-size: 9px; color: #555; border-bottom: 1px solid #ccc; padding-bottom: 3px;
      margin-bottom: 5px; display: flex; justify-content: space-between; }
pre { margin: 0; font-size: 8px; line-height: 1.2; white-space: pre-wrap;
      word-break: break-all; }
.note { text-align: center; font-family: "PingFang SC", serif; font-size: 12px; color: #333;
        border: 1px solid #bbb; background: #f7f7f7; padding: 14px; margin: 20px 0; }
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("未找到 Google Chrome / Chromium。")


def main() -> None:
    lines: list[str] = []
    for fn in ORDER:
        with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
            flines = f.read().splitlines()
        lines.append(f"# ========== {fn}  （{len(flines)} 行） ==========")
        lines.extend(flines)

    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    total = len(pages)

    def render(page_lines: list[str], pageno: int) -> str:
        code = html.escape("\n".join(page_lines))
        return (f'<div class="page"><div class="hd"><span>{TITLE}　·　源代码</span>'
                f'<span>第 {pageno} / {total} 页</span></div><pre>{code}</pre></div>')

    if total > 60:
        front = "".join(render(pages[i], i + 1) for i in range(30))
        note = (f'<div class="note">—— 依软件著作权登记规定，源代码共约 {total} 页，'
                f'此处提交前 30 页与后 30 页；中间第 31 – {total - 30} 页略去 ——</div>')
        back = "".join(render(pages[i], i + 1) for i in range(total - 30, total))
        content = front + note + back
        submitted = 60
    else:
        content = "".join(render(pages[i], i + 1) for i in range(total))
        submitted = total

    open(BUILD_HTML, "w", encoding="utf-8").write(
        f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><style>{CSS}</style>'
        f'</head><body>{content}</body></html>')
    chrome = find_chrome()
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files",
                    f"--print-to-pdf={PDF}", f"file://{BUILD_HTML}"], check=True)
    os.remove(BUILD_HTML)
    print(f"源代码总 {total} 页；已生成 {PDF}（提交 {submitted} 页）")


if __name__ == "__main__":
    main()
