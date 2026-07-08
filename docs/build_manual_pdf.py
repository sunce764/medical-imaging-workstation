"""从 docs/manual_zh.md 生成提交规范的《软件说明书》PDF（docs/manual_zh.pdf）。

用途：软件著作权登记 / 用户手册。PDF 内容取自 manual_zh.md，本脚本只负责排版
      （封面 + 目录 + 每节独立起页 + 图片），不增删任何功能描述。

依赖：
  - Python 库 markdown（`pip install markdown`）
  - Google Chrome（无头模式打印 PDF；支持中文系统字体）

用法：  python docs/build_manual_pdf.py
"""
import os
import re
import shutil
import subprocess
import sys

import markdown

DOCS = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(DOCS, "manual_zh.md")
BUILD_HTML = os.path.join(DOCS, "_manual_build.html")
PDF = os.path.join(DOCS, "manual_zh.pdf")

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
* { box-sizing: border-box; }
body { font-family: "PingFang SC","Songti SC","Heiti SC",serif; font-size: 13px;
       line-height: 1.85; color: #1a1a1a; }
.cover { min-height: 23cm; display: flex; flex-direction: column; justify-content: center;
         text-align: center; break-after: page; }
.cover .t { font-size: 26px; font-weight: 700; margin-bottom: 10px; }
.cover .s { font-size: 15px; color: #444; margin: 4px 0; }
.cover .meta { margin-top: 40px; font-size: 14px; color: #333; line-height: 2.2; }
.cover .note { margin-top: 48px; font-size: 11.5px; color: #666; padding: 12px 24px;
               border: 1px solid #ddd; background: #f8f8f8; text-align: left; }
.toc { break-after: page; }
.toc h2 { border: none; text-align: center; }
.toc-h2 { font-size: 14px; font-weight: 600; margin: 10px 0 3px; }
.toc-h3 { font-size: 12.5px; color: #444; margin: 2px 0 2px 1.5em; }
h2 { font-size: 17px; margin: 0 0 10px; padding-bottom: 6px; border-bottom: 2px solid #333;
     break-before: page; break-after: avoid; }
h3 { font-size: 14px; margin: 16px 0 6px; break-after: avoid; }
p { margin: 7px 0; }
blockquote { color: #555; border-left: 3px solid #bbb; margin: 8px 0; padding: 3px 12px;
             background: #f7f7f7; font-size: 12px; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px;
       font-family: "SF Mono",Menlo,monospace; }
pre { background: #f5f5f5; border: 1px solid #ddd; border-radius: 5px; padding: 8px 12px;
      break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 10px auto 3px; border: 1px solid #ccc;
      break-inside: avoid; }
em { color: #555; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12px;
        break-inside: avoid; }
th,td { border: 1px solid #bbb; padding: 5px 9px; text-align: left; }
th { background: #eee; }
hr { display: none; }
"""

COVER = """
<div class="cover">
  <div class="t">医学影像工作站软件</div>
  <div class="s">Medical Imaging Workstation Software</div>
  <div class="s">软件说明书</div>
  <div class="meta">软件版本：V1.0<br>著作权人：盛超、赖胜圣<br>编写日期：2026 年 7 月</div>
  <div class="note">本说明书文中截图均使用<b>公开数据集 TotalSegmentator-CT-Lite（CC-BY-4.0）</b>演示，
  <b>非患者数据、不含个人隐私信息（PHI）</b>，患者信息栏已明确标注为公开数据。</div>
</div>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("未找到 Google Chrome / Chromium，无法生成 PDF。")


def main() -> None:
    raw = open(MD, encoding="utf-8").read()

    toc = [("h2", ln[3:].strip()) if ln.startswith("## ")
           else ("h3", ln[4:].strip())
           for ln in raw.splitlines() if ln.startswith(("## ", "### "))]
    toc_html = "".join(f'<div class="toc-{lv}">{t}</div>' for lv, t in toc)

    body_md = re.sub(r'^# .+\n', '', raw, count=1)                       # 首个 H1 移入封面
    body_md = re.sub(r'\((img/[^)]+)\)',
                     lambda m: f'(file://{os.path.join(DOCS, m.group(1))})', body_md)  # 图片绝对路径
    body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])

    html = (f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><style>{CSS}</style>'
            f'</head><body>{COVER}<div class="toc"><h2>目　录</h2>{toc_html}</div>{body}</body></html>')
    open(BUILD_HTML, "w", encoding="utf-8").write(html)

    chrome = find_chrome()
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files",
                    f"--print-to-pdf={PDF}", f"file://{BUILD_HTML}"], check=True)
    os.remove(BUILD_HTML)
    print(f"已生成 {PDF}")


if __name__ == "__main__":
    main()
