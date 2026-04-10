#!/usr/bin/env python3
"""
debug_chinese_font.py
不依赖任何已有修复，从零找到一个能显示中文的字体，强制设置，生成测试图。

用法：
  python debug_chinese_font.py
输出：
  debug_font_report.txt  — 完整报告
  debug_test_*.png       — 每个候选字体的测试图
"""
import os, sys, glob, subprocess
from pathlib import Path

# ── Step 1: 收集所有字体文件 ─────────────────────────────────────────────────
print("Step 1: 扫描字体文件...")
search_roots = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    os.path.join(sys.prefix, "fonts"),
    os.path.join(sys.prefix, "lib", "fonts"),
    # matplotlib 自带字体
]
import matplotlib
mpl_font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
search_roots.append(mpl_font_dir)

all_fonts = []
for root in search_roots:
    if os.path.isdir(root):
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            all_fonts.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))

print(f"  共找到 {len(all_fonts)} 个字体文件")

# ── Step 2: 筛选可能含 CJK 的字体 ────────────────────────────────────────────
cjk_kw = ["cjk", "chinese", "noto", "simhei", "simsun", "wqy", "wenquanyi",
           "droid", "arphic", "hanzi", "pingfang", "mingliu", "dengxian",
           "adobe", "source han", "思源", "黑体", "宋体"]
cjk_fonts = [f for f in all_fonts
             if any(k in f.lower() for k in cjk_kw)]
print(f"  CJK 候选字体: {len(cjk_fonts)} 个")
for f in cjk_fonts:
    print(f"    {f}")

# ── Step 3: 如果没有候选，用 fc-list 查 ──────────────────────────────────────
if not cjk_fonts:
    print("\n  系统字体目录里没找到，尝试 fc-list...")
    try:
        out = subprocess.check_output(
            ["fc-list", ":lang=zh", "--format=%{file}\n"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if out:
            cjk_fonts = [l for l in out.splitlines() if l.strip()]
            print(f"  fc-list 找到 {len(cjk_fonts)} 个:")
            for f in cjk_fonts:
                print(f"    {f}")
    except Exception as e:
        print(f"  fc-list 失败: {e}")

# ── Step 4: 逐个测试能不能真正渲染中文 ───────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

TEST_TEXT = "中文测试 低幅度 坐姿抬手"
report_lines = []
working = []

print(f"\nStep 4: 逐个测试渲染（最多测 10 个）...")
for font_path in cjk_fonts[:10]:
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        fname = prop.get_name()

        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.text(0.5, 0.5, TEST_TEXT, fontproperties=prop,
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        safe = Path(font_path).stem.replace(" ", "_")[:30]
        out_png = f"debug_test_{safe}.png"
        plt.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close()

        # 检查图片里是否出现方框（通过检测像素颜色变化粗略判断）
        try:
            import numpy as np
            from PIL import Image
            img = np.array(Image.open(out_png).convert("L"))
            # 如果全是接近白色（方框显示），像素方差很小
            variance = float(img.var())
            has_content = variance > 50
        except ImportError:
            has_content = None  # PIL 不可用，跳过自动检测

        status = "✓ 有内容" if has_content else ("? 需人工确认" if has_content is None else "✗ 疑似方框")
        msg = f"  {status}  {fname}  ({font_path})"
        print(msg)
        report_lines.append(msg)
        if has_content:
            working.append((fname, font_path))

    except Exception as e:
        msg = f"  ✗ 加载失败  {font_path}: {e}"
        print(msg)
        report_lines.append(msg)

# ── Step 5: 找到能用的字体，写入 matplotlibrc ─────────────────────────────────
print()
if working:
    best_name, best_path = working[0]
    print(f"Step 5: 使用字体 [{best_name}]，写入配置...")

    # 运行时生效
    matplotlib.rcParams["font.sans-serif"] = [best_name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 持久化
    config_dir = Path(matplotlib.get_configdir())
    rc_path = config_dir / "matplotlibrc"
    existing = rc_path.read_text().splitlines() if rc_path.exists() else []
    existing = [l for l in existing if not l.strip().startswith(("font.", "axes.unicode"))]
    existing += [
        f"font.family      : sans-serif",
        f"font.sans-serif  : {best_name}, DejaVu Sans",
        f"axes.unicode_minus : False",
    ]
    rc_path.write_text("\n".join(existing) + "\n")
    print(f"  写入: {rc_path}")

    # 清缓存
    for f in Path(matplotlib.get_cachedir()).glob("fontlist-*.json"):
        f.unlink()
    fm._load_fontmanager(try_read_cache=False)
    print("  字体缓存已清除")

    # 最终测试图
    prop = fm.FontProperties(fname=best_path)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([1,2,3,4], [0.8, 0.3, 0.9, 0.5], "o-", color="steelblue")
    ax.set_title("中文显示修复验证 — 坐姿抬手 低幅度区段", fontproperties=prop, fontsize=14)
    ax.set_xlabel("时间（秒）", fontproperties=prop, fontsize=11)
    ax.set_ylabel("幅度", fontproperties=prop, fontsize=11)
    ax.text(2.5, 0.6, f"字体：{best_name}", fontproperties=prop, fontsize=10, color="green")
    plt.tight_layout()
    plt.savefig("debug_final_test.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n最终测试图: debug_final_test.png")
    print(f"如果这张图里中文正常，后续在脚本开头加入：")
    print(f"  import matplotlib; matplotlib.rcParams['font.sans-serif'] = ['{best_name}', 'DejaVu Sans']")
    print(f"  matplotlib.rcParams['axes.unicode_minus'] = False")

else:
    print("Step 5: 没有找到可用的 CJK 字体。")
    print()
    print("解决方案（选一个执行）：")
    print()
    print("  方案 A — 下载字体（无需 root）：")
    print("    mkdir -p ~/.local/share/fonts")
    print("    wget -O ~/.local/share/fonts/NotoSansSC.ttf \\")
    print("      'https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf'")
    print("    python debug_chinese_font.py   # 再跑一次")
    print()
    print("  方案 B — conda 安装（推荐）：")
    print("    conda install -c conda-forge mplfonts -y")
    print("    python -c \"import mplfonts; mplfonts.init()\"")
    print("    python debug_chinese_font.py")
    print()
    print("  方案 C — apt 安装（需 sudo）：")
    print("    sudo apt-get install -y fonts-noto-cjk")
    print("    python debug_chinese_font.py")

# 写报告
with open("debug_font_report.txt", "w") as f:
    f.write(f"matplotlib version: {matplotlib.__version__}\n")
    f.write(f"sys.prefix: {sys.prefix}\n\n")
    f.write(f"All font files ({len(all_fonts)}):\n")
    for ff in all_fonts:
        f.write(f"  {ff}\n")
    f.write(f"\nCJK candidates ({len(cjk_fonts)}):\n")
    for ff in cjk_fonts:
        f.write(f"  {ff}\n")
    f.write(f"\nTest results:\n")
    for l in report_lines:
        f.write(f"  {l}\n")
print(f"\n完整报告: debug_font_report.txt")
