#!/usr/bin/env python3
"""
apply_chinese_font.py
一次性修复：让 matplotlib 能同时正确渲染中文和 ASCII 字符（/、+、数字等）。

问题根源：
  DroidSansFallback = CJK-only 字体（没有 Latin glyph，/ 显示为方块）
  DejaVu Sans       = Latin-only 字体（没有 CJK glyph，汉字显示为方块）
  matplotlib Agg backend 的 per-glyph fallback 不可靠，靠两个字体拼凑必然失败。

解决方案：
  优先使用 WenQuanYi Micro Hei（同时包含 Latin 和 CJK），一个字体解决所有问题。
  如未安装则自动 apt-get install；如 apt 失败则回退到 DroidSansFallback（中文OK，/可能有方块）。
"""
import shutil
import subprocess
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

# ── 候选字体（按优先级，需要同时包含 Latin 和 CJK）────────────────────────────
COMPREHENSIVE_FONTS = [
    # WenQuanYi Micro Hei：最常见的 Linux CJK 字体，含完整 Latin
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "WenQuanYi Micro Hei"),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",   "WenQuanYi Zen Hei"),
    # Noto CJK
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf", "Noto Sans CJK SC"),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",        "Noto Sans CJK SC"),
]

# 退路：仅 CJK（中文OK，但 / + - 等 ASCII 可能有方块）
FALLBACK_FONT = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "Droid Sans Fallback",
)


def find_font():
    """返回 (path, name, is_comprehensive)。先找 Latin+CJK 综合字体，找不到返回 Droid。"""
    for path, name in COMPREHENSIVE_FONTS:
        if Path(path).exists():
            print(f"找到综合字体（Latin+CJK）: {name}\n  {path}")
            return path, name, True

    path, name = FALLBACK_FONT
    if Path(path).exists():
        print(f"找到 CJK 字体（仅 CJK，ASCII 可能有方块）: {name}\n  {path}")
        return path, name, False

    return None, None, False


def try_apt_install():
    """尝试 apt-get 安装 WenQuanYi（需要 sudo）。"""
    print("未找到综合字体，尝试 apt-get install fonts-wqy-microhei ...")
    try:
        ret = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "fonts-wqy-microhei"],
            capture_output=True, timeout=60
        )
        if ret.returncode == 0:
            print("  apt-get 安装成功，刷新系统字体缓存 ...")
            subprocess.run(["fc-cache", "-fv"], capture_output=True, timeout=30)
            return True
        else:
            print(f"  apt-get 失败（返回码 {ret.returncode}），跳过")
    except Exception as e:
        print(f"  apt-get 异常: {e}")
    return False


def install_to_matplotlib(src_path: str, font_name: str):
    """把字体文件复制进 matplotlib 的 ttf 目录（让字体缓存能找到它）。"""
    mpl_font_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    dst = mpl_font_dir / Path(src_path).name
    if not dst.exists():
        shutil.copy2(src_path, dst)
        print(f"已复制: {src_path}\n  → {dst}")
    else:
        print(f"字体已在 matplotlib 目录: {dst}")


def rebuild_cache():
    """清除 matplotlib 字体缓存，强制重建。"""
    cache_dir = Path(matplotlib.get_cachedir())
    n = 0
    for f in cache_dir.glob("fontlist-*.json"):
        f.unlink()
        n += 1
    fm._load_fontmanager(try_read_cache=False)
    print(f"字体缓存已重建（删除 {n} 个旧缓存）")


def write_matplotlibrc(font_name: str, is_comprehensive: bool):
    """写入 matplotlibrc，持久化字体配置。"""
    config_dir = Path(matplotlib.get_configdir())
    config_dir.mkdir(parents=True, exist_ok=True)
    rc_path = config_dir / "matplotlibrc"

    lines = rc_path.read_text().splitlines() if rc_path.exists() else []
    lines = [l for l in lines if not l.strip().startswith(("font.", "axes.unicode"))]

    if is_comprehensive:
        # 单一字体，同时包含 Latin 和 CJK，无需 fallback
        sans_serif = f"{font_name}, DejaVu Sans"
    else:
        # CJK-only 字体，把 DejaVu Sans 放第一（Latin），CJK 字体放第二（中文）
        # 注意：这依赖 matplotlib 的 per-glyph fallback，不一定对所有版本生效
        sans_serif = f"DejaVu Sans, {font_name}"
        print("  警告：当前字体仅含 CJK，/ + - 等 ASCII 字符可能仍显示为方块。")
        print("  建议运行: sudo apt-get install -y fonts-wqy-microhei  然后重跑本脚本。")

    lines += [
        f"font.family         : sans-serif",
        f"font.sans-serif     : {sans_serif}",
        f"axes.unicode_minus  : False",
    ]
    rc_path.write_text("\n".join(lines) + "\n")
    print(f"已写入 matplotlibrc: {rc_path}")
    print(f"  font.sans-serif = {sans_serif}")


def make_test_plot(font_name: str, is_comprehensive: bool):
    """生成包含中文和 ASCII 混合的测试图。"""
    if is_comprehensive:
        matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", font_name]
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot([1, 2, 3, 4], [0.8, 0.3, 0.9, 0.5], "o-", color="steelblue")
    ax.set_title("中文验证：坐姿抬手 / 慢走 / 站立坐下 / 低幅度区段", fontsize=13)
    ax.set_xlabel("时间（秒）", fontsize=11)
    ax.set_ylabel("幅度（mV）", fontsize=11)
    plt.tight_layout()
    out = "chinese_ok.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n测试图: {out}")
    print("请确认：中文和 / 符号都应正常显示，无方块。")


# ── 主流程 ────────────────────────────────────────────────────────────────────
print("=" * 55)
font_path, font_name, is_comprehensive = find_font()

if not is_comprehensive and font_path is None:
    # 两种字体都没有，尝试安装
    if try_apt_install():
        font_path, font_name, is_comprehensive = find_font()

if font_path is None:
    print("\n未找到任何可用的 CJK 字体。请手动运行：")
    print("  sudo apt-get install -y fonts-wqy-microhei")
    print("然后重跑本脚本。")
    sys.exit(1)

print("=" * 55)
install_to_matplotlib(font_path, font_name)
rebuild_cache()
write_matplotlibrc(font_name, is_comprehensive)
make_test_plot(font_name, is_comprehensive)
