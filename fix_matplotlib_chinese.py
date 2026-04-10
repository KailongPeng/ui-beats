#!/usr/bin/env python3
"""
fix_matplotlib_chinese.py
诊断并修复 matplotlib 在服务器上无法显示中文的问题。

运行方式：
  python fix_matplotlib_chinese.py          # 诊断 + 自动修复 + 生成测试图
  python fix_matplotlib_chinese.py --check  # 仅诊断，不修复
"""
import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# ── 1. 诊断 ─────────────────────────────────────────────────────────────────

def find_cjk_fonts() -> list[str]:
    """在系统 font manager 中搜索支持 CJK 的字体。"""
    cjk_keywords = ["cjk", "chinese", "noto", "simhei", "simsun", "wenquanyi",
                    "droid", "adobe", "arphic", "wqy", "hanzi", "pingfang",
                    "hiragino", "mingliu", "dengxian"]
    found = []
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        path_lower = f.fname.lower()
        if any(k in name_lower or k in path_lower for k in cjk_keywords):
            found.append((f.name, f.fname))
    return found


def diagnose():
    print("=" * 60)
    print("matplotlib 中文诊断报告")
    print("=" * 60)
    print(f"matplotlib version : {matplotlib.__version__}")
    print(f"backend            : {matplotlib.get_backend()}")
    print(f"config dir         : {matplotlib.get_configdir()}")
    print(f"cache dir          : {matplotlib.get_cachedir()}")
    print(f"matplotlibrc       : {matplotlib.matplotlib_fname()}")
    print()

    # 当前默认字体
    rc = matplotlib.rcParams
    print(f"font.family        : {rc['font.family']}")
    print(f"font.sans-serif    : {rc['font.sans-serif'][:5]}...")
    print()

    # 搜索 CJK 字体
    cjk = find_cjk_fonts()
    if cjk:
        print(f"找到 {len(cjk)} 个 CJK 相关字体：")
        for name, path in cjk[:10]:
            print(f"  {name:30s}  {path}")
        if len(cjk) > 10:
            print(f"  ... 共 {len(cjk)} 个")
    else:
        print("未找到任何 CJK 相关字体（需要下载或安装）")
    print()

    # 检查系统字体路径
    sys_font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
    print("系统字体目录：")
    for d in sys_font_dirs:
        exists = os.path.isdir(d)
        count  = len(list(Path(d).rglob("*.tt[fc]"))) if exists else 0
        print(f"  {'✓' if exists else '✗'} {d}  ({count} 个字体文件)" if exists
              else f"  ✗ {d}  (不存在)")
    print()
    return cjk


# ── 2. 修复策略 ──────────────────────────────────────────────────────────────

# 优先尝试下载这些字体（Google Fonts / GitHub，无需 root）
FONT_URLS = [
    (
        "NotoSansSC-Regular",
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    ),
    (
        "WenQuanYi-Micro-Hei",
        "https://github.com/anthonyrawlinsuom/fonts/raw/master/wqy-microhei.ttc",
    ),
]

def get_user_font_dir() -> Path:
    d = Path(os.path.expanduser("~/.local/share/fonts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def try_install_from_system() -> str | None:
    """尝试用包管理器安装（需要 sudo，可能失败）。"""
    candidates = [
        ["apt-get", "install", "-y", "fonts-noto-cjk"],
        ["apt-get", "install", "-y", "fonts-wqy-microhei"],
        ["yum",     "install", "-y", "wqy-microhei-fonts"],
    ]
    for cmd in candidates:
        print(f"  尝试: sudo {' '.join(cmd)}")
        try:
            ret = subprocess.run(["sudo"] + cmd, capture_output=True, timeout=30)
            if ret.returncode == 0:
                print(f"  ✓ 安装成功: {cmd[-1]}")
                return cmd[-1]
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    return None


def download_font(name: str, url: str, font_dir: Path) -> Path | None:
    ext    = ".otf" if url.endswith(".otf") else ".ttc" if url.endswith(".ttc") else ".ttf"
    target = font_dir / f"{name}{ext}"
    if target.exists():
        print(f"  已存在: {target}")
        return target
    print(f"  下载 {name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(target, "wb") as f:
            f.write(r.read())
        print(f"  ✓ 保存至: {target}")
        return target
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None


def find_local_cjk_file() -> Path | None:
    """
    在本机已有字体文件里找一个支持 CJK 的（不需要网络）。
    常见位置：/usr/share/fonts, ~/.fonts, conda 环境内等。
    """
    search_roots = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        # conda 环境
        os.path.join(sys.prefix, "fonts"),
        os.path.join(sys.prefix, "lib", "fonts"),
    ]
    cjk_names = ["noto", "cjk", "wqy", "simhei", "simsun", "chinese",
                 "wenquanyi", "droid", "arphic"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for p in Path(root).rglob("*.[to][tt][fc]"):
            if any(k in p.name.lower() for k in cjk_names):
                return p
    return None


def rebuild_font_cache():
    """删除 matplotlib 字体缓存，强制下次重建。"""
    cache_dir = Path(matplotlib.get_cachedir())
    for f in cache_dir.glob("fontlist-*.json"):
        f.unlink()
        print(f"  已删除缓存: {f}")
    # 重建
    fm._load_fontmanager(try_read_cache=False)
    print("  字体缓存已重建")


def write_matplotlibrc(font_name: str):
    """在用户 config 目录写入 matplotlibrc，设置默认中文字体。"""
    config_dir = Path(matplotlib.get_configdir())
    rc_path    = config_dir / "matplotlibrc"
    lines = []
    if rc_path.exists():
        lines = rc_path.read_text().splitlines()
        # 移除旧的 font 相关行
        lines = [l for l in lines if not l.strip().startswith("font.")]
    lines += [
        f"font.family      : sans-serif",
        f"font.sans-serif  : {font_name}, DejaVu Sans, Bitstream Vera Sans, Arial",
        f"axes.unicode_minus : False",
    ]
    rc_path.write_text("\n".join(lines) + "\n")
    print(f"  已写入 matplotlibrc: {rc_path}")


def apply_font_runtime(font_path: Path) -> str:
    """运行时直接注册字体，返回字体名（立即生效，无需重启）。"""
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    name = prop.get_name()
    matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print(f"  已运行时注册字体: {name}  ({font_path})")
    return name


# ── 3. 测试图 ────────────────────────────────────────────────────────────────

def make_test_plot(out: str = "chinese_test.png"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([1, 2, 3, 4], [0.8, 0.6, 0.9, 0.4], "o-", color="steelblue")
    ax.set_title("中文显示测试 — PN-QRS 信号质量", fontsize=14)
    ax.set_xlabel("时间（秒）", fontsize=11)
    ax.set_ylabel("幅度", fontsize=11)
    ax.text(2, 0.7, "坐姿抬手 / 慢走 / 站立坐下", fontsize=10, color="red")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n测试图已保存: {out}")
    print("如果图中汉字正常显示（不是方框），修复成功 ✓")
    print("如果仍然是方框，请把测试图发给我继续排查。")


# ── 4. 主流程 ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅诊断，不修复")
    ap.add_argument("--out",   default="chinese_test.png", help="测试图输出路径")
    args = ap.parse_args()

    cjk_fonts = diagnose()

    if args.check:
        return

    print("=" * 60)
    print("开始修复")
    print("=" * 60)

    font_path = None
    font_name = None

    # 策略 1：系统已有 CJK 字体，直接用
    if cjk_fonts:
        name, path = cjk_fonts[0]
        print(f"策略 1：使用已有字体 {name}  ({path})")
        font_path = Path(path)
        font_name = name

    # 策略 2：在本机字体文件里找（不需要网络）
    if font_path is None:
        print("策略 2：在本机字体文件里搜索 CJK 字体...")
        font_path = find_local_cjk_file()
        if font_path:
            print(f"  找到: {font_path}")

    # 策略 3：尝试系统包管理器安装
    if font_path is None:
        print("策略 3：尝试系统包管理器安装...")
        try_install_from_system()
        rebuild_font_cache()
        cjk_fonts = find_cjk_fonts()
        if cjk_fonts:
            font_name, fpath = cjk_fonts[0]
            font_path = Path(fpath)

    # 策略 4：从网络下载到用户目录
    if font_path is None:
        print("策略 4：从网络下载字体...")
        font_dir = get_user_font_dir()
        for name, url in FONT_URLS:
            font_path = download_font(name, url, font_dir)
            if font_path:
                break

    if font_path is None:
        print("\n✗ 所有策略均失败，请手动下载字体文件并告知路径。")
        print("  推荐：wget https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf")
        sys.exit(1)

    # 注册字体并更新配置
    if font_name is None:
        font_name = apply_font_runtime(font_path)
    else:
        apply_font_runtime(font_path)

    rebuild_font_cache()
    write_matplotlibrc(font_name)

    print(f"\n修复完成，使用字体: {font_name}")
    print("提示：在已运行的 Python 进程里需要重新 import matplotlib 才能生效。")

    # 生成测试图验证
    make_test_plot(args.out)


if __name__ == "__main__":
    main()
