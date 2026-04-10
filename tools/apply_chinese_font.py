#!/usr/bin/env python3
"""
apply_chinese_font.py
针对本机环境的一次性修复：
  DroidSansFallbackFull.ttf 已存在于 /usr/share/fonts/truetype/droid/
  把它复制进 matplotlib 自己的 ttf 目录，清除缓存，写入 matplotlibrc。
  之后所有脚本无需任何改动即可显示中文。
"""
import shutil, os
from pathlib import Path
import matplotlib
import matplotlib.font_manager as fm

FONT_SRC = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FONT_NAME = "Droid Sans Fallback"

# 1. 复制字体到 matplotlib 的 ttf 目录
mpl_font_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
dst = mpl_font_dir / "DroidSansFallbackFull.ttf"
if not dst.exists():
    shutil.copy2(FONT_SRC, dst)
    print(f"已复制: {FONT_SRC} → {dst}")
else:
    print(f"已存在: {dst}")

# 2. 清除字体缓存（强制重建）
cache_dir = Path(matplotlib.get_cachedir())
removed = 0
for f in cache_dir.glob("fontlist-*.json"):
    f.unlink()
    removed += 1
print(f"已删除 {removed} 个缓存文件: {cache_dir}")

# 3. 重建字体缓存
fm._load_fontmanager(try_read_cache=False)
print("字体缓存已重建")

# 4. 写入 matplotlibrc（持久化）
config_dir = Path(matplotlib.get_configdir())
config_dir.mkdir(parents=True, exist_ok=True)
rc_path = config_dir / "matplotlibrc"
lines = rc_path.read_text().splitlines() if rc_path.exists() else []
lines = [l for l in lines if not l.strip().startswith(("font.", "axes.unicode"))]
lines += [
    f"font.family         : sans-serif",
    # DejaVu Sans first: handles Latin/ASCII (e.g. / + - numbers)
    # Droid Sans Fallback second: handles CJK via matplotlib's per-glyph fallback
    f"font.sans-serif     : DejaVu Sans, {FONT_NAME}",
    f"axes.unicode_minus  : False",
]
rc_path.write_text("\n".join(lines) + "\n")
print(f"已写入 matplotlibrc: {rc_path}")

# 5. 验证：生成测试图
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", FONT_NAME]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot([1,2,3,4], [0.8, 0.3, 0.9, 0.5], "o-", color="steelblue")
ax.set_title("中文验证：坐姿抬手 / 慢走 / 站立坐下 / 低幅度区段", fontsize=13)
ax.set_xlabel("时间（秒）", fontsize=11)
ax.set_ylabel("幅度", fontsize=11)
plt.tight_layout()
plt.savefig("chinese_ok.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n测试图: chinese_ok.png  — 汉字应正常显示")
print("如果正常，后续所有脚本无需改动，重启 Python 后自动生效。")
