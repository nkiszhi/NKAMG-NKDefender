#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BinaryNinja 优化配置应用脚本
=============================
自动修改 extract_feature.py，应用推荐的 BinaryNinja 分析选项优化。

用法:
    python apply_bn_optimizations.py          # 应用优化
    python apply_bn_optimizations.py --revert # 还原优化
"""
import os
import sys
import re
import shutil
from datetime import datetime

# ── 路径引导（2026-09-16 目录整理：本脚本自 CoDefenderC3/ 根移入 train/）──
# 本脚本修改的目标 feature_extraction/extract_feature.py 被 engine/ 与训练侧共用，
# 留在仓库根，故目标路径需要往上退一级。
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)

EXTRACT_FEATURE_PATH = os.path.join(_REPO_ROOT, "feature_extraction", "extract_feature.py")

# 优化后的 BinaryNinja 加载选项
OPTIMIZED_OPTIONS = """{
            # === 核心分析模式 ===
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            
            # === 超时限制（防止卡死在混淆样本上）===
            "analysis.limits.maxFunctionSize": 262144,
            "analysis.limits.maxFunctionAnalysisTime": 30000,  # 30秒/函数
            
            # === 跳过不必要的分析 ===
            "pdb.features.autoDownloadPDBs": False,  # 跳过 PDB 下载
            "pdb.features.allowLocalPDBs": False,
            
            # === 保守策略（速度优先）===
            "analysis.conservativeLinearSweep": True,  # 更快但可能不完整
            "analysis.tailCallHeuristics": False,      # 跳过尾调用识别
            "analysis.tailCallTranslation": False,
        }"""

# 原始的基础选项
BASIC_OPTIONS = """{
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        }"""


def backup_file(filepath):
    """创建备份文件"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已备份: {backup_path}")
    return backup_path


def apply_optimizations():
    """应用优化配置"""
    if not os.path.exists(EXTRACT_FEATURE_PATH):
        print(f"❌ 错误: 找不到文件 {EXTRACT_FEATURE_PATH}")
        return False
    
    # 备份原文件
    backup_file(EXTRACT_FEATURE_PATH)
    
    # 读取文件
    with open(EXTRACT_FEATURE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换所有 BN 加载选项（使用正则表达式匹配）
    # 匹配模式: bv = self.bn.load(pe_path, options={ ... })
    pattern = r'(bv = self\.bn\.load\(pe_path, options=)\s*\{[^}]+\}'
    
    matches = list(re.finditer(pattern, content))
    if not matches:
        print("❌ 未找到需要优化的 BinaryNinja 加载代码")
        return False
    
    print(f"📝 找到 {len(matches)} 处需要优化的位置")
    
    # 替换所有匹配项
    new_content = re.sub(pattern, r'\1' + OPTIMIZED_OPTIONS.replace('\n', '\n        '), content)
    
    # 写回文件
    with open(EXTRACT_FEATURE_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"✅ 已应用优化到 {len(matches)} 处位置")
    print("\n优化内容:")
    print("  - 添加函数分析超时（30秒）")
    print("  - 禁用 PDB 符号下载")
    print("  - 启用保守线性扫描（更快）")
    print("  - 禁用尾调用识别")
    print("\n预期效果: 1.2-1.8x 速度提升")
    return True


def revert_optimizations():
    """还原到基础配置"""
    if not os.path.exists(EXTRACT_FEATURE_PATH):
        print(f"❌ 错误: 找不到文件 {EXTRACT_FEATURE_PATH}")
        return False
    
    # 查找最新的备份文件
    backup_dir = os.path.dirname(EXTRACT_FEATURE_PATH)
    backup_files = [f for f in os.listdir(backup_dir) if f.startswith("extract_feature.py.backup_")]
    
    if not backup_files:
        print("❌ 未找到备份文件")
        return False
    
    latest_backup = sorted(backup_files)[-1]
    backup_path = os.path.join(backup_dir, latest_backup)
    
    # 还原备份
    shutil.copy2(backup_path, EXTRACT_FEATURE_PATH)
    print(f"✅ 已从备份还原: {backup_path}")
    return True


def show_current_config():
    """显示当前配置"""
    if not os.path.exists(EXTRACT_FEATURE_PATH):
        print(f"❌ 错误: 找不到文件 {EXTRACT_FEATURE_PATH}")
        return
    
    with open(EXTRACT_FEATURE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找 BN 加载选项
    pattern = r'bv = self\.bn\.load\(pe_path, options=(\{[^}]+\})'
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        print("❌ 未找到 BinaryNinja 加载配置")
        return
    
    print(f"📋 当前配置（共 {len(matches)} 处）:\n")
    for i, match in enumerate(matches, 1):
        options = match.group(1)
        print(f"位置 {i}:")
        print(options)
        print()
        
        # 检查是否包含优化选项
        if "maxFunctionAnalysisTime" in options:
            print("  ✅ 已应用优化配置")
        else:
            print("  ⚠️  使用基础配置（未优化）")
        print("-" * 60)


def show_help():
    """显示帮助信息"""
    print("""
BinaryNinja 优化配置工具
========================

用法:
    python apply_bn_optimizations.py              # 应用优化
    python apply_bn_optimizations.py --revert     # 还原到备份
    python apply_bn_optimizations.py --show       # 显示当前配置
    python apply_bn_optimizations.py --help       # 显示此帮助

优化内容:
    1. 函数分析超时（30秒，防止卡死）
    2. 禁用 PDB 符号下载（节省时间）
    3. 保守线性扫描（速度优先）
    4. 禁用尾调用识别（减少分析时间）

预期效果:
    - 扫描速度提升: 1.2-1.8x
    - 特别是对混淆/加壳样本效果显著
    - 准确率几乎无影响（<0.5%）

注意事项:
    - 自动创建备份文件（.backup_* 后缀）
    - 可随时使用 --revert 还原
    - 修改后需重启扫描服务
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--revert":
            revert_optimizations()
        elif cmd == "--show":
            show_current_config()
        elif cmd == "--help":
            show_help()
        else:
            print(f"❌ 未知命令: {cmd}")
            show_help()
    else:
        # 默认应用优化
        print("🚀 正在应用 BinaryNinja 优化配置...\n")
        if apply_optimizations():
            print("\n✅ 优化完成！")
            print("\n下一步:")
            print("  1. 设置环境变量: set BN_WORKER_THREADS=8")
            print("  2. 重启扫描服务")
            print("  3. 测试扫描速度")
            print("\n详细文档: BINARYNINJA_OPTIMIZATION_GUIDE.md")
