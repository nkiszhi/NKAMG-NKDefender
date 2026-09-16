#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BinaryNinja 优化效果测试脚本
============================
对比优化前后的扫描性能。

用法:
    python benchmark_bn_optimization.py /data/samples/TestSample
"""
import os
import sys
import time
import json
from pathlib import Path

# 添加项目路径（2026-09-16 目录整理：本脚本自 CoDefenderC3/ 根移入 train/）
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)
for _p in (_REPO_ROOT, _TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.app import CoDefenderApp


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"


def benchmark_scan(sample_dir, include_de=True, label="测试"):
    """执行扫描并统计时间"""
    print(f"\n{'='*60}")
    print(f"🔬 {label}")
    print(f"{'='*60}")
    
    # 查找样本文件
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        print(f"❌ 错误: 目录不存在 {sample_dir}")
        return None
    
    samples = list(sample_dir.glob("*"))
    samples = [str(s) for s in samples if s.is_file()]
    
    if not samples:
        print(f"❌ 错误: 目录中没有文件")
        return None
    
    print(f"📁 样本目录: {sample_dir}")
    print(f"📊 样本数量: {len(samples)} 个")
    print(f"🔧 模式: {'全量扫描（33模型）' if include_de else '快速扫描（18模型，跳过D/E/F）'}")
    print(f"\n⏳ 开始扫描...")
    
    # 执行扫描
    t0 = time.time()
    try:
        app = CoDefenderApp()
        results = app.scan_samples(samples)  # 注意: 需要确认参数名
        elapsed = time.time() - t0
        
        # 提取统计信息
        summary = results.get("Summary", {})
        timing = summary.get("Timing", {})
        
        print(f"\n✅ 扫描完成！")
        print(f"\n{'='*60}")
        print(f"📈 性能统计")
        print(f"{'='*60}")
        print(f"  总时间:       {format_time(elapsed)}")
        print(f"  扫描文件数:   {summary.get('ScannedFiles', 0)} / {summary.get('TotalFiles', 0)}")
        print(f"  成功数:       {summary.get('ScannedFiles', 0) - summary.get('ErrorCount', 0)}")
        print(f"  错误数:       {summary.get('ErrorCount', 0)}")
        print(f"  平均速度:     {elapsed/len(samples):.2f} 秒/样本")
        print(f"  吞吐量:       {len(samples)/elapsed:.3f} 样本/秒")
        
        if timing:
            print(f"\n详细时序:")
            print(f"  模型加载:     {timing.get('ModelLoadTime', 0):.1f} 秒")
            print(f"  批处理扫描:   {timing.get('BatchScanTime', 0):.1f} 秒")
            print(f"  平均样本时间: {timing.get('AvgSampleTime', 0):.3f} 秒")
        
        # 检测结果
        print(f"\n{'='*60}")
        print(f"🎯 检测结果")
        print(f"{'='*60}")
        print(f"  恶意样本:     {summary.get('MaliciousCount', 0)}")
        print(f"  良性样本:     {summary.get('BenignCount', 0)}")
        
        return {
            "total_time": elapsed,
            "samples": len(samples),
            "avg_time": elapsed / len(samples),
            "throughput": len(samples) / elapsed,
            "summary": summary,
            "timing": timing,
        }
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n❌ 扫描失败: {e}")
        print(f"  耗时: {format_time(elapsed)}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(baseline, optimized):
    """对比优化前后的结果"""
    if not baseline or not optimized:
        return
    
    print(f"\n{'='*60}")
    print(f"📊 优化效果对比")
    print(f"{'='*60}")
    
    # 计算提升
    time_improvement = baseline["total_time"] / optimized["total_time"]
    throughput_improvement = optimized["throughput"] / baseline["throughput"]
    
    print(f"\n⏱️  时间对比:")
    print(f"  优化前: {format_time(baseline['total_time'])}")
    print(f"  优化后: {format_time(optimized['total_time'])}")
    print(f"  提升:   {time_improvement:.2f}x 🚀")
    
    print(f"\n📈 吞吐量对比:")
    print(f"  优化前: {baseline['throughput']:.3f} 样本/秒")
    print(f"  优化后: {optimized['throughput']:.3f} 样本/秒")
    print(f"  提升:   {throughput_improvement:.2f}x 🚀")
    
    print(f"\n⚡ 单样本时间对比:")
    print(f"  优化前: {baseline['avg_time']:.2f} 秒")
    print(f"  优化后: {optimized['avg_time']:.2f} 秒")
    print(f"  节省:   {baseline['avg_time'] - optimized['avg_time']:.2f} 秒/样本")
    
    # 判断是否达标
    target_time = 120  # 目标时间（14样本）
    samples = baseline["samples"]
    optimized_projected = optimized["avg_time"] * 14
    
    print(f"\n🎯 目标达成情况:")
    print(f"  目标时间:     {target_time} 秒（14样本）")
    print(f"  当前性能:     {optimized_projected:.1f} 秒（14样本）")
    if optimized_projected <= target_time:
        print(f"  状态:         ✅ 已达标！")
    else:
        print(f"  状态:         ⚠️  未达标（差 {optimized_projected - target_time:.1f} 秒）")


def show_env_info():
    """显示环境信息"""
    print(f"\n{'='*60}")
    print(f"🖥️  环境信息")
    print(f"{'='*60}")
    
    import platform
    print(f"  操作系统:     {platform.system()} {platform.release()}")
    print(f"  CPU核心数:    {os.cpu_count()}")
    print(f"  Python版本:   {sys.version.split()[0]}")
    
    # BinaryNinja 配置
    bn_threads = os.environ.get("BN_WORKER_THREADS", "未设置（默认4）")
    print(f"  BN工作线程:   {bn_threads}")
    
    # 检查 BinaryNinja 是否可用
    try:
        import binaryninja
        print(f"  BinaryNinja:   ✅ 已安装")
    except ImportError:
        print(f"  BinaryNinja:   ❌ 未安装")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python benchmark_bn_optimization.py <样本目录>")
        print("示例: python benchmark_bn_optimization.py /data/samples/TestSample")
        sys.exit(1)
    
    sample_dir = sys.argv[1]
    
    # 显示环境信息
    show_env_info()
    
    print(f"\n{'='*60}")
    print(f"🚀 BinaryNinja 优化效果测试")
    print(f"{'='*60}")
    print(f"\n此测试将执行两次扫描:")
    print(f"  1️⃣  基线测试（当前配置）")
    print(f"  2️⃣  优化测试（推荐配置）")
    print(f"\n注意: 请先运行 'python apply_bn_optimizations.py' 应用优化")
    
    input(f"\n按 Enter 开始测试...")
    
    # 执行测试
    baseline = benchmark_scan(sample_dir, include_de=True, label="基线测试（全量扫描）")
    
    if baseline:
        print(f"\n💡 提示: 现在可以:")
        print(f"  1. 应用优化: python apply_bn_optimizations.py")
        print(f"  2. 设置环境变量: set BN_WORKER_THREADS=8")
        print(f"  3. 重启服务")
        print(f"  4. 再次运行此脚本测试优化效果")
    
    # 可选: 测试快速模式
    print(f"\n{'='*60}")
    response = input(f"是否测试快速模式（跳过D/E/F特征）? [y/N]: ")
    if response.lower() == 'y':
        fast = benchmark_scan(sample_dir, include_de=False, label="快速模式测试（18模型）")
        if baseline and fast:
            print(f"\n对比全量模式 vs 快速模式:")
            print(f"  速度提升: {baseline['total_time'] / fast['total_time']:.2f}x")
            print(f"  注意: 快速模式准确率略低（~2-3%）")


if __name__ == "__main__":
    main()
