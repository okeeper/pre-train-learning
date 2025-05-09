#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import argparse
import os
import signal
import sys
import datetime

def log(message):
    """记录日志，包含时间戳"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def get_gpu_memory_usage():
    """获取所有GPU内存使用情况，返回利用率列表"""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
        lines = output.decode('utf-8').strip().split('\n')
        
        utilizations = []
        for line in lines:
            memory_used, memory_total = map(int, line.split(','))
            utilization = memory_used / memory_total * 100.0
            utilizations.append(utilization)
        
        return utilizations
    except Exception as e:
        log(f"获取GPU内存使用情况出错: {e}")
        return None

def get_gpu_usage():
    """获取GPU计算负载情况，返回利用率列表"""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        lines = output.decode('utf-8').strip().split('\n')
        return [float(line) for line in lines]
    except Exception as e:
        log(f"获取GPU计算负载出错: {e}")
        return None

def is_gpu_idle(memory_threshold, utilization_threshold, gpu_id=None):
    """检查指定GPU是否空闲，不指定时检查所有GPU"""
    memory_utilizations = get_gpu_memory_usage()
    compute_utilizations = get_gpu_usage()
    
    if memory_utilizations is None or compute_utilizations is None:
        return False
    
    gpu_count = len(memory_utilizations)
    
    # 如果指定了GPU ID，只检查该GPU
    if gpu_id is not None:
        if gpu_id >= gpu_count:
            log(f"指定的GPU ID {gpu_id} 超出范围，总共有 {gpu_count} 个GPU")
            return False
        
        return (memory_utilizations[gpu_id] <= memory_threshold and 
                compute_utilizations[gpu_id] <= utilization_threshold)
    
    # 检查所有GPU
    all_gpus_idle = gpu_count > 0
    for i in range(gpu_count):
        mem_usage = memory_utilizations[i]
        comp_usage = compute_utilizations[i]
        log(f"GPU {i}: 内存使用率 {mem_usage:.2f}%, 计算负载 {comp_usage:.2f}%")
        
        if mem_usage > memory_threshold or comp_usage > utilization_threshold:
            all_gpus_idle = False
    
    return all_gpus_idle

def run_command(command):
    """运行命令并实时输出结果"""
    try:
        log(f"开始执行命令: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  bufsize=1, universal_newlines=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()
        
        return_code = process.poll()
        if return_code == 0:
            log("命令执行成功")
        else:
            log(f"命令执行失败，返回码: {return_code}")
        return return_code
    except Exception as e:
        log(f"执行命令时出错: {e}")
        return -1

def handle_signal(sig, frame):
    """处理中断信号"""
    log("收到中断信号，正在退出...")
    sys.exit(0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU监控器：当GPU空闲时执行命令")
    parser.add_argument("--command", "-c", required=True, help="当GPU空闲时要执行的命令")
    parser.add_argument("--memory-threshold", "-m", type=float, default=20.0,
                      help="内存使用率阈值，低于此值视为空闲 (默认: 20.0%%)")
    parser.add_argument("--utilization-threshold", "-u", type=float, default=10.0,
                      help="计算负载阈值，低于此值视为空闲 (默认: 10.0%%)")
    parser.add_argument("--check-interval", "-i", type=int, default=60,
                      help="检查间隔时间，单位秒 (默认: 60)")
    parser.add_argument("--consecutive-checks", "-n", type=int, default=3,
                      help="连续几次检查空闲才执行命令 (默认: 3)")
    parser.add_argument("--gpu-id", "-g", type=int, default=None,
                      help="指定要监控的GPU ID，不指定则监控所有GPU")
    parser.add_argument("--retry", "-r", action="store_true",
                      help="命令执行完毕后继续监控并在GPU空闲时再次执行")
    
    args = parser.parse_args()
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    log("GPU监控器已启动")
    log(f"内存使用率阈值: {args.memory_threshold}%")
    log(f"计算负载阈值: {args.utilization_threshold}%")
    log(f"检查间隔: {args.check_interval}秒")
    log(f"连续检查次数: {args.consecutive_checks}")
    if args.gpu_id is not None:
        log(f"监控GPU ID: {args.gpu_id}")
    else:
        log("监控所有GPU")
    log(f"待执行命令: {args.command}")
    
    consecutive_idle_count = 0
    
    while True:
        if is_gpu_idle(args.memory_threshold, args.utilization_threshold, args.gpu_id):
            consecutive_idle_count += 1
            log(f"GPU空闲检测 ({consecutive_idle_count}/{args.consecutive_checks})")
            
            if consecutive_idle_count >= args.consecutive_checks:
                log(f"GPU已连续{args.consecutive_checks}次检测为空闲状态，开始执行命令")
                run_command(args.command)
                
                if not args.retry:
                    log("命令执行完毕，退出监控")
                    break
                else:
                    log("命令执行完毕，继续监控GPU")
                    consecutive_idle_count = 0
        else:
            if consecutive_idle_count > 0:
                log("GPU不再空闲，重置计数器")
            consecutive_idle_count = 0
        
        time.sleep(args.check_interval)

if __name__ == "__main__":
    main() 