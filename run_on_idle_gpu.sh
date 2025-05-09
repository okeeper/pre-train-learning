#!/bin/bash

# 使用方法提示函数
usage() {
  echo "用法: $0 [选项] <命令>"
  echo "选项:"
  echo "  -m <阈值>    内存使用率阈值，低于此值视为空闲 (默认: 20.0%)"
  echo "  -u <阈值>    计算负载阈值，低于此值视为空闲 (默认: 10.0%)"
  echo "  -i <秒数>    检查间隔时间，单位秒 (默认: 60)"
  echo "  -n <次数>    连续几次检查空闲才执行命令 (默认: 3)"
  echo "  -g <GPU ID>  指定要监控的GPU ID，不指定则监控所有GPU"
  echo "  -r           命令执行完毕后继续监控并在GPU空闲时再次执行"
  echo "  -h           显示此帮助信息"
  echo
  echo "示例:"
  echo "  $0 -g 0 -m 30 -i 30 'python train.py --batch_size 32'"
  echo "  $0 -r -n 2 'bash run_training.sh'"
  exit 1
}

# 默认参数
MEMORY_THRESHOLD=20.0
UTILIZATION_THRESHOLD=10.0
CHECK_INTERVAL=10
CONSECUTIVE_CHECKS=3
GPU_ID=""
RETRY=""

# 解析命令行参数
while getopts "m:u:i:n:g:rh" opt; do
  case $opt in
    m) MEMORY_THRESHOLD=$OPTARG ;;
    u) UTILIZATION_THRESHOLD=$OPTARG ;;
    i) CHECK_INTERVAL=$OPTARG ;;
    n) CONSECUTIVE_CHECKS=$OPTARG ;;
    g) GPU_ID="--gpu-id $OPTARG" ;;
    r) RETRY="--retry" ;;
    h) usage ;;
    \?) echo "无效的选项: -$OPTARG" >&2; usage ;;
  esac
done

# 移除已处理的选项，剩余的是要执行的命令
shift $((OPTIND-1))

# 检查是否提供了命令
if [ $# -eq 0 ]; then
  echo "错误: 必须提供要执行的命令" >&2
  usage
fi

# 获取完整命令
COMMAND="$*"

# 检查gpu_monitor.py是否存在
if [ ! -f "$(dirname "$0")/gpu_monitor.py" ]; then
  echo "错误: 未找到gpu_monitor.py脚本，请确保它与此脚本在同一目录" >&2
  exit 1
fi

# 将命令中的单引号替换为转义形式，以防止命令行解析错误
ESCAPED_COMMAND=$(echo "$COMMAND" | sed "s/'/\\\'/g")

# 运行GPU监控脚本
echo "启动GPU监控器，等待GPU空闲..."
python "$(dirname "$0")/gpu_monitor.py" \
  --memory-threshold "$MEMORY_THRESHOLD" \
  --utilization-threshold "$UTILIZATION_THRESHOLD" \
  --check-interval "$CHECK_INTERVAL" \
  --consecutive-checks "$CONSECUTIVE_CHECKS" \
  $GPU_ID \
  $RETRY \
  --command "$ESCAPED_COMMAND" 