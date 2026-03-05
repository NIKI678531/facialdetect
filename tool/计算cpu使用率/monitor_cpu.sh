#!/bin/bash

SAMPLES=${1:-5}
INTERVAL=${2:-1}

total=0
echo "开始采集CPU使用率，共采样 $SAMPLES 次，每次间隔 $INTERVAL 秒..."

for ((i=1; i<=SAMPLES; i++))
do
    cpu_idle=$(top -b -n 1 | awk -F, '/%Cpu\(s\)/ { for(i=1;i<=NF;i++){ if($i~/id/){ gsub(/[^0-9.]/,"",$i); print $i } } }')

    # 使用 awk 计算浮点数
    cpu_usage=$(awk -v idle="$cpu_idle" 'BEGIN{printf "%.2f", 100 - idle}')

    # 累加总使用率（用字符串拼接实现累加）
    total=$(awk -v t="$total" -v u="$cpu_usage" 'BEGIN{printf "%.2f", t + u}')

    echo "采样 $i: CPU使用率 = ${cpu_usage}%"

    [ $i -ne $SAMPLES ] && sleep $INTERVAL
done

average=$(awk -v t="$total" -v s="$SAMPLES" 'BEGIN{printf "%.2f", t/s}')

echo "------------------------"
echo "平均CPU使用率: ${average}%"