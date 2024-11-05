#!/bin/bash

bash pstop_service.sh

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
# 日志文件名
log_dir="logs"
output_log="output.log"
# 输出文件名
output_txt="output.txt"

# 如果日志文件存在，则重命名为带有时间戳的新文件名
if [ -f "${log_dir}/${output_log}" ]; then
    mv "${log_dir}/${output_log}" "${log_dir}/${timestamp}_${output_log}"
    echo "Existing log file renamed to ${timestamp}_${output_log}"
fi

# 启动服务并将输出重定向到 out_app.log
nohup /home/zxd/.conda/envs/api/bin/python tool_api.py > "${log_dir}/${output_log}" 2>&1 &

# 获取启动的进程 ID 并打印
PID=$!
echo "---------------[${timestamp}]-------------" > "${output_txt}"
echo "Service started successfully with PID $PID" | tee -a "${output_txt}"

# 保持前台运行
# while kill -0 $PID 2>/dev/null; do
#     sleep 10
# done

