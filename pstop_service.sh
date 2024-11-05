#!/bin/bash

# 查找监听指定端口的进程并终止它
PORT=8092
PID=$(lsof -t -i:${PORT})
# 输出文件名
output_txt="output.txt"

if [ -z "${PID}" ]; then
    echo "No process is listening on port ${PORT}."
else
    echo "Stopping process on port ${PORT} with PID ${PID}..."
    kill ${PID}
    sleep 1
    # 检查进程是否仍然存在，强制停止
    if ps -p ${PID} > /dev/null; then
      echo "Process ${PID} on port ${PORT} did not stop, forcing stop."
      sudo kill -9 ${PID}
    else
      echo "Process ${PID} stopped successfully."
    fi
fi

echo "---------------[${timestamp}]-------------" > "${output_txt}"
echo "Service on ${PORT} stop successfully" | tee -a "${output_txt}"


