# 基础镜像
FROM base/python:ubuntu22.04-cuda11.8-python3.10

# 设置环境变量
#ENV all_proxy=http://192.168.0.64:7890
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# 设置默认时区禁止交互
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# 设置访问密钥
ENV TOOL-API-SECRET-KEY=sk-tool-api

# 设置工作目录
WORKDIR /app/Tool-API

# 克隆项目
RUN git clone https://github.com/zxd147/ToolAPI ../Tool-API

# 安装依赖
RUN pip install -r requirements.txt && pip cache purge

# 映射端口
EXPOSE 8092

# 容器启动时默认执行的命令
CMD ["python", "tool_api.py"]
