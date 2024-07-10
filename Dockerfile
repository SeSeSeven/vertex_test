# 使用官方Python 3.10镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 将requirements.txt文件和当前目录下的所有文件复制到容器中
COPY requirements.txt .
COPY . .

# 安装requirements.txt中指定的依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 安装DVC（如果需要）
# RUN pip install dvc

# 复制run_project_commands.sh脚本到容器中并给予执行权限
COPY run_project_commands.sh /app/run_project_commands.sh
RUN chmod +x /app/run_project_commands.sh

# 使用run_project_commands.sh脚本作为容器的入口点
ENTRYPOINT ["/app/run_project_commands.sh"]
