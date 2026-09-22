# JONSWAP 频谱核算与反演后端 —— 容器镜像
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# 先装依赖，利用构建缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 拷贝服务与测试
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini

# 容器内工况档持久化目录（运行时可通过 JONSWAP_DATA_DIR 覆盖）
ENV JONSWAP_DATA_DIR=/srv/data
RUN mkdir -p /srv/data

# 镜像构建时执行自动化测试：覆盖谱形正算、矩积分收敛、波高反演闭合、
# γ=1 退化（PM）与非法参数拦截。任一不过，镜像构建失败。
RUN python -m pytest -v

# 运行镜像：容器一启动即对外提供 HTTP 接口（仅 JSON，无网页界面）
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
