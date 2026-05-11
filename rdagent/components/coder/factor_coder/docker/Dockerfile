FROM python:3.10-slim-bookworm

ARG USE_CHINA_MIRROR=false

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
      sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
      && sed -i 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources \
      && apt-get clean \
      && apt-get update \
      && apt-get install -y --no-install-recommends libgomp1 \
      && rm -rf /var/lib/apt/lists/*; \
    else \
      apt-get update \
      && apt-get install -y --no-install-recommends libgomp1 \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
      python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      && python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
    fi \
    && python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --prefer-binary --retries 5 --timeout 120 \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    tables \
    h5py \
    pyarrow \
    numba \
    statsmodels \
    tqdm \
    pyyaml \
    filelock \
    jinja2 \
    networkx \
    sympy \
    bottleneck \
    openpyxl \
    xlrd \
    joblib \
    requests \
    charset-normalizer \
    chardet \
    einops \
    xgboost \
    lightgbm

RUN python -m pip install --prefer-binary --retries 5 --timeout 120 \
    torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org

WORKDIR /workspace/factor_workspace
