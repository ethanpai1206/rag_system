# 第一階段：構建基礎映像
# FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# 設置環境變數
ENV UBUNTU_HOME=/home/ubuntu \
    PASSWORD="22601576" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    LANG=zh_TW.UTF8 \
    LC_ALL=zh_TW.UTF8 \
    LANGUAGE=zh_TW.UTF8

# 配置時區和本地化設置
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata locales && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    locale-gen zh_TW.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安裝必要的依賴包
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake pkg-config unzip libgtk2.0-dev \
    wget curl ca-certificates libcurl4-openssl-dev libssl-dev \
    libavcodec-dev libavformat-dev libswscale-dev libtbbmalloc2 libtbb-dev \
    libharfbuzz-dev libfreetype-dev libpq-dev \
    libaio-dev libgoogle-perftools-dev libopenblas-dev tini supervisor openssh-server \
    clang-format clang-tidy lcov libtool m4 autoconf automake \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstrtspserver-1.0-dev libx11-dev \
    libjpeg-turbo8-dev libpng-dev libtiff-dev libdc1394-dev nasm \
    sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安裝 Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-gst-1.0 python3-pybind11 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    rm get-pip.py

# 安裝 Node.js 18.x 與 npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && \
    sudo apt-get install -y nodejs && \
    node -v && npm -v

# 創建用戶 ubuntu
RUN useradd -m -s /bin/bash ubuntu && \
    echo "ubuntu:22601576" | chpasswd

# 創建目錄和文件，確保目標存在
RUN mkdir -p ${UBUNTU_HOME} && \
    touch ${UBUNTU_HOME}/.bashrc

# 配置用戶環境
RUN echo 'export LANGUAGE="zh_TW.UTF-8"' >> ${UBUNTU_HOME}/.bashrc && \
    echo 'export LANG="zh_TW.UTF-8"' >> ${UBUNTU_HOME}/.bashrc && \
    echo 'export LC_ALL="zh_TW.UTF-8"' >> ${UBUNTU_HOME}/.bashrc && \
    chown ubuntu:ubuntu -R ${UBUNTU_HOME}

# 配置 npm 全局安裝設置
RUN mkdir -p ${UBUNTU_HOME}/.npm-global && \
    chown ubuntu:ubuntu ${UBUNTU_HOME}/.npm-global && \
    echo 'export PATH=~/.npm-global/bin:$PATH' >> ${UBUNTU_HOME}/.bashrc

# 設置工作目錄
WORKDIR $UBUNTU_HOME

# 添加 ubuntu 用戶到 sudo 群組
RUN usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 切換到 ubuntu 用戶
USER ubuntu

# 配置 npm 使用全局目錄並安裝 claude-code
RUN npm config set prefix '~/.npm-global' && \
    npm install -g @anthropic-ai/claude-code

# 創建專案目錄
RUN mkdir -p ${UBUNTU_HOME}/workspace/rag_system

# 複製整個專案到容器內
COPY --chown=ubuntu:ubuntu . ${UBUNTU_HOME}/workspace/rag_system/

# 設置工作目錄為專案目錄
WORKDIR ${UBUNTU_HOME}/workspace/rag_system

# 安裝 Python 依賴包
RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]