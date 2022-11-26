# ==================================================================
# module list
# ------------------------------------------------------------------
# darknet       latest (git)
# python        3.8    (conda)
# chainer       latest (pip)
# jupyter       latest (pip)
# mxnet         latest (pip)
# onnx          latest (pip)
# paddle        latest (pip)
# pytorch       latest (pip)
# tensorflow    latest (pip)
# jupyterlab    latest (pip)
# keras         latest (pip)
# ==================================================================

ARG SPARK_VERSION=3.0.0

ENV LANG C.UTF-8
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV SPARK_HOME="/opt/miniconda3/lib/python3.8/site-packages/pyspark"

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


# ==================================================================
# configs
# ------------------------------------------------------------------
RUN set -ex && \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get clean && \
    apt-get update \
    && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        bzip2 \
        unrar \
        cmake \
        openjdk-8-jre-headless \
        && \
# ==================================================================
# python¡¢pyspark
# ------------------------------------------------------------------
    curl -s -L --url "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \ 
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda install pip && \
    conda clean -tipy && \
    echo "PATH=/opt/miniconda3/bin:\${PATH}" > /etc/profile.d/miniconda.sh && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        scikit-image \
        scikit-learn \
        matplotlib \
        seaborn \
        Cython \
        tqdm \
        pyspark==${SPARK_VERSION} \
        && \
    echo "export SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py)" > /etc/profile.d/spark.sh && \
    curl -s -L --url "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar" --output ${SPARK_HOME}/jars/aws-java-sdk-1.7.4.jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.3/hadoop-aws-2.7.3.jar" --output ${SPARK_HOME}/jars/hadoop-aws-2.7.2jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/io/delta/delta-core_2.12/1.0.0/delta-core_2.12-1.0.0.jar" --output ${SPARK_HOME}/jars/delta-core_2.12-1.0.0.jar && \
    mkdir -p ${SPARK_HOME}/conf && \
    echo "spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> ${SPARK_HOME}/conf/spark-defaults.conf 
    && \ 
# ==================================================================
# jupyter
# ------------------------------------------------------------------
    $PIP_INSTALL \
        jupyter \
        && \
# ==================================================================
# paddle
# ------------------------------------------------------------------
    $PIP_INSTALL \
        paddlepaddle-gpu \
        && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html \
        && \
# ==================================================================
# jupyterlab
# ------------------------------------------------------------------
    $PIP_INSTALL \
        jupyterlab \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*