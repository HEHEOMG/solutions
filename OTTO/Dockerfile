# 定义两个局部变量，用于定义openjdk的版本等
ARG OPENJDK_VERSION=8 
ARG DEBIAN_RELEASE=""
FROM openjdk:${OPENJDK_VERSION}-jre-slim${DEBIAN_RELEASE}

# 定义spark版本和类似于pyspark带后缀的pip包
ARG SPARK_VERSION=3.0.0
ARG SPARK_EXTRAS=

# 修改这个镜像的元数据，即增加一点描述
LABEL org.opencontainers.image.title="Apache PySpark $SPARK_VERSION" \
      org.opencontainers.image.version=$SPARK_VERSION

# 定义环境变量，增加一个搜索路径和python环境地址
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"

# 
RUN set -ex && \
    apt-get update && \
    apt-get install -y curl bzip2 --no-install-recommends && \
    curl -s -L --url "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda install pip && \
    conda clean -tipy && \
    echo "PATH=/opt/miniconda3/bin:\${PATH}" > /etc/profile.d/miniconda.sh && \
    pip install --no-cache pyspark[$SPARK_EXTRAS]==${SPARK_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple && \ 
    echo "export SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py)" > /etc/profile.d/spark.sh && \
    curl -s -L --url "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar" --output $SPARK_HOME/jars/aws-java-sdk-1.7.4.jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.3/hadoop-aws-2.7.3.jar" --output $SPARK_HOME/jars/hadoop-aws-2.7.2jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/io/delta/delta-core_2.12/1.0.0/delta-core_2.12-1.0.0.jar" --output $SPARK_HOME/jars/delta-core_2.12-1.0.0.jar && \
    mkdir -p $SPARK_HOME/conf && \
    echo "spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> $SPARK_HOME/conf/spark-defaults.conf && \ 
    apt-get remove -y curl bzip2 && \ 
    apt-get autoremove -y && \
    apt-get clean 

