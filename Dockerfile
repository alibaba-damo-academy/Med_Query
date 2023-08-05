FROM nvidia/cuda:11.4.3-base-centos7

RUN mkdir -p /workspace/
WORKDIR /workspace

RUN yum -y update && yum -y install wget

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n pytorch1.7 python=3.7.3
ENV PATH=$CONDA_DIR/envs/pytorch1.7/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch1.7

RUN pip install -U pip
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com

# For opencv-python
RUN yum install mesa-libGL

RUN pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

