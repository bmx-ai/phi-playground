ARG CUDA_RELEASE=12.1.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_RELEASE} AS base

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt update \
    && apt install --no-install-recommends -y \
    curl git \
    ca-certificates \
    sudo \
    && apt-get clean

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh  \
    && bash ~/miniconda.sh -b -p /miniconda \
    && /miniconda/bin/conda init \
    && rm ~/miniconda.sh

ENV PATH=/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Create a Python 3.11 environment
RUN /miniconda/bin/conda install conda-build \
    && /miniconda/bin/conda create -y --name py311 python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    && /miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py311
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

ARG USER_NAME=ubuntu
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} -o ${USER_NAME}
RUN useradd --shell /bin/bash \
    --no-log-init \
    --system \
    --create-home \
    --gid ${GID} \
    --uid ${UID} \
    ${USER_NAME}

USER ${USER_NAME}

# Install apex
# RUN git clone https://github.com/NVIDIA/apex --depth 1
# RUN cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
