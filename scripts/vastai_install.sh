export GITHUB_USERNAME=Mistobaan
sh -c "$(curl -fsLS get.chezmoi.io)" -- init --apply $GITHUB_USERNAME

sudo apt install git git-lfs openmpi-bin libopenmpi-dev -y

# Install Miniconda
curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh  \
    && bash ~/miniconda.sh -b -p /miniconda \
    && /miniconda/bin/conda init \
    && rm ~/miniconda.sh

# Create a Python 3.11 environment
/miniconda/bin/conda install conda-build \
    && /miniconda/bin/conda create -y --name py311 python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    && /miniconda/bin/conda clean -ya
