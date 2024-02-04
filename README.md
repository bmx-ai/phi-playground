# phi-playground
A series of utilities to play with microsoft/phi-* models 


download official model:
```
HF_HUB_ENABLE_HF_TRANSFER 1 huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2 --local-dir-use-symlinks False
```

this creates a directory inside models/phi-2


# Environment
setup
```bash
conda create --name bmx --file conda-linux-64.lock
conda activate my_project_env
poetry install
```

```bash
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```
