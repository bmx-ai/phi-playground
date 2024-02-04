```
sudo apt install git git-lfs openmpi-bin libopenmpi-dev -y
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout b57221b764bc579cbb2490154916a871f620e2c4 -b phi
git submodule update --init --recursive
git lfs install
git lfs pull
echo $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
make -C docker release_build CUDA_ARCHS=90-real
LOCAL_USER=1 make -C docker release_run
# inside the docker
cd /code/tensorrt_llm/examples/phi
pip install huggingface-cli hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download microsoft/phi-2 --local-dir phi_model --local-dir-use-symlinks False
python3 build.py --dtype=float16                    \
                 --log_level=verbose                \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16          \
                 --max_batch_size=16                \
                 --max_input_len=1024               \
                 --max_output_len=1024              \
                 --output_dir=phi_engine            \
                 --model_dir=./phi_model 2>&1 | tee build.log

# To build the TensorRT-LLM code. (CPython3.10)
cd /code/tensorrt_llm/TensorRT-LLM/ && python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt
```

# Benchmark
```
./benchmarks/gptSessionBenchmark \
    --model phi \
    --engine_dir /code/tensorrt_llm/examples/phi/phi_engine/ \
    --batch_size "1" \
    --input_output_len "60,20"
```

## Eval

python eval.py --max_output_len=50 --engine_dir=/code/tensorrt_llm/TensorRT-LLM/examples/phi/phi_engine --tokenizer_dir=/code/tensorrt_llm/TensorRT-LLM/examples/phi/phi_model


# Run the build script. The flags for some features or endpoints can be removed if not needed.
BASE_CONTAINER_IMAGE_NAME=nvcr.io/nvidia/tritonserver:23.10-py3-min
TENSORRTLLM_BACKEND_REPO_TAG=releases/v0.7.2
PYTHON_BACKEND_REPO_TAG=r23.10

./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
              --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --image=base,${BASE_CONTAINER_IMAGE_NAME} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}