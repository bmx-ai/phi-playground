```
sudo apt install git git-lfs openmpi-bin libopenmpi-dev
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout b57221b764bc579cbb2490154916a871f620e2c4 -b phi
git submodule update --init --recursive
git lfs install
git lfs pull
make -C docker release_build CUDA_ARCHS="86-real"
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