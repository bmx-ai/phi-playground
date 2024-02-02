#!/bin/env bash
set -e

export PYTHONPATH=models/pytorch/
python -m eval_gsm8k \
    --model_module "pytorch_microsoft_phi_model" \
    --model_storage ".cache/models/microsoft/phi-2" \
    --evaluation_module "gsm8k" \
    --evaluation_storage ".cache/datasets/gsm8k" 