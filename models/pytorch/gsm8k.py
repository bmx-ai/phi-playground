import re
import os
import datasets
import huggingface_hub
import torch
import logging

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"

ds = None
def load_from_storage(path:str):
    global ds
    if not os.path.exists(path):
       raise ValueError('Path does not exist.')
    try:
        ds = datasets.load_dataset('gsm8k', 'main', split='test', cache_dir=path)
    except Exception as e:
        # download if not found
        huggingface_hub.hf_hub_download("gsm8k", cache_dir=path)
        try:
            # try to load it again
            ds = datasets.load_dataset('gsm8k', 'main', cache_dir=path)
        except:
            raise e


def sample(model, tokenizer, prompt, sample_len):
    outidx = 0
    answer = prompt
    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([answer], padding=False, return_tensors="pt").to(model.device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, max_length=orig_len + 1, pad_token_id=tokenizer.eos_token_id, use_cache=True
            )
            text = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
            print(text[outidx:], end="", flush=True)
            outidx = len(text) 
            if text.strip().endswith("<|endoftext|>"):
                break
            # # this function triggers a calculator when it sees an equal token
            # if out[0, -1].item() in EQUALS_TOKENS:
            #     answer = use_calculator(text)
            #     if answer is not None:
            #         print("Triggered calculator, answer", answer)
            #         text = text + str(answer) + ">>"

            answer = text
    print()
    print('-'*80)
    return answer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return float(match_str)
    else:
        return INVALID_ANS


def is_correct(model_completion, answer):
    gt_answer = extract_answer(answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

PYTHON_EVAL="""\
def simple_math_problem() -> float:
    '''
    {problem}
    '''\
"""
import textwrap

from code_utils import execute_code

def evaluate(model, tokenizer):
    correct = 0
    total = 0
    for ex in ds:
        p = textwrap.wrap(ex['question'])
        # p = textwrap.indent('\n'.join(p))
        question = PYTHON_EVAL.format(problem='\n'.join(p))
        code = sample(model, tokenizer, question, 2048)
        try:
            timeout = 60 # sec
            exit_code, result, _ = execute_code(code + '\nprint("####", simple_math_problem())', timeout=timeout, use_docker=True)
            success = exit_code == 0
            ok = is_correct(result, ex['answer'])
            result = result.strip()
            if ok:
                correct += 1
            print(ex['answer'], result, ": ok ? ", ok)
            print('='*80)
        except:
            logging.error("could not evaluate", exc_info=True)
        total += 1
    return { "accuracy": correct / float(total) }
    