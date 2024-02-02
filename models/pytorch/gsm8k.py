import re
import os
import datasets
import huggingface_hub
import torch

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"

ds = None
def load_from_storage(path:str):
    global ds
    if not os.path.exists(path):
       raise ValueError('Path does not exist.')
    try:
        ds = datasets.load_dataset('gsm8k', 'main', cache_dir=path)
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
                **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id, use_cache=True
            )
            text = tokenizer.batch_decode(out)[0]
            print(text[outidx:], end="", flush=True)
            outidx = len(text) 
            
            if out == model.config.eos_token_id:
                break
            # # this function triggers a calculator when it sees an equal token
            # if out[0, -1].item() in EQUALS_TOKENS:
            #     answer = use_calculator(text)
            #     if answer is not None:
            #         print("Triggered calculator, answer", answer)
            #         text = text + str(answer) + ">>"

            answer = text
    return answer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, answer):
    gt_answer = extract_answer(answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def evaluate(model, tokenizer):
    correct = 0
    total = 0
    for ex in ds['test']:
        output = sample(model, tokenizer, ex['question'], 150)
        ans = extract_answer(output)
        if is_correct(ans, ex['answer']):
            correct += 1
        total += 1
    return { "accuracy": correct / float(total) }
    