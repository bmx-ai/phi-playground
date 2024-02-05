import re
import os
import datasets
import huggingface_hub
import torch
import logging
import concurrent.futures

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['TOKENIZERS_PARALLELISM']="1"
           
ds = None
def load_from_storage(path:str):
    global ds
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


def stream_sample(model, tokenizer, prompt, sample_len):
    outidx = 0
    answer = prompt
    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([answer], padding=False, return_tensors="pt").to(model.device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, max_length=sample_len, pad_token_id=tokenizer.eos_token_id, use_cache=True
            )
            text = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
            # print(text[outidx:], end="", flush=True)
            # outidx = len(text) 
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

def sample(model, tokenizer, prompts, sample_len, **genargs):
    tokenizer.pad_token = tokenizer.eos_token
    toks = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
    out = model.generate(
        **toks, max_length=sample_len, pad_token_id=tokenizer.eos_token_id, use_cache=True, **genargs
    )
    return tokenizer.batch_decode(out, skip_special_tokens=False)

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
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
import hashlib

def test_example(batch, model, tokenizer, shots, genargs):
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = "\n"
    prompts = []
    ids = []
    for ex in batch['question']:
        md5 = hashlib.md5(ex.encode('utf-8')).hexdigest()
        p = textwrap.wrap(ex)
        p = textwrap.indent('\n'.join(p), ' ' * 4)
        question = PYTHON_EVAL.format(problem=p)
        for _ in range(shots):
            prompts.append(question)
            ids.append(md5)
    batch['prompt'] = prompts
    batch['md5'] = ids

    outputs = sample(model, tokenizer, batch['prompt'], 512, **genargs)
    batch['outputs'] = outputs
    df = pd.DataFrame(batch)
    return df

from collections import defaultdict, Counter

def submit_example(ex_group, timeout):
    candidates = Counter()
    for ix, ex in ex_group.iterrows():
        success, result, _ = execute_code(ex['outputs'].replace('<|endoftext|>', '').strip()+ '\nprint("####", simple_math_problem())',
            timeout=timeout, 
            use_docker=True)
        ex['success'] = success
        ex['result'] = result
        # TODO reduce here 
        if result != '[invalid]':
            # normalize
            try:
                value = float(extract_answer(result))
                value = "#### %.02f" % value
            except ValueError:
                value = '[invalid]'  
        else:
            value = '[invalid]'

        candidates[value] += 1
    # reduce
    return {
        'question': ex['question'],
        'answer': ex['answer'],
        'success': len(candidates) > 0, 
        'result': candidates.most_common()[0][0]
    }
    
@torch.no_grad
def evaluate(model, tokenizer, shots, genargs):
    correct = 0
    total = 0
    model.eval()
    timeout = 60
    crashes = 0
    
    import torch.utils.data
    # batches = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    # batches = BucketIterator(ds, batch_size=8, device='cuda', sort_key=lambda x: len(tokenizer(x['question'][0])))

    # try to group questions of the same length
    buckets = defaultdict(list)
    for ex in ds:
        txt = ex['question']
        toks = tokenizer([txt], padding=False, return_tensors="pt")
        token_len = len(toks[0])
        buckets[token_len].append(ex)
        
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            futures = set()
            pbar = tqdm(total=len(ds), desc='creating batches')
            for bucket_len, bucket in buckets.items():
                batches = torch.utils.data.DataLoader(bucket, batch_size=8, shuffle=False) 
                for batch in batches:
                    future = executor.submit(
                        test_example,
                        batch,
                        model,
                        tokenizer,
                        shots,
                        genargs
                    )
                    futures.add(future)
                    pbar.update(len(batch))
                    
            pbar = tqdm(total=len(ds), desc='processing:')
            metrics = { "correct": 0, "crashes": 0, 'total':0}
            with ThreadPoolExecutor(max_workers=16) as executor_two:
                try:
                    for future in concurrent.futures.as_completed(futures):
                        futures_two = set()
                        df = future.result(timeout=timeout)
                        for id, ex_group in df.groupby('md5'):
                            task = executor_two.submit(
                                submit_example, ex_group, timeout
                            )
                            futures_two.add(task)
                            
                        for task in concurrent.futures.as_completed(futures_two):
                            try:
                                pbar.update(1)
                                metrics['total'] += 1
                                ex = task.result(timeout=timeout)
                                timeout = 60 # sec
                                success = ex['success'] == 0
                                if not success:
                                    metrics['crashes'] += 1
                                ok = is_correct(ex['result'], ex['answer'])
                                if ok: 
                                    metrics['correct'] += 1
                            except:
                                logging.error("could not evaluate", exc_info=True)   
                            
                            correct = metrics['correct']
                            total = metrics['total']
                            pbar.set_description("accuracy: %.2f" % (correct / float(total)))
                except TimeoutError:
                    logging.error('timeout', exc_info=True)
                except Exception as e:
                    logging.error('error: %s',e, exc_info=True)
        except Exception as e:
            logging.error('error: %s',e, exc_info=True)
    return { "accuracy": correct / float(total), "crashes": crashes }

 