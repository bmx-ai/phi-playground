import os
import math
import sys
import uuid
from pathlib import Path

from beartype import beartype
from beartype.typing import Optional, Dict, List, Tuple, Union, Callable, Type, Any
from torchtyping import TensorType

from contextlib import nullcontext, contextmanager
from functools import partial

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
import torch.nn.functional as F

import humanize
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from datasets import load_dataset

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

from einops import rearrange, repeat

from accelerate import Accelerator
from pytorch_warmup import LinearWarmup

from tqdm.auto import tqdm

from accelerate.logging import get_logger
logger = get_logger('finetune')


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def prompt_mask_from_len(length, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device=device) < rearrange(length, "... -> ... 1")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {humanize.naturalsize(trainable_params)} || all params: {humanize.naturalsize(all_param)} || trainable: {(100 * trainable_params / all_param):0.2f}%"
    )


def create_prompt(context, question, answer):
    if len(answer["text"]) < 1:
        answer = "Cannot Find Answer"
    else:
        answer = answer["text"][0]
    prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
    return prompt_template


ConstantLRScheduler = partial(LambdaLR, lr_lambda=lambda step: 1.0)


class OptimizerWithWarmupSchedule(Module):
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0,
        max_grad_norm: Optional[float] = None,
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm
        has_warmup = warmup_steps > 0

        self.warmup = (
            LinearWarmup(optimizer, warmup_period=warmup_steps) if has_warmup else None
        )

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(optimizer)

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(
            self.optimizer, self.scheduler
        )
        self.accelerator = accelerator

    def state_dict(self):
        pkg = dict(
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict()
        )

        if exists(self.warmup):
            pkg["warmup"] = self.warmup.state_dict()

        return pkg

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg["optimizer"])
        self.scheduler.load_state_dict(pkg["scheduler"])

        if exists(self.warmup):
            self.warmup.load_state_dict(pkg["warmup"])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if exists(self.max_grad_norm):
            for param_group in self.optimizer.param_groups:
                self.accelerator.clip_grad_norm_(
                    param_group["params"], self.max_grad_norm
                )

        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            context = nullcontext if not exists(self.warmup) else self.warmup.dampening

            with context():
                self.scheduler.step()


@contextmanager
def combine_contexts(a, b):
    with a() as c1, b() as c2:
        yield (c1, c2)


def model_forward_contexts(
    accelerator: Accelerator, model: Module, grad_accum_steps: int = 1
):
    for i in range(grad_accum_steps):
        is_last_step = i == grad_accum_steps - 1

        maybe_no_sync = (
            partial(accelerator.no_sync, model) if not is_last_step else nullcontext
        )

        yield partial(combine_contexts, accelerator.autocast, maybe_no_sync)


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params


def get_adam_optimizer(
    params,
    lr: float = 1e-4,
    wd: float = 1e-2,
    betas: Tuple[int, int] = (0.9, 0.99),
    eps: float = 1e-8,
    filter_by_requires_grad=False,
    omit_gammas_and_betas_from_wd=True,
    **kwargs,
):
    has_weight_decay = wd > 0.0

    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(lr=lr, betas=betas, eps=eps)

    if not has_weight_decay:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {"weight_decay": wd, **opt_kwargs}

    if not omit_gammas_and_betas_from_wd:
        return AdamW(params, **opt_kwargs)

    # there is an early practice where betas and gammas are omitted from weight decay in transformers
    # unsure whether it is really needed or not

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {"params": wd_params},
        {"params": no_wd_params, "weight_decay": 0},
    ]

    return AdamW(params, **opt_kwargs)


def adam_optimizer_with_linear_decay(
    model: Module,
    start_learning_rate: float,
    end_learning_rate: float,
    num_decay_steps: int,
    accelerator: Accelerator,
    weight_decay: float,
    adam_kwargs: dict = dict(),
) -> OptimizerWithWarmupSchedule:
    adam = get_adam_optimizer(
        model.parameters(), lr=start_learning_rate, wd=weight_decay
    )

    scheduler = None
    if start_learning_rate != end_learning_rate:
        scheduler = LinearLR

    return OptimizerWithWarmupSchedule(
        optimizer=adam,
        accelerator=accelerator,
        scheduler=scheduler,
        scheduler_kwargs=dict(
            start_factor=1.0,
            end_factor=end_learning_rate / start_learning_rate,
            total_iters=num_decay_steps,
        ),
    )


class HFDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return self.dset[idx]

    def __len__(self):
        return len(self.dset)


# This special index is used to ignore certain tokens during loss calculation.  (??)
IGNORE_INDEX = -100   

# def evaluate_batch_cross_entropy_loss(
#     model_output,
#     labels,
# ):
#     return F.cross_entropy(
#         rearrange(logits, "b n l -> b l n"), labels, ignore_index=IGNORE_INDEX
#     )

class GSMDataset(Dataset):
    def __init__(self, tokenizer, examples, max_seq_len=None):
        self._examples = examples
        self._max_length = max_seq_len
        self._tokenizer = tokenizer
        
    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        assert idx < len(self._examples)
        
        ex = self._examples[idx]
        
        qns = ex["question"]
        ans = ex["answer"]
        
        return self._tokenizer(qns + ans, padding=False)


class SFTTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size: int = 8,
        grad_accum_steps: int = 2,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        learning_rate_num_decay_steps: Optional[int] = None,
        weight_decay: float = 0.0,
        adam_kwargs: dict = dict(),
        valid_every: int = 1,
        collate_fn: Callable | None = None,
        train_storage_folder: Path | None = None
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model

        self.num_epochs = num_epochs

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
            collate_fn=collate_fn
        )
        
        self._id = str(uuid.uuid1()).split('-')[0]
         
        self.checkpoints_folder = Path(f'{train_storage_folder}/{self._id}/checkpoints')
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)
        
        self.num_train_steps = (
            len(self.train_dataloader) // grad_accum_steps * num_epochs
        )
        self.grad_accum_steps = grad_accum_steps

        (self.model, self.train_dataloader) = self.accelerator.prepare(
            self.model, self.train_dataloader
        )

        if not exists(learning_rate_num_decay_steps):
            # default learning rate decay num steps to half of training dataset length
            learning_rate_num_decay_steps = len(train_dataset) // 2

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps=learning_rate_num_decay_steps,
            accelerator=accelerator,
            weight_decay=weight_decay,
            adam_kwargs=adam_kwargs,
        )

        self.valid_every = valid_every

        self.valid_dataloader = None
        if exists(valid_dataset):
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

        self.steps = 0

    def log(self, **data):
        data['step'] = self.steps
        for k,v in data.items():
            print(k, ':', v)
        self.accelerator.log(data)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def forward(self):
        self.model.train()
        self.min_train_loss  = torch.inf
        
        train_dl_iter = cycle(self.train_dataloader)
        pbar = tqdm(range(self.num_train_steps), desc="training")
        for step in pbar:
            for forward_context in model_forward_contexts(
                self.accelerator, self.model, self.grad_accum_steps
            ):
                with forward_context():
                    prompt_and_mask = next(train_dl_iter)

                    #loss = self.get_cross_entropy_loss(prompt_and_mask)
                    output = self.model(**prompt_and_mask)
                    ce_batch_loss = output.loss
                    self.accelerator.backward(ce_batch_loss / self.grad_accum_steps)
            
            self.optimizer.step()
            metrics = {}
            # Collect metrics and check for NaN loss.
            # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
            if torch.isnan(ce_batch_loss):
                raise ValueError("nan loss encountered")
            #if z_batch_loss is not None and torch.isnan(z_batch_loss):
            #    raise ValueError("nan loss encountered")
            # for key, value in optim_metrics.items():
            #     metrics[f"optim/{key}"] = value.item()
            self.cur_train_loss = ce_batch_loss.item()
            self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
            metrics["train/cross-entropy-loss"] = self.cur_train_loss
            metrics["train/perplexity"] = math.exp(self.cur_train_loss)
            # if z_batch_loss is not None:
            #    metrics["train/ZLoss"] = z_batch_loss.item()
            self.log(**metrics) 
            self.optimizer.zero_grad()

            self.steps += 1

            # validate
            if exists(self.valid_dataloader) and not (step % self.valid_every) and self.steps > 1:
                self.wait()

                if self.accelerator.is_main_process:
                    total_valid_loss = 0.0
                    total_batches = 0.0

                    self.model.eval()

                    with torch.no_grad():
                        for validation_batch in tqdm(self.valid_dataloader, desc='validating'):
                            seq = validation_batch['input_ids']
                            batch = seq.shape[0]

                            ce_batch_loss = self.model(**validation_batch).loss

                            total_valid_loss += ce_batch_loss.item() * batch
                            total_batches += batch


                        valid_loss = total_valid_loss / total_batches

                    self.log(valid_loss=valid_loss)

                self.wait()
                self.model.train()
                
    def load_checkpoint(self, path):
        self.wait()
        if self.accelerator.is_main_process:
            path = self.checkpoints_folder / path 
            persisted_state = torch.load(path)
            self.model.load_state_dict(persisted_state['model'])
            self.optimizer.load_state_dict(persisted_state['optimizer'])
            self.steps = persisted_state['global_step']

        self.wait()
   
    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)
            
    def save(self, path: str, overwrite: bool = False):
        self.wait()

        if self.accelerator.is_main_process:
            logger.info('checkpointing:start')
            path = self.checkpoints_folder / (path + f'-{self.steps:04d}.ckpt')

            assert not path.exists() or overwrite, f'file already exists'

            pkg = dict(
                global_step = self.steps,
                optimizer = self.optimizer.state_dict(),
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))
            logger.info('checkpointing:end')
        self.wait()

from pytorch_microsoft_phi_model import PhiForCausalLM
#

def main(model_path="microsoft/phi-2", sft_dataset_name="gsm8k"):
    cuda = False
    dtype = torch.float16
    if torch.cuda.is_available():
        cuda = True
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = PhiForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        if cuda
        else None,
        torch_dtype=dtype,
    )

    # prepare for training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA
    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    accelerator.init_trackers('sft-train-%s' % sft_dataset_name)
 
    # collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to a single dictionary forming a batch { input_ids: [..], labels: [..], attention_mask: [..] }  
    def collate(examples):  
    
        # Extract input_ids from each element and find the maximum length among them 
        tokens = [e["input_ids"] for e in examples]  
        tokens_maxlen = max([len(t) for t in tokens])  
    
        for sample in examples:  
            input_ids = sample["input_ids"]  
            attention_mask = sample["attention_mask"]  
            
            labels = input_ids[1:]
    
            # Calculate the padding length required to match the maximum token length  
            pad_len = tokens_maxlen-len(input_ids)  
    
            # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0  
            input_ids.extend( pad_len * [tokenizer.pad_token_id] )  
            labels.extend( (pad_len + 1) * [IGNORE_INDEX] )  
            attention_mask.extend( pad_len * [0] )  
            
            sample['labels'] = labels
            
        # create and return batch with all the data in elements  
        batch={  
            "input_ids": torch.tensor( [e["input_ids"] for e in examples] ),  
            "labels": torch.tensor( [e["labels"] for e in examples] ),  
            "attention_mask": torch.tensor( [e["attention_mask"] for e in examples] ),  
        }  
        return batch
    
    ds = load_dataset(sft_dataset_name, 'main', split='train[:1%]')
    train_valid_ds = ds.train_test_split(test_size=0.1)

    train = SFTTrainer(
        model,
        accelerator=accelerator,
        train_dataset=GSMDataset(tokenizer, train_valid_ds["train"], max_seq_len=512),
        valid_dataset=GSMDataset(tokenizer, train_valid_ds["test"], max_seq_len=512),
        collate_fn=collate,
        batch_size=8,
        valid_every=len(train_valid_ds['train']), # every epoch
        train_storage_folder=Path("./train-model-weights")
    )
    
    train.save('phi-2-gsm8k') 

    train()
    
    train.save('phi-2-gsm8k') 
    # from transformers import TrainingArguments, Trainer  

    # bs=1         # batch size  
    # ga_steps=16  # gradient acc. steps  
    # epochs=20  
    # lr=0.00002  
    
    # steps_per_epoch=len(ds["train"])//(bs*ga_steps)  
    
    # args = TrainingArguments(  
    #     output_dir="out",  
    #     per_device_train_batch_size=bs,  
    #     per_device_eval_batch_size=16, 
    #     evaluation_strategy="steps",  
    #     logging_steps=1,  
    #     eval_steps=steps_per_epoch//2,      # eval twice per epoch  
    #     save_steps=steps_per_epoch,         # save once per epoch  
    #     gradient_accumulation_steps=ga_steps,  
    #     num_train_epochs=epochs,  
    #     lr_scheduler_type="constant",  
    #     optim="paged_adamw_32bit",      # val_loss will go NaN with paged_adamw_8bit  
    #     learning_rate=lr,  
    #     group_by_length=False,  
    #     bf16=True,  
    #     ddp_find_unused_parameters=False,  
    # )  
    
    # trainer = Trainer(  
    #     model=model,  
    #     tokenizer=tokenizer,  
    #     args=args,  
    #     data_collator=collate,  
    #     train_dataset=ds["train"],  
    #     eval_dataset=ds["test"],  
    # )  
    
    # print('training') 
    # trainer.train()


if __name__ == "__main__":
    main()
