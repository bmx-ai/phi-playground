from beartype import beartype
from beartype.typing import Optional, Dict, List, Tuple, Union, Callable, Type
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

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.phi import PhiForCausalLM

from datasets import load_dataset

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

from accelerate import Accelerator
from pytorch_warmup import LinearWarmup

from tqdm.auto import tqdm

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    qa_dataset = load_dataset("squad_v2")


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


class SFTTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size: int = 16,
        grad_accum_steps: int = 2,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        learning_rate_num_decay_steps: Optional[int] = None,
        weight_decay: float = 0.0,
        ignore_index: int = -1,
        adam_kwargs: dict = dict(),
        valid_every: int = 1,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
        )

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
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

        self.steps = 0

    def log(self, **data):
        self.accelerator.log(data, step=self.steps)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def forward(self):
        self.model.train()

        train_dl_iter = cycle(self.train_dataloader)

        for step in tqdm(range(self.num_train_steps), desc="training"):
            for forward_context in model_forward_contexts(
                self.accelerator, self.model, self.grad_accum_steps
            ):
                with forward_context():
                    seq, prompt_len_or_mask = next(train_dl_iter)

                    loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)

                    self.accelerator.backward(loss / self.grad_accum_steps)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.log(loss=loss.item())

            self.steps += 1

            if exists(self.valid_dataloader) and not (step % self.valid_every):
                self.wait()

                if self.accelerator.is_main_process:
                    total_valid_loss = 0.0
                    total_batches = 0.0

                    self.model.eval()

                    with torch.no_grad():
                        for seq, prompt_len_or_mask in self.valid_dataloader:
                            batch = seq.shape[0]

                            loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)

                            total_valid_loss += loss.item() * batch
                            total_batches += batch

                    valid_loss = total_valid_loss / total_batches

                    self.log(valid_loss=valid_loss)

                self.wait()


def main(model_path="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = PhiForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # prepare for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=mapped_qa_dataset["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            max_steps=100,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()

    from datasets import load_dataset


if __name__ == "__main__":
    main()
