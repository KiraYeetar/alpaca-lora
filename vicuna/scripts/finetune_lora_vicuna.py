import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))

from dataclasses import dataclass, field
import pathlib
import typing

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, HfArgumentParser, TrainingArguments
from data_module._vicuna import make_origin_supervised_data_module, make_alpaca_data_module
from _hidden_funcs_ import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()


@dataclass
class LoadArguments:
    base_model: str = field(default="")
    retrain_model: str = field(default="")
    save_path: str = field(default="output")
    base_model_type: str = field(default="vicuna", metadata={"help": "llama | chatglm | vicuna"})
    data_path: str = field(default="")
    data_type: str = field(default="alpaca", metadata={"help": "alpaca | chatglm | vicuna"})
    lazy_preprocess: bool = False
    max_length: int = 2048


@dataclass
class LoraArguments:
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_bias: str = field(default="none")
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    parser = HfArgumentParser((LoadArguments, LoraArguments, TrainingArguments))
    (load_args, lora_args, training_args) = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(load_args.base_model)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        load_args.base_model,
        model_max_length=load_args.max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if load_args.data_type == "alpaca":
        data_module = make_alpaca_data_module(
            tokenizer=tokenizer,
            data_args=load_args,
        )
    else:
        data_module = make_origin_supervised_data_module(
            tokenizer=tokenizer,
            data_args=load_args,
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        args=training_args,
    )
    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(model.state_dict(), lora_args.lora_bias)
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
