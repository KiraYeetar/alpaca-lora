import transformers
import json
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict
from fastchat.conversation import get_default_conv_template, SeparatorStyle
from transformers.trainer_pt_utils import LabelSmoother
from typing import Union

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# conv_vicuna_v1_1 = Conversation(
#     system="A chat between a curious user and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     roles=("USER", "ASSISTANT"),
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.ADD_COLON_TWO,
#     sep=" ",
#     sep2="</s>",
# )

# elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
#     seps = [self.sep, self.sep2]
#                 ret = self.system + seps[0]
#                 for i, (role, message) in enumerate(self.messages):
#                     if message:
#                         ret += role + ": " + message + seps[i % 2]
#                     else:
#                         ret += role + ":"
#                 return ret

""" e.g.
conv.append_message = [["USER", "hello"], ["ASSISTANT", "hello!!!!"]]
->
ret = {system} + {seps[0]} + 
        "USER" + ": " + "hello" + {seps[0]} + 
        "ASSISTANT" + ": " + "hello!!!!" + {seps[1]}
    = f"{system} USER: hello ASSISTANT: hello!!!!</s>"
"""
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]  # ("USER", "ASSISTANT")
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "

    # [[对话1], [对话2]...]       [[ids1], [ids2]...]
    for conversation, target in zip(conversations, targets):
        # 非填充字符的总数
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)

        # 用 -100 填充
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            # 用 " ASSISTANT: " 切分单次对话
            # f"{system} USER: hello ASSISTANT: hello!!!!</s>"
            # -> "{system} USER: hello" 与 "hello!!!!</s>"
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # "{system} USER: hello "
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # 回答前的 id 都替换为 -100
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += round_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            # 如果填 -100 后的分词数不匹配，则直接全部填-100
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_origin_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def alpaca_generate_prompt(
        instruction: str,
        label: Union[None, str] = None
):
    conv = get_default_conv_template("vicuna").copy()
    seps = [conv.sep, conv.sep2]
    system = conv.system + seps[0] + \
            "USER" + ": " + "{instruction}" + seps[0] + \
            "ASSISTANT" + ": "
    res = system.format(instruction=instruction)
    if label:
        res = f"{res}{label}"
    else:
        sep = conv.sep + "ASSISTANT" + ": "
        res = res.split(res, sep)[0]
    return res


def make_alpaca_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
):
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        result["labels"] = result["input_ids"].clone()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = alpaca_generate_prompt(
            data_point["instruction"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = alpaca_generate_prompt(
            data_point["instruction"],
        )
        tokenized_user_prompt = tokenize(
            user_prompt
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = torch.cat([
            torch.from_numpy(np.array([IGNORE_TOKEN_ID] * user_prompt_len)),
            tokenized_full_prompt["labels"][user_prompt_len:]]
        )
        tokenized_full_prompt["attention_mask"] = tokenized_full_prompt["input_ids"].ne(tokenizer.pad_token_id)
        return tokenized_full_prompt

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    train_val = data["train"].train_test_split(test_size=0.98, shuffle=True, seed=42)
    train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
