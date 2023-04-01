""" PyTorch ChatGLM Dataset. """

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones(
        (1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        # is equal to `seq_length = seq.index(150004)`
        seq_length = seq_len - 1
        position_ids = torch.arange(
            context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(
            context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


def chat_data_collator(features: list) -> dict:
    # 只对target的部分计算loss
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1)
            + ids[(seq_len - 1):]
            + [tokenizer.eos_token_id]
            + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.eos_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


class Chat_Dataset(Dataset):
    def __init__(self, data_dir, max_seq_length) -> None:
        super().__init__()
        self.content = self.load_json(data_dir)
        self.encoded_content = self.encode(
            tokenizer, self.content, max_seq_length)
        self.features = self.encoded_content[0].keys()

    def load_json(self, data_dir):
        content = []
        with open(data_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                content.append(json.loads(line))
        return content

    def __getitem__(self, index):
        return self.encoded_content[index]

    def __len__(self):
        return len(self.encoded_content)

    def get_ori_item(self, index):
        return self.content[index]

    def encode(self, tokenizer, content, max_seq_length):
        encoded_content = []
        for example in content:
            prompt = example["context"]
            target = example["target"]
            prompt_ids = tokenizer.encode(
                prompt, max_length=max_seq_length, truncation=True)
            target_ids = tokenizer.encode(
                target, max_length=max_seq_length, truncation=True, add_special_tokens=False
            )
            input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            encoded_content.append(
                {"input_ids": input_ids, "seq_len": len(prompt_ids)})
        return encoded_content
