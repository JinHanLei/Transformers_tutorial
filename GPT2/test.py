# -*- coding: utf-8 -*-
"""
Author  : Hanlei Jin
Date    : 2023/7/20
E-mail  : jin@smail.swufe.edu.cn
"""
import torch
from torch import nn
from tqdm import trange
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, set_seed

set_seed(42)
raw_text = "Hello, I'm a language model,"
tokenizer = AutoTokenizer.from_pretrained("./gpt2/")
model = GPT2LMHeadModel.from_pretrained("./gpt2/")
inputs = tokenizer(raw_text, return_tensors="pt").input_ids

MAX_LEN = 30
generated = inputs
repetition_penalty = 10.
temperature = 100
top_k = 10
for _ in range(MAX_LEN):
    logits = model(generated).logits
    next_token_logits = nn.functional.softmax(logits[0, -1, :], dim=-1)
    for id in set(generated[0]):
        next_token_logits[id] /= repetition_penalty
    next_token_logits /= temperature
    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1]
    next_token_logits[indices_to_remove] = -float('Inf')
    next_token_logits = nn.functional.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(next_token_logits, num_samples=1)
    if next_token.cpu().numpy()[0] == model.config.eos_token_id:
        break
    generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

res = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(res)