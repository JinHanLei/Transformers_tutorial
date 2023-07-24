# -*- coding: utf-8 -*-
"""
Author  : Hanlei Jin
Date    : 2023/7/20
E-mail  : jin@smail.swufe.edu.cn
"""
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, set_seed

set_seed(42)
raw_text = "Hello, I'm a language model,"
tokenizer = AutoTokenizer.from_pretrained("./gpt2/")
model = GPT2LMHeadModel.from_pretrained("./gpt2/")
inputs = tokenizer(raw_text, return_tensors="pt").input_ids
generated = model.generate(inputs, max_length=30)
res = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(res)