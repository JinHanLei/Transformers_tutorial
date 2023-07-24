# -*- coding: utf-8 -*-
"""
Author  : Hanlei Jin
Date    : 2023/5/12
E-mail  : jin@smail.swufe.edu.cn
"""
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from peft import PeftModel

CKPTS = "THUDM/chatglm-6b"
model = AutoModel.from_pretrained(CKPTS, trust_remote_code=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(CKPTS, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./output/")
input_texts = ["你是谁？", "你是哪位？", "介绍一下你自己～", "你好"]
with torch.no_grad():
    for input_text in input_texts:
        answer, _ = model.chat(tokenizer, input_text)
        print(answer)