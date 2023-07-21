# -*- coding: utf-8 -*-
"""
Author  : Hanlei Jin
Date    : 2023/7/15
E-mail  : jin@smail.swufe.edu.cn
"""
import copy
import json


class GPT2Config:
    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=50256,
            eos_token_id=50256,
            architectures=None,
            task_specific_params=None,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.architectures = architectures
        self.task_specific_params = task_specific_params
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    model_type = "gpt2"

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self) -> str:
        output = copy.deepcopy(self.__dict__)
        self.dict2str(output)
        return json.dumps(output, indent=2, sort_keys=True) + "\n"

    def dict2str(self, d) -> None:
        for value in d.values():
            if isinstance(value, dict):
                self.dict2str(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        with open(pretrained_model_path + "config.json", "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        config = cls(**config_dict)
        return config
