## **Transformer-based model** tutorial

[transformers](https://huggingface.co/docs/transformers/index)库为我们提供了很好的NLP模型调用范例。然而，随着模型量的增多，其代码越来越复杂和抽象，不利于初学者学习和理解。

本项目旨在脱离transformers库，完全用PyTorch底层实现各个模型的代码，并力求适配transformers库的类、方法和超参。计划中的模型包括：

- Transformer类，如BART、T5等
- BERT类，如BERT、RoBerta、DeBerta等
- GPT类，如GPT2等

将模型结构原原本本展现在您面前，便于您的学习和魔改。

## Quick Start

1. 下载本仓库并配置必要环境

```shell
git clone https://github.com/JinHanLei/Transformers_tutorial && cd Transformers_tutorial
pip install -r requirements.txt
```

2. 进入模型各自的文件夹，运行main.py

本仓库正在快速更新中，请关注我或者star本项目，及时获取更新状态！

## Dependencies

- python==3.9
- PyTorch==2.0.0
- transformers==4.31.0
- gradio
- sentencepiece
- protobuf
- cpm-kernels

## Author

- Jinhanlei: jin@smail.swufe.edu.cn
