# -*- coding: utf-8 -*-
"""
Author  : Hanlei Jin
Date    : 2023/7/14
E-mail  : jin@smail.swufe.edu.cn
"""
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(history):
        bot_message, _ = model.chat(tokenizer, history[-1][0], history=[])
        history[-1][0] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history


    def bot_stream(history):
        for character in model.stream_chat(tokenizer, history[-1][0], history=[]):
            history[-1][1] = character[1][0][-1]
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_stream, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()
