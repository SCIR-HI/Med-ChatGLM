import torch
from transformers import AutoTokenizer, AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained(
    "./model", trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(
    "./model").half().cuda()
while True:
    a = input("请输入您的问题：（输入q以退出）")
    if a.strip() == 'q':
        exit()
    response, history = model.chat(tokenizer, "问题：" + a.strip() + '\n答案：', max_length=256, history=[])
    print("回答：", response)
