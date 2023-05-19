
# ChatGLM-Med: 基于中文医学知识的ChatGLM模型微调

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/SCIR-HI/Med-ChatGLM/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)


本项目开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的ChatGLM-6B模型。我们通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对ChatGLM-6B进行了指令微调，提高了ChatGLM在医疗领域的问答效果。

基于相同的数据，我们还训练了医疗版本的LLaMA模型: [华驼](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

## A Quick Start
Update: 由于chatglm更新很快，推荐使用chatglm官方的微调方案：https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md



首先安装依赖包，python环境建议3.9+

```
pip install -r requirements.txt
```
## 模型下载
训练好的模型参数可以通过如下方式下载：

| 模型名称              | 大小 |                                                                              模型下载地址                                                                              | 
| :----------------- | :------: |:----------------------------------------------------------------------------------------------------------------------------------------------------------------:| 
| ChatGLM-6B-Med   |    约13.4GB       | [[百度网盘]](https://pan.baidu.com/s/1Sfi1bRwV741GIChIEOUW0A?pwd=i73e)<br>[[GoogleDrive]](https://drive.google.com/drive/folders/1ZQSN56DloRGQ-Qj7IwzY4jV3ZHKMe9Bc)  | 


## 交互式测试
在安装好环境后，即可进行交互式测试：

```
python infer.py
```
## 数据集构建
我们采用了公开和自建的中文医学知识库，主要参考了[cMeKG](https://github.com/king-yyf/CMeKG_tools)。
医学知识库围绕疾病、药物、检查指标等构建，字段包括并发症，高危因素，组织学检查，临床症状，药物治疗，辅助治疗等。知识库示例如下:

```
{"中心词": "偏头痛", "相关疾病": ["妊娠合并偏头痛", "恶寒发热"], "相关症状": ["皮肤变硬", "头部及眼后部疼痛并能听到连续不断的隆隆声", "晨起头痛加重"], "所属科室": ["中西医结合科", "内科"], "发病部位": ["头部"]}
```
我们利用GPT3.5接口围绕医学知识库构建问答数据，设置了多种Prompt形式来充分利用知识。

指令微调的训练集数据示例如下：

```
"问题：一位年轻男性长期使用可卡因，突然出现胸痛、呕吐、出汗等症状，经检查发现心电图反映心肌急性损伤，请问可能患的是什么疾病？治疗方式是什么？"
回答: 可能患的是心肌梗塞，需要进行维拉帕米、依普利酮、硝酸甘油、ß阻滞剂、吗啡等药物治疗，并进行溶栓治疗、低分子量肝素、钙通道阻滞剂等辅助治疗。此外需要及时停用可卡因等药物，以防止病情加重。"
```
指令微调数据集质量仍有限，后续将进行不断迭代，同时医学知识库和数据集构建代码还在整理中，整理完成将会发布。

## Instruct-tuning 指令微调  
### 下载ChatGLM-6B原始参数
请参照[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，并将下载的模型放入`model`目录下。

### 指令微调

```
sh scripts/sft_medchat.sh
```
请根据实际情况调整batch size等超参，默认为我们使用的参数。

### 新数据指令微调

如果想用自己的数据集继续指令微调/指令精调（Instruct-tuning）ChatGLM，可以构建与`data`目录中训练集类似的文件格式进行。或参照[ChatGLM-Instruct-Tuning](https://github.com/thinksoso/ChatGLM-Instruct-Tuning/issues)仓库。
### 计算资源需求
我们在一张A100-SXM-80GB显卡上进行了训练，根据经验，训练显存建议选择32G及以上。

## 模型效果对比

| 测试输入 | ChatGLM-6B输出  | ChatGLM-6B-Med输出 |
|:------------- |:---------------| :-------------|
| 谷氨酰转肽酶水平会因吸毒或饮酒而升高吗?     | 吸毒和饮酒都会对身体的健康产生负面影响,包括对肝脏和心脏的影响。<br>饮酒可能导致肝脏损伤和心脏疾病,因为酒精会破坏肝脏和心脏的功能。长期饮酒还可能导致脂肪肝、肝硬化、心脏病等健康问题。<br>吸毒会对肝脏和心脏产生严重的损伤,导致肝脏纤维化、肝硬化、心脏血管疾病等。吸毒还可能导致精神障碍、社会退缩、暴力和犯罪等问题。<br>因此,无论是饮酒还是吸毒,都会对健康产生负面影响,应该避免过度使用。 | 谷氨酰转肽酶升高的原因有很多,包括感染、消化性溃疡、感染、药物副作用等,而吸毒或饮酒等不良行为可能会导致谷氨酰转肽酶的升高。 |
| 小李最近出现了心动过速的症状，伴有轻度胸痛。体检发现P-R间期延长，伴有T波低平和ST段异常    | 非常抱歉,我不理解您的问题。请提供更多背景信息或上下文,以便我更好地回答您的问题。谢谢! | 小李出现了心动过速的症状,并且伴有胸痛,需要考虑是否有心肌病、冠状动脉粥样硬化等心血管疾病,建议进行心电图检查、血液检查、心脏超声检查等 |
| ......     |         ...... |......|

## 常见问题
1. Q: 报错 ValueError: 130001 is not in list / ValueError: 150001 is not in list

   A: 由于相关依赖更新较快，版本的不同会导致一些bug

    (1) 如果报错为150001 is not in list，请将仓库更新至最新版本
    
    (2) 如果报错为130001 is not in list，请将仓库回退至commit为cb9d827的版本，链接为https://github.com/SCIR-HI/Med-ChatGLM/tree/cb9d82738021ec6f82b307d6031e8595a49dcb00
2. Q: 模型效果有限

   A: 由于ChatGLM本身训练策略没有开源，本仓库内提供的指令微调训练方法可能会对ChatGLM的基础能力带来负面影响。出于版权等因素考虑，未来我们的模型迭代会侧重开源模型。


## 项目参与者
本项目由哈尔滨工业大学社会计算与信息检索研究中心健康智能组[王昊淳](https://github.com/s65b40) 、[刘驰](https://github.com/thinksoso)完成，指导教师为[赵森栋](http://homepage.hit.edu.cn/stanzhao?lang=zh)副教授，秦兵教授以及刘挺教授。 

## 致谢

本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。

- ChatGLM: [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
- ChatGLM-Instruct-Tuning: [https://github.com/thinksoso/ChatGLM-Instruct-Tuning/issues](https://github.com/thinksoso/ChatGLM-Instruct-Tuning/issues)
- CMeKG: [https://github.com/king-yyf/CMeKG_tools](https://github.com/king-yyf/CMeKG_tools)

## 免责声明
本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。本项目数据集绝大部分由模型生成，即使符合某些医学事实，也不能被用作实际医学诊断的依据。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。


## Citation
如果你使用了本项目的数据或者代码，请声明引用

```
@misc{ChatGLM-Med,
  author={Haochun Wang, Chi Liu, Sendong Zhao, Bing Qin, Ting Liu},
  title = {ChatGLM-Med: 基于中文医学知识的ChatGLM模型微调},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SCIR-HI/Med-ChatGLM}},
}
```
