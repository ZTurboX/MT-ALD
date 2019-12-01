from transformers import RobertaConfig,RobertaModel,RobertaTokenizer,BertConfig, BertModel, BertTokenizer,XLNetConfig,XLNetModel,XLNetTokenizer,XLNetForSequenceClassification
import torch

'''
config = RobertaConfig.from_pretrained("./roberta-base/roberta-base-config.json")
tokenizer = RobertaTokenizer.from_pretrained("./roberta-base/roberta-base-vocab.json")
model = RobertaModel.from_pretrained("./roberta-base/roberta-base-pytorch_model.bin", config=config)
'''

config = XLNetConfig.from_pretrained("./xlnet-base-cased/xlnet-base-cased-config.json")
tokenizer = XLNetTokenizer.from_pretrained("./xlnet-base-cased/xlnet-base-cased-spiece.model")
model = XLNetModel.from_pretrained("./xlnet-base-cased/xlnet-base-cased-pytorch_model.bin", config=config)

input_ids = torch.tensor(tokenizer.encode("toxicity")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]
print(last_hidden_states)
