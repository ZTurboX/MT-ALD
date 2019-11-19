from transformers import (BertForSequenceClassification, BertTokenizer,BertModel,BertConfig)
import torch.nn as nn
import torch

'''
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
model = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, logits = outputs[:2]
print("loss:",loss)
print("logits:",logits)
'''




class Multi_Model(nn.Module):
    def __init__(self):
        super(Multi_Model,self).__init__()

        self.config = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.bert=BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin',config=self.config)

        self.embedding = nn.Embedding(30522, 25)



    def forward(self, input,aggression_label,attack_label,toxicity_label):

        outputs=self.bert(input)
        hidden=outputs[0]#torch.Size([1, 6, 768])
        pooler_output=outputs[1]#torch.Size([1, 768])

        aggression_embedding=self.embedding(aggression_label)
        attack_embedding=self.embedding(attack_label)
        toxicity_embedding=self.embedding(toxicity_label)


        return hidden,pooler_output

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt',do_lower_case=True)
    model=Multi_Model()
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
    aggression_label=torch.tensor(tokenizer.encode("aggression"))
    attack_label = torch.tensor(tokenizer.encode("attack"))
    toxicity_label = torch.tensor(tokenizer.encode("toxicity"))
    hidden,pooler_outputs=model(input_ids,aggression_label,attack_label,toxicity_label)
    print(hidden)
    print(pooler_outputs)