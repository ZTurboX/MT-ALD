
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)


class Multi_Model(nn.Module):
    def __init__(self,args,bert_config):
        super(Multi_Model,self).__init__()

        self.config = bert_config
        self.bert=BertModel.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),config=self.config)
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size//2)
        self.char_embedding=nn.Embedding(14,self.config.hidden_size//2)
        self.convs1=nn.ModuleList([nn.Conv2d(1, 192, (K, self.config.hidden_size//2), padding=(K-1, 0)) for K in [2,3]])
        self.attn=nn.Linear(self.config.hidden_size*2,1)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, 2)
        self.mPos = args.mPos
        self.mNeg = args.mNeg
        self.gamma = args.gamma

        self.all_task=args.all_task
        self.aggression_attack_task=args.aggression_attack_task
        self.aggression_toxicity_task=args.aggression_toxicity_task
        self.attack_toxicity_task=args.attack_toxicity_task


    def word_char_embedding(self,task_word_tensor,task_char_tensor):
        task_char_tensor=task_char_tensor.unsqueeze(0)

        task_char_embedding=self.char_embedding(task_char_tensor)
        task_char_embedding=task_char_embedding.unsqueeze(1)
        task_char_embedding = [F.relu(conv(task_char_embedding)).squeeze(3) for conv in self.convs1]
        task_char_embedding = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in task_char_embedding]
        task_char_embedding = torch.cat(task_char_embedding, 1)

        task_char_embedding=task_char_embedding.unsqueeze(0)
        task_word_embedding=self.embedding(task_word_tensor.unsqueeze(0))
        emnedding_list=[task_word_embedding,task_char_embedding]

        word_vecs=torch.cat(emnedding_list,2)

        return word_vecs


    def task_specific_attention(self,task_embedding,encoder_outputs):
        attn_weight = torch.cat((task_embedding.permute(1, 0, 2).expand_as(encoder_outputs), encoder_outputs),dim=2)
        attn_weight = F.softmax(F.selu(self.attn(attn_weight)), dim=1)
        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(1)
        return attn_applied

    def  ranking_loss(self,logit,target):
        val,ind=torch.topk(logit,2,dim=1)
        noneOtherInd=target!=0
        rows=torch.tensor(list(range(len(logit))))
        positive_score=logit[rows,target]

        positive_loss = torch.log(1 + torch.exp(self.gamma * (self.mPos - positive_score))) + \
                torch.log(1 + torch.exp(self.gamma * (-100 + positive_score)))  # positive loss
        predT = ind[:, 0] == target
        predF = ind[:, 0] != target

        negative_loss = torch.log(1 + torch.exp(self.gamma * (self.mNeg + val))) + \
                torch.log(1 + torch.exp(self.gamma * (-100 - val)))  # negative loss
        negative_loss = torch.dot(predT.float(), negative_loss[:, 1]) + torch.dot(predF.float(), negative_loss[:, 0])
        loss = torch.dot(noneOtherInd.float(), positive_loss) + negative_loss  # exclusive other loss
        return loss / len(target)

    def task_classifier(self,task_specific_atten,output,gold_labels):
        logits=torch.cat((task_specific_atten,output),dim=1)#torch.Size([16, 1536])
        logits=self.classifier(logits)#torch.Size([16, 2])

        loss_fct = CrossEntropyLoss()
        logits=logits.view(-1, self.config.num_labels)
        loss = loss_fct(logits.view(-1, self.config.num_labels), gold_labels.view(-1))


        #loss=self.ranking_loss(logits,gold_labels)

        return logits,loss



    def forward(self, input_ids,attention_mask,token_type_ids,
                aggression_labels,attack_labels,toxicity_labels,
                aggression_tensor,attack_tensor,toxicity_tensor,
                aggression_char_tensor,attack_char_tensor,toxicity_char_tensor):

        outputs=self.bert(input_ids,attention_mask,token_type_ids)
        hidden=outputs[0]#torch.Size([1, 6, 768])
        pooled_output=outputs[1]#torch.Size([1, 768])
        pooled_output = self.dropout(pooled_output)




        if self.all_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)
            attack_embedding = self.word_char_embedding(attack_tensor, attack_char_tensor)
            toxicity_embedding = self.word_char_embedding(toxicity_tensor, toxicity_char_tensor)


            aggression_atten = self.task_specific_attention(aggression_embedding, hidden)
            attack_atten = self.task_specific_attention(attack_embedding, hidden)
            toxicity_atten = self.task_specific_attention(toxicity_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            attack_logits, attack_loss = self.task_classifier(attack_atten, pooled_output, attack_labels)
            toxicity_logits, toxicity_loss = self.task_classifier(toxicity_atten, pooled_output, toxicity_labels)

            all_loss=0.33*aggression_loss+0.33*attack_loss+0.33*toxicity_loss

            return aggression_logits, attack_logits, toxicity_logits, all_loss

        if self.aggression_attack_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)
            attack_embedding = self.word_char_embedding(attack_tensor, attack_char_tensor)


            aggression_atten = self.task_specific_attention(aggression_embedding, hidden)
            attack_atten = self.task_specific_attention(attack_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            attack_logits, attack_loss = self.task_classifier(attack_atten, pooled_output, attack_labels)

            aggression_attack_loss=0.5*aggression_loss+0.5*attack_loss

            return aggression_logits, attack_logits, aggression_attack_loss

        if self.aggression_toxicity_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)

            toxicity_embedding = self.word_char_embedding(toxicity_tensor, toxicity_char_tensor)

            aggression_atten = self.task_specific_attention(aggression_embedding, hidden)
            toxicity_atten = self.task_specific_attention(toxicity_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            toxicity_logits, toxicity_loss = self.task_classifier(toxicity_atten, pooled_output, toxicity_labels)

            aggression_toxicity_loss=0.5*aggression_loss+0.5*toxicity_loss

            return aggression_logits, toxicity_logits, aggression_toxicity_loss

        if self.attack_toxicity_task:

            attack_embedding = self.word_char_embedding(attack_tensor, attack_char_tensor)
            toxicity_embedding = self.word_char_embedding(toxicity_tensor, toxicity_char_tensor)

            attack_atten = self.task_specific_attention(attack_embedding, hidden)
            toxicity_atten = self.task_specific_attention(toxicity_embedding, hidden)

            attack_logits, attack_loss = self.task_classifier(attack_atten, pooled_output, attack_labels)
            toxicity_logits, toxicity_loss = self.task_classifier(toxicity_atten, pooled_output, toxicity_labels)

            attack_loss_toxicity_loss=0.5*attack_loss+0.5*toxicity_loss

            return attack_logits,toxicity_logits,attack_loss_toxicity_loss






if __name__=='__main__':
    '''
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt',do_lower_case=True)
    model=Multi_Model()
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
    aggression_label=torch.tensor(tokenizer.encode("aggression"))
    attack_label = torch.tensor(tokenizer.encode("attack"))
    toxicity_label = torch.tensor(tokenizer.encode("toxicity"))
    gold_labels=[1,0,0]
    aggression_logits, attack_logits, toxicity_logits, loss=model(input_ids,aggression_label,attack_label,toxicity_label,gold_labels)
    '''
    pass
