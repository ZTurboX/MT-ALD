
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel,RobertaModel,XLNetModel,RobertaForSequenceClassification,XLNetForSequenceClassification
import logging

logger = logging.getLogger(__name__)

try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """
        def __init__(self, *args, **kwargs):
            super(Identity, self).__init__()

        def forward(self, input):
            return input



class SequenceSummary(nn.Module):
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':

            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):

        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class Multi_Model(nn.Module):
    def __init__(self,args,bert_config):
        super(Multi_Model,self).__init__()

        self.model_type=args.model_type

        self.config = bert_config
        self.hidden_size=self.config.hidden_size if self.model_type in ["bert",'roberta'] else self.config.d_model
        if self.model_type=="bert":
            self.transformer=BertModel.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),config=self.config)
            self.embedding = nn.Embedding(self.config.vocab_size, self.hidden_size // 2)
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        elif self.model_type=="roberta":
            self.transformer = RobertaModel.from_pretrained(args.model_name_or_path,from_tf=bool('.ckpt' in args.model_name_or_path), config=self.config)
            self.embedding = nn.Embedding(3, self.hidden_size // 2)
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        elif self.model_type=="xlnet":
            self.transformer = XLNetModel.from_pretrained(args.model_name_or_path,from_tf=bool('.ckpt' in args.model_name_or_path),config=self.config)
            self.embedding = nn.Embedding(self.config.n_token, self.hidden_size // 2)
            self.dropout = nn.Dropout(self.config.dropout)
            self.sequence_summary = SequenceSummary(self.config)


        self.char_embedding=nn.Embedding(14,self.hidden_size//2)
        self.convs1=nn.ModuleList([nn.Conv2d(1, 192, (K, self.hidden_size//2), padding=(K-1, 0)) for K in [2,3]])
        self.attn=nn.Linear(self.hidden_size*2,1)


        self.classifier = nn.Linear(self.hidden_size*2, 2)

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
        return attn_applied,attn_weight



    def task_classifier(self,task_specific_atten,output,gold_labels):
        logits=torch.cat((task_specific_atten,output),dim=1)#torch.Size([16, 1536])
        logits=self.classifier(logits)#torch.Size([16, 2])

        loss_fct = CrossEntropyLoss()
        logits=logits.view(-1, self.config.num_labels)
        loss = loss_fct(logits.view(-1, self.config.num_labels), gold_labels.view(-1))


        return logits,loss



    def forward(self, input_ids,attention_mask,token_type_ids,
                aggression_labels,attack_labels,toxicity_labels,
                aggression_tensor,attack_tensor,toxicity_tensor,
                aggression_char_tensor,attack_char_tensor,toxicity_char_tensor):

        outputs=self.transformer(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

        hidden=outputs[0]#torch.Size([1, 6, 768])
        if self.model_type in ["bert","roberta"]:
            pooled_output=outputs[1]#torch.Size([1, 768])
        elif self.model_type=="xlnet":
            pooled_output = self.sequence_summary(hidden)
        pooled_output = self.dropout(pooled_output)

        if self.all_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)
            attack_embedding = self.word_char_embedding(attack_tensor, attack_char_tensor)
            toxicity_embedding = self.word_char_embedding(toxicity_tensor, toxicity_char_tensor)


            aggression_atten,aggression_weight = self.task_specific_attention(aggression_embedding, hidden)
            attack_atten,attack_weight = self.task_specific_attention(attack_embedding, hidden)
            toxicity_atten,toxicity_weight = self.task_specific_attention(toxicity_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            attack_logits, attack_loss = self.task_classifier(attack_atten, pooled_output, attack_labels)
            toxicity_logits, toxicity_loss = self.task_classifier(toxicity_atten, pooled_output, toxicity_labels)

            all_loss=0.33*aggression_loss+0.33*attack_loss+0.33*toxicity_loss

            return aggression_logits, attack_logits, toxicity_logits, all_loss,aggression_weight,attack_weight,toxicity_weight

        elif self.aggression_attack_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)
            attack_embedding = self.word_char_embedding(attack_tensor, attack_char_tensor)


            aggression_atten = self.task_specific_attention(aggression_embedding, hidden)
            attack_atten = self.task_specific_attention(attack_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            attack_logits, attack_loss = self.task_classifier(attack_atten, pooled_output, attack_labels)

            aggression_attack_loss=0.5*aggression_loss+0.5*attack_loss

            return aggression_logits, attack_logits, aggression_attack_loss

        elif self.aggression_toxicity_task:
            aggression_embedding = self.word_char_embedding(aggression_tensor, aggression_char_tensor)

            toxicity_embedding = self.word_char_embedding(toxicity_tensor, toxicity_char_tensor)

            aggression_atten = self.task_specific_attention(aggression_embedding, hidden)
            toxicity_atten = self.task_specific_attention(toxicity_embedding, hidden)

            aggression_logits, aggression_loss = self.task_classifier(aggression_atten, pooled_output,aggression_labels)
            toxicity_logits, toxicity_loss = self.task_classifier(toxicity_atten, pooled_output, toxicity_labels)

            aggression_toxicity_loss=0.5*aggression_loss+0.5*toxicity_loss

            return aggression_logits, toxicity_logits, aggression_toxicity_loss

        elif self.attack_toxicity_task:

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
