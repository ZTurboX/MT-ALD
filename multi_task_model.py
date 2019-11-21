from transformers import (BertForSequenceClassification, BertTokenizer,BertModel,BertConfig)
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import logging
import json
logger = logging.getLogger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

class Multi_Model(nn.Module):
    def __init__(self,bert_config,bert_model):
        super(Multi_Model,self).__init__()

        self.config = bert_config
        self.bert=bert_model

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.attn=nn.Linear(self.config.hidden_size*2,1)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*2, 2)


    def task_specific_attention(self,task_embedding,encoder_outputs):
        attn_weight = torch.cat((task_embedding.permute(1, 0, 2).expand_as(encoder_outputs), encoder_outputs),dim=2)  # torch.Size([100, 100, 2000])
        attn_weight = F.softmax(self.attn(attn_weight), dim=1)
        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(1)
        return attn_applied

    def task_classifier(self,task_specific_atten,output,gold_labels):
        logits=torch.cat((task_specific_atten,output),dim=1)
        logits=self.classifier(logits)

        loss_fct = CrossEntropyLoss()
        logits=logits.view(-1, self.config.num_labels)
        loss = loss_fct(logits.view(-1, self.config.num_labels), gold_labels.view(-1))

        return logits,loss

    def forward(self, input_ids,attention_mask,token_type_ids,
                aggression_labels,attack_labels,toxicity_labels,
                aggression_embed,attack_embed,toxicity_embed):

        outputs=self.bert(input_ids,attention_mask,token_type_ids)
        hidden=outputs[0]#torch.Size([1, 6, 768])
        pooled_output=outputs[1]#torch.Size([1, 768])
        pooled_output = self.dropout(pooled_output)

        aggression_embedding=self.embedding(aggression_embed.unsqueeze(0))
        attack_embedding=self.embedding(attack_embed.unsqueeze(0))
        toxicity_embedding=self.embedding(toxicity_embed.unsqueeze(0))

        aggression_atten=self.task_specific_attention(aggression_embedding,hidden)
        attack_atten=self.task_specific_attention(attack_embedding,hidden)
        toxicity_atten=self.task_specific_attention(toxicity_embedding,hidden)


        aggression_logits,aggression_loss=self.task_classifier(aggression_atten,pooled_output,aggression_labels)
        attack_logits,attack_loss=self.task_classifier(attack_atten,pooled_output,attack_labels)
        toxicity_logits,toxicity_loss=self.task_classifier(toxicity_atten,pooled_output,toxicity_labels)

        loss=aggression_loss+attack_loss+toxicity_loss


        return aggression_logits,attack_logits,toxicity_logits,loss

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        #output_config_file = os.path.join(save_directory, CONFIG_NAME)
        #self.to_json_file(output_config_file)

        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def load_pretrained(self,path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

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
