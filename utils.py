import copy
import json
import logging
import os
from data_processor import Multi_Task_Processor,AggressionProcessor,AttackProcessor,ToxicityProcessor
import torch
from torch.utils.data import TensorDataset




processors = {"aggression": AggressionProcessor,"attack":AttackProcessor,"toxicity":ToxicityProcessor,"multi_task":Multi_Task_Processor}
output_modes = {"aggression": "classification", "attack":"classification","toxicity":"classification","multi_task":"classification"}

class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, aggression_label,attack_label,toxicity_label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.aggression_label = aggression_label
        self.attack_label=attack_label
        self.toxicity_label=toxicity_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

logger = logging.getLogger(__name__)
def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

    is_tf_dataset = False


    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            aggression_label = label_map[example.label[0]]
            attack_label=label_map[example.label[1]]
            toxicity_label=label_map[example.label[2]]

        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("aggression_label: %s (id = %d)" % (example.label[0], aggression_label))
            logger.info("attack_label: %s (id = %d)" % (example.label[1], attack_label))
            logger.info("toxicity_label: %s (id = %d)" % (example.label[2], toxicity_label))


        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              aggression_label=aggression_label, attack_label=attack_label,toxicity_label=toxicity_label))



    return features

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format('dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(), str(args.max_seq_length), str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()


        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples, tokenizer, label_list=label_list,
                                                max_length=args.max_seq_length, output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    aggression_all_label=torch.tensor([f.aggression_label for f in features], dtype=torch.long)
    attack_all_label_label=torch.tensor([f.attack_label for f in features], dtype=torch.long)
    toxicity_all_label=torch.tensor([f.toxicity_label for f in features], dtype=torch.long)


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, aggression_all_label,attack_all_label_label,toxicity_all_label)
    return dataset



glue_processors = {
    "multi_task": Multi_Task_Processor,

}

glue_output_modes = {
    "multi_task": "classification",

}

def get_char_vocab():
    vocab=['<p>']
    labels=["aggression","attack","toxicity"]
    for word in labels:
        for char in word:
            if char not in vocab:
                vocab.append(char)
    char_vocab={}
    for ids,item in enumerate(vocab):
        char_vocab[item]=ids
    return char_vocab

def char2ids(task_label,char_vocab):
    labels = ["aggression", "attack", "toxicity"]
    max_len=[len(label) for label in labels]
    max_len=max(max_len)
    task_label_char_ids=[char_vocab[char] for char in task_label]+[char_vocab['<p>']]*(max_len-len(task_label))
    return task_label_char_ids

if __name__=="__main__":
    '''
    char_vocab=get_char_vocab()
    print(char_vocab)
    task_label_char_ids=char2ids("attack",char_vocab)
    print(task_label_char_ids)
    '''




