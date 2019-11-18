

import os
import csv
import sys

from transformers import InputExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class AggressionProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_examples,_=self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        _,dev_examples, = self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")
        return dev_examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        train_examples = []
        dev_examples=[]
        for  i in range(len(lines)):
            if i>0:
                line=lines[i]
                guid = "%s-%s" % (set_type, i)
                text_a = ''.join(line[1:-8])
                text_a = text_a.replace("NEWLINE_TOKEN", "")
                text_a = text_a.replace("TAB_TOKEN", "")
                mode=line[-4]
                label = line[-3]
                '''
                attack=line[-2]
                toxicity=line[-1]
                '''
                if label=='True':
                    label='1'
                elif label=='False':
                    label='0'
                if mode=='train':
                    train_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                elif mode=='dev':
                    dev_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return train_examples,dev_examples



class AttackProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_examples,_=self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        _,dev_examples, = self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")
        return dev_examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        train_examples = []
        dev_examples=[]
        for  i in range(len(lines)):
            if i>0:
                line=lines[i]
                guid = "%s-%s" % (set_type, i)
                text_a = ''.join(line[1:-8])
                text_a = text_a.replace("NEWLINE_TOKEN", "")
                text_a = text_a.replace("TAB_TOKEN", "")
                mode=line[-4]
                label = line[-2]
        
                if label=='True':
                    label='1'
                elif label=='False':
                    label='0'
                if mode=='train':
                    train_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                elif mode=='dev':
                    dev_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return train_examples,dev_examples

class ToxicityProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_examples,_=self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        _,dev_examples, = self._create_examples(self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")
        return dev_examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        train_examples = []
        dev_examples=[]
        for  i in range(len(lines)):
            if i>0:
                line=lines[i]
                guid = "%s-%s" % (set_type, i)
                text_a = ''.join(line[1:-8])
                text_a = text_a.replace("NEWLINE_TOKEN", "")
                text_a = text_a.replace("TAB_TOKEN", "")
                mode=line[-4]
                label = line[-1]
    
                if label=='True':
                    label='1'
                elif label=='False':
                    label='0'
                if mode=='train':
                    train_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                elif mode=='dev':
                    dev_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return train_examples,dev_examples
