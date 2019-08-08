# -*- coding: utf-8 -*-

import pandas as pd
from data_collections.settings import *



agressive_comments = pd.read_csv(wiki_agression+"/aggression_annotated_comments.tsv", sep = '\t', index_col = 0)
agressive_annotations = pd.read_csv(wiki_agression+"/aggression_annotations.tsv",  sep = '\t')

attack_comments = pd.read_csv(wiki_attack+"/attack_annotated_comments.tsv", sep = '\t', index_col = 0)
attack_annotations = pd.read_csv(wiki_attack+"/attack_annotations.tsv",  sep = '\t')

toxic_comments = pd.read_csv(wiki_toxic+"/toxicity_annotated_comments.tsv", sep = '\t', index_col = 0)
toxic_annotations = pd.read_csv(wiki_toxic+"/toxicity_annotations.tsv",  sep = '\t')



agressive_labels = agressive_annotations.groupby('rev_id')['aggression'].mean() > 0.5
agressive_comments['aggression'] = agressive_labels

attack_labels = attack_annotations.groupby('rev_id')['attack'].mean() > 0.5
attack_comments['attack'] = attack_labels

toxic_labels = toxic_annotations.groupby('rev_id')['toxicity'].mean() > 0.5
toxic_comments['toxicity'] = toxic_labels

all_data_combine=agressive_comments.join([attack_comments['attack'],toxic_comments['toxicity']])


all_data=all_data_combine[all_data_combine.toxicity.notnull()]

all_data.to_csv(wiki_data)



