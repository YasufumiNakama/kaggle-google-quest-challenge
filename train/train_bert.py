#===========================================================
# Base code from https://www.kaggle.com/phoenix9032/pytorch-bert-plain
#===========================================================
import os
import sys
import gc
import time
import glob
import multiprocessing
import re
from urllib.parse import urlparse
from tqdm import tqdm
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import math
from math import floor, ceil
import random

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.model_selection import GroupKFold
import category_encoders as ce
import re
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import transformers
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_bert import BertPreTrainedModel 


#===========================================================
# Utils
#===========================================================
def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    logger.info(f'[{name}] start')
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#===========================================================
# Config
#===========================================================
class PipeLineConfig:
    def __init__(self, lr, warmup, accum_steps, epochs, seed, expname, 
                 head_tail, head, freeze, question_weight, answer_weight, fold, train, cv, test):
        self.lr = lr
        self.warmup = warmup
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.expname = expname
        self.head_tail = head_tail
        self.head = head
        self.freeze = freeze
        self.question_weight = question_weight
        self.answer_weight = answer_weight
        self.fold = fold
        self.train = train
        self.cv = cv
        self.test = test

config = PipeLineConfig(lr=1e-4, warmup=0.1, accum_steps=1, epochs=5,
                        seed=42, expname='uncased_7', head_tail=True, head=0.5, freeze=False,
                        question_weight=0., answer_weight=0., fold=5, train=True, cv=True, test=True)

DEBUG = False
ID = 'qa_id'
target_cols = ['question_asker_intent_understanding', 'question_body_critical', 
               'question_conversational', 'question_expect_short_answer', 
               'question_fact_seeking', 'question_has_commonly_accepted_answer', 
               'question_interestingness_others', 'question_interestingness_self', 
               'question_multi_intent', 'question_not_really_a_question', 
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity', 
               'question_type_instructions', 'question_type_procedure', 
               'question_type_reason_explanation', 'question_type_spelling', 
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 
               'answer_relevance', 'answer_satisfaction', 
               'answer_type_instructions', 'answer_type_procedure', 
               'answer_type_reason_explanation', 'answer_well_written']
NUM_FOLDS = config.fold
#ROOT = '../input/google-quest-challenge/'
ROOT = '../input/'
SEED = config.seed
seed_everything(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#MODEL_DIR = '../input/googlequestchallenge-exp/'
MODEL_DIR = './'
COMBINE_INPUT = False
T_MAX_LEN = 30
Q_MAX_LEN = 479 # 382
A_MAX_LEN = 479 # 254 
MAX_SEQUENCE_LENGTH = T_MAX_LEN + Q_MAX_LEN + A_MAX_LEN + 4
q_max_sequence_length = T_MAX_LEN + Q_MAX_LEN + 3
a_max_sequence_length = T_MAX_LEN + A_MAX_LEN + 3

#===========================================================
# Model
#===========================================================
def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        print(f'len(tokens): {len(tokens)}')
        print(f'max_seq_length: {max_seq_length}')
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
        
    segments = []
    first_sep = True
    current_segment_id = 0
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def _trim_input(tokenizer, title, question, answer, max_sequence_length, t_max_len, q_max_len, a_max_len):
    
    # 350+128+30 = 508 +4 = 512
    
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))
        # Head+Tail method 
        q_len_head = round(q_new_len * config.head)
        q_len_tail = -1 * (q_new_len - q_len_head)
        a_len_head = round(a_new_len * config.head)
        a_len_tail = -1 * (a_new_len - a_len_head)
        t_len_head = round(t_new_len * config.head)
        t_len_tail = -1 * (t_new_len - t_len_head)  
        #t = t[:t_new_len]
        if config.head_tail :
            q = q[:q_len_head]+q[q_len_tail:]
            a = a[:a_len_head]+a[a_len_tail:]
            #t = t[:t_len_head]+t[t_len_tail:]
            t = t[:t_new_len]
        else:
            # No Head+Tail , usual processing
            q = q[:q_new_len]
            a = a[:a_new_len]
            t = t[:t_new_len]
    
    return t, q, a


def q_trim_input(tokenizer, title, question, q_max_sequence_length, t_max_len, q_max_len):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)

    t_len = len(t)
    q_len = len(q)

    if (t_len+q_len+3) > q_max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            q_max_len = q_max_len + (t_max_len - t_len)
        else:
            t_new_len = t_max_len

        if q_max_len > q_len:
            q_new_len = q_len
            t_new_len = t_max_len + (q_max_len - q_len)
        else:
            q_new_len = q_max_len

        # Head+Tail method
        q_len_head = round(q_new_len * config.head)
        q_len_tail = -1 * (q_new_len - q_len_head)
        t_len_head = round(t_new_len * config.head)
        t_len_tail = -1 * (t_new_len - t_len_head)
        #t = t[:t_new_len]
        if config.head_tail :
            q = q[:q_len_head]+q[q_len_tail:]
            t = t[:t_len_head]+t[t_len_tail:]
            #t = t[:t_new_len]
        else:
            # No Head+Tail , usual processing
            q = q[:q_new_len]
            t = t[:t_new_len]

    return t, q

"""
def a_trim_input(tokenizer, answer, a_max_sequence_length, a_max_len):

    a = tokenizer.tokenize(answer)

    a_len = len(a)

    if (a_len+2) > a_max_sequence_length:

        a_new_len = a_max_len

        # Head+Tail method
        a_len_head = round(a_new_len * config.head)
        a_len_tail = -1 * (a_new_len - a_len_head)
        if config.head_tail :
            a = a[:a_len_head]+a[a_len_tail:]
        else:
            # No Head+Tail , usual processing
            a = a[:a_new_len]

    return a
"""

def a_trim_input(tokenizer, title, answer, a_max_sequence_length, t_max_len, a_max_len):

    t = tokenizer.tokenize(title)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    a_len = len(a)

    if (t_len+a_len+3) > a_max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + (t_max_len - t_len)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            t_new_len = t_max_len + (a_max_len - a_len)
        else:
            a_new_len = a_max_len

        # Head+Tail method
        a_len_head = round(a_new_len * config.head)
        a_len_tail = -1 * (a_new_len - a_len_head)
        t_len_head = round(t_new_len * config.head)
        t_len_tail = -1 * (t_new_len - t_len_head)
        #t = t[:t_new_len]
        if config.head_tail :
            a = a[:a_len_head]+a[a_len_tail:]
            t = t[:t_len_head]+t[t_len_tail:]
            #t = t[:t_new_len]
        else:
            # No Head+Tail , usual processing
            a = a[:a_new_len]
            t = t[:t_new_len]

    return t, a


def _convert_to_bert_inputs(title_q, title_a, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    if COMBINE_INPUT:
        stoken = ["[CLS]"] + title + ["[QBODY]"] + question + ["[ANS]"] + answer + ["[SEP]"]
        #stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
        #stoken = ["[CLS]"] + title  + question  + answer + ["[SEP]"]
    
        input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = _get_masks(stoken, max_sequence_length)
        input_segments = _get_segments(stoken, max_sequence_length)

        return [input_ids, input_masks, input_segments]
    else:
        q_token = ["[CLS]"] + title_q + ["[SEP]"] + question + ["[SEP"]
        q_input_ids = _get_ids(q_token, tokenizer, T_MAX_LEN+Q_MAX_LEN+3)
        q_input_masks = _get_masks(q_token, T_MAX_LEN+Q_MAX_LEN+3)
        q_input_segments = _get_segments(q_token, T_MAX_LEN+Q_MAX_LEN+3)
        
        #a_token = ["[CLS]"] + answer + ["[SEP]"]
        #a_input_ids = _get_ids(a_token, tokenizer, A_MAX_LEN+2)
        #a_input_masks = _get_masks(a_token, A_MAX_LEN+2)
        #a_input_segments = _get_segments(a_token, A_MAX_LEN+2)
        a_token = ["[CLS]"] + title_a + ["[SEP]"] + answer + ["[SEP"]
        a_input_ids = _get_ids(a_token, tokenizer, T_MAX_LEN+A_MAX_LEN+3)
        a_input_masks = _get_masks(a_token, T_MAX_LEN+A_MAX_LEN+3)
        a_input_segments = _get_segments(a_token, T_MAX_LEN+A_MAX_LEN+3)
        
        return [q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length, num_features, cat_features, 
                        t_max_len=T_MAX_LEN, q_max_len=Q_MAX_LEN, a_max_len=A_MAX_LEN):
    if COMBINE_INPUT:
        input_ids, input_masks, input_segments = [], [], []
        for _, instance in df[columns].iterrows():
            t, q, a = instance.question_title, instance.question_body, instance.answer
            t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length, t_max_len, q_max_len, a_max_len)
            ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        return [
                torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(), 
                torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(num_features, dtype=np.float32)).float(),
                torch.from_numpy(np.asarray(cat_features, dtype=np.int32)).long(),
                ]
    else:
        q_input_ids, q_input_masks, q_input_segments = [], [], []
        a_input_ids, a_input_masks, a_input_segments = [], [], []
        for _, instance in df[columns].iterrows():
            t, q, a = instance.question_title, instance.question_body, instance.answer
            t_q, q = q_trim_input(tokenizer, t, q, q_max_sequence_length, t_max_len, q_max_len)
            #a = a_trim_input(tokenizer, a, a_max_sequence_length, a_max_len)
            t_a, a = a_trim_input(tokenizer, t, a, a_max_sequence_length, t_max_len, a_max_len)
            q_ids, q_masks, q_segments, a_ids, a_masks, a_segments = _convert_to_bert_inputs(t_q, t_a, q, a, tokenizer, max_sequence_length)
            q_input_ids.append(q_ids)
            q_input_masks.append(q_masks)
            q_input_segments.append(q_segments)
            a_input_ids.append(a_ids)
            a_input_masks.append(a_masks)
            a_input_segments.append(a_segments)
        return [
                torch.from_numpy(np.asarray(q_input_ids, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(q_input_masks, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(q_input_segments, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(a_input_ids, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(a_input_masks, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(a_input_segments, dtype=np.int32)).long(),
                torch.from_numpy(np.asarray(num_features, dtype=np.float32)).float(),
                torch.from_numpy(np.asarray(cat_features, dtype=np.int32)).long(),
                ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


if COMBINE_INPUT:

    class QuestDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, lengths, labels = None):

            self.inputs = inputs
            if labels is not None:
                self.labels = labels
            else:
                self.labels = None
            self.lengths = lengths

        def __getitem__(self, idx):

            input_ids       = self.inputs[0][idx]
            input_masks     = self.inputs[1][idx]
            input_segments  = self.inputs[2][idx]
            num_features    = self.inputs[3][idx]
            cat_features    = self.inputs[4][idx]
            lengths         = self.lengths[idx]
            if self.labels is not None: # targets
                labels = self.labels[idx]
                return input_ids, input_masks, input_segments, num_features, cat_features, labels, lengths
            return input_ids, input_masks, input_segments, num_features, cat_features, lengths

        def __len__(self):
            return len(self.inputs[0])


    class CustomBert(BertPreTrainedModel):

        def __init__(self, config, cat_dims):
            super(CustomBert, self).__init__(config)
            self.num_labels = config.num_labels
            self.bert = BertModel(config)
            self.embeddings = nn.ModuleList([
                nn.Embedding(x, y) for x, y in cat_dims
            ])
            self.emb_drop = nn.Dropout(0.2)
            n_emb_out = sum([y for x, y in cat_dims])
            self.dropout = nn.Dropout(0.2)
            self.classifier_final = nn.Linear(config.hidden_size+n_emb_out+4, self.config.num_labels)  # num_features=4

            self.init_weights()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            num_features=None,
            cat_features=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
        ):

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

            emb = [
                emb_layer(cat_features[:, j]) for j, emb_layer in enumerate(self.embeddings)
            ]
            emb = self.emb_drop(torch.cat(emb, 1))

            pooled_output = torch.cat([pooled_output, num_features, emb], 1)
            logits = self.classifier_final(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions)

else:

    class QuestDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, lengths, labels = None):

            self.inputs = inputs
            if labels is not None:
                self.labels = labels
            else:
                self.labels = None
            self.lengths = lengths

        def __getitem__(self, idx):

            q_input_ids       = self.inputs[0][idx]
            q_input_masks     = self.inputs[1][idx]
            q_input_segments  = self.inputs[2][idx]
            a_input_ids       = self.inputs[3][idx]
            a_input_masks     = self.inputs[4][idx]
            a_input_segments  = self.inputs[5][idx]
            num_features    = self.inputs[6][idx]
            cat_features    = self.inputs[7][idx]
            lengths         = self.lengths[idx]
            if self.labels is not None: # targets
                labels = self.labels[idx]
                return q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels, lengths
            return q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, lengths

        def __len__(self):
            return len(self.inputs[0])


    class CustomBert(BertPreTrainedModel):

        def __init__(self, config, cat_dims):
            super(CustomBert, self).__init__(config)
            self.num_labels = config.num_labels
            self.bert = BertModel(config)
            self.embeddings = nn.ModuleList([
                nn.Embedding(x, y) for x, y in cat_dims
            ])
            self.emb_drop = nn.Dropout(0.1)
            n_emb_out = sum([y for x, y in cat_dims])
            self.num_drop = nn.Dropout(0.1)
            self.q_dropout = nn.Dropout(0.1)
            self.a_dropout = nn.Dropout(0.1)
            #self.dropout_all = nn.Dropout(0.2)
            #self.dropout_a = nn.Dropout(0.2)
            #self.dropout_q = nn.Dropout(0.2)
            #self.classifier_all = nn.Linear(config.hidden_size*2+n_emb_out+4, 64)  # num_features=4
            #self.classifier_all = nn.Sequential(
            #    nn.Linear(config.hidden_size*2+n_emb_out+4, 64),
            #    nn.ReLU(inplace=True),
            #)
            #self.classifier_a = nn.Linear(config.hidden_size+n_emb_out+4, 64)  # num_features=4
            #self.classifier_a = nn.Sequential(
            #    nn.Linear(config.hidden_size+n_emb_out+4, 64),
            #    nn.ReLU(inplace=True),
            #)
            #self.classifier_q = nn.Linear(config.hidden_size+n_emb_out+4, 64)  # num_features=4
            #self.classifier_q = nn.Sequential(
            #    nn.Linear(config.hidden_size+n_emb_out+4, 64),
            #    nn.ReLU(inplace=True),
            #)
            self.classifier_final = nn.Linear(config.hidden_size*2+n_emb_out+4, self.config.num_labels)
            #self.classifier_final = nn.Linear(64*3, self.config.num_labels)  # num_features=4
            #self.classifier_final = nn.Sequential(
            #    nn.BatchNorm1d(64*3),
            #    nn.Linear(64*3, self.config.num_labels),
            #)
            self.init_weights()

        def forward(
            self,
            q_input_ids=None,
            q_attention_mask=None,
            q_token_type_ids=None,
            a_input_ids=None,
            a_attention_mask=None,
            a_token_type_ids=None,
            num_features=None,
            cat_features=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
        ):

            q_outputs = self.bert(
                q_input_ids,
                attention_mask=q_attention_mask,
                token_type_ids=q_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            q_pooled_output = q_outputs[1]
            q_pooled_output = self.q_dropout(q_pooled_output)

            a_outputs = self.bert(
                a_input_ids,
                attention_mask=a_attention_mask,
                token_type_ids=a_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            a_pooled_output = a_outputs[1]
            a_pooled_output = self.a_dropout(a_pooled_output)

            emb = [
                emb_layer(cat_features[:, j]) for j, emb_layer in enumerate(self.embeddings)
            ]
            emb = self.emb_drop(torch.cat(emb, 1))

            num_features = self.num_drop(num_features)

            pooled_output = torch.cat([q_pooled_output, a_pooled_output, num_features, emb], 1)
            #all_logits = self.classifier_all(pooled_output)
            #all_logits = self.dropout_all(all_logits)
            logits = self.classifier_final(pooled_output)
            
            #a_pooled_output = torch.cat([a_pooled_output, num_features, emb], 1)
            #a_logits = self.classifier_a(a_pooled_output)
            #a_logits = self.dropout_a(a_logits)

            #q_pooled_output = torch.cat([q_pooled_output, num_features, emb], 1)
            #q_logits = self.classifier_q(q_pooled_output)
            #q_logits = self.dropout_q(q_logits)

            #concat_logits = torch.cat([all_logits, q_logits, a_logits], 1)
            #logits = self.classifier_final(concat_logits)

            #logits = torch.cat([q_logits, a_logits], 1)

            outputs = (logits,) + q_outputs[2:] + a_outputs[2:]  # add hidden states and attention if they are here
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions)


def train_model(model, train_loader, optimizer, criterion, scheduler, config):
    
    model.train()
    avg_loss = 0.
    avg_loss_1 = 0.
    avg_loss_2 = 0.
    avg_loss_3 = 0.
    avg_loss_4 = 0.
    avg_loss_5 = 0.
    #tk0 = tqdm(enumerate(train_loader),total =len(train_loader))
    optimizer.zero_grad()
    for idx, batch in enumerate(train_loader):
        if COMBINE_INPUT:
            input_ids, input_masks, input_segments, num_features, cat_features, labels, _ = batch
            input_ids, input_masks, input_segments, num_features, cat_features, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)            
        
            output_train = model(input_ids = input_ids.long(),
                             labels = None,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                             num_features = num_features,
                             cat_features = cat_features,
                            )
        else:
            q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels, _ = batch
            q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels = q_input_ids.to(device), q_input_masks.to(device), q_input_segments.to(device), a_input_ids.to(device), a_input_masks.to(device), a_input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)

            output_train = model(q_input_ids = q_input_ids.long(),
                             labels = None,
                             q_attention_mask = q_input_masks,
                             q_token_type_ids = q_input_segments,
                             a_input_ids = a_input_ids.long(),
                             a_attention_mask = a_input_masks,
                             a_token_type_ids = a_input_segments,
                             num_features = num_features,
                             cat_features = cat_features,
                            )
        logits = output_train[0] #output preds
        loss = criterion(logits, labels)
        loss.backward()
        if (idx + 1) % config.accum_steps == 0:    
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_loss += loss.item() / (len(train_loader)*config.accum_steps)
        if COMBINE_INPUT:
            del input_ids, input_masks, input_segments, labels
        else:
            del q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5


def val_model(model, criterion, val_loader, val_shape, batch_size=8):

    avg_val_loss = 0.
    model.eval() # eval mode
    
    valid_preds = np.zeros((val_shape, len(target_cols)))
    original = np.zeros((val_shape, len(target_cols)))
    
    #tk0 = tqdm(enumerate(val_loader))
    with torch.no_grad():
        
        for idx, batch in enumerate(val_loader):
            if COMBINE_INPUT:
                input_ids, input_masks, input_segments, num_features, cat_features, labels, _ = batch
                input_ids, input_masks, input_segments, num_features, cat_features, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)            
            
                output_val = model(input_ids = input_ids.long(),
                               labels = None,
                               attention_mask = input_masks,
                               token_type_ids = input_segments,
                               num_features = num_features,
                               cat_features = cat_features,
                              )
            else:
                q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels, _ = batch
                q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels = q_input_ids.to(device), q_input_masks.to(device), q_input_segments.to(device), a_input_ids.to(device), a_input_masks.to(device), a_input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)

                output_val = model(q_input_ids = q_input_ids.long(),
                             labels = None,
                             q_attention_mask = q_input_masks,
                             q_token_type_ids = q_input_segments,
                             a_input_ids = a_input_ids.long(),
                             a_attention_mask = a_input_masks,
                             a_token_type_ids = a_input_segments,
                             num_features = num_features,
                             cat_features = cat_features,
                            )
            logits = output_val[0] #output preds
            
            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()
        
        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
        
        # np.save("preds.npy", preds)
        # np.save("actuals.npy", original)
        
        rho_val = np.mean([spearmanr(original[:, i], preds[:,i]).correlation for i in range(preds.shape[1])])
        print('\r val_spearman-rho: %s' % (str(round(rho_val, 5))), end = 100*' '+'\n')
        
        for i in range(len(target_cols)):
            logger.info(f"{i}, {spearmanr(original[:,i], preds[:,i])}")
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
        
    return avg_val_loss, score/len(target_cols)


def predict_valid_result(model, val_loader, val_length, batch_size=32):

    val_preds = np.zeros((val_length, len(target_cols)))
    original = np.zeros((val_length, len(target_cols)))

    model.eval()
    tk0 = tqdm(enumerate(val_loader))
    for idx, batch in tk0:
        if COMBINE_INPUT:
            input_ids, input_masks, input_segments, num_features, cat_features, labels, _ = batch
            input_ids, input_masks, input_segments, num_features, cat_features, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)            
            with torch.no_grad():
                outputs = model(input_ids = input_ids.long(),
                            labels = None,
                            attention_mask = input_masks,
                            token_type_ids = input_segments,
                            num_features = num_features,
                            cat_features = cat_features,
                            )
        else:
            q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels, _ = batch
            q_input_ids, q_input_masks, q_input_segments, a_input_ids, a_input_masks, a_input_segments, num_features, cat_features, labels = q_input_ids.to(device), q_input_masks.to(device), q_input_segments.to(device), a_input_ids.to(device), a_input_masks.to(device), a_input_segments.to(device), num_features.to(device), cat_features.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(q_input_ids = q_input_ids.long(),
                             labels = None,
                             q_attention_mask = q_input_masks,
                             q_token_type_ids = q_input_segments,
                             a_input_ids = a_input_ids.long(),
                             a_attention_mask = a_input_masks,
                             a_token_type_ids = a_input_segments,
                             num_features = num_features,
                             cat_features = cat_features,
                            )

        predictions = outputs[0]
        val_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()
        original[idx*batch_size : (idx+1)*batch_size] = labels.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(val_preds)).numpy()
    return output, original


def predict_result(model, test_loader, test_length, batch_size=32):

    test_preds = np.zeros((test_length, len(target_cols)))

    model.eval()
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        if COMBINE_INPUT:
            with torch.no_grad():
                outputs = model(input_ids = x_batch[0].to(device),
                            labels = None,
                            attention_mask = x_batch[1].to(device),
                            token_type_ids = x_batch[2].to(device),
                            num_features = x_batch[3].to(device),
                            cat_features = x_batch[4].to(device),
                           )
        else:
            with torch.no_grad():
                outputs = model(q_input_ids = x_batch[0].to(device),
                            labels = None,
                            q_attention_mask = x_batch[1].to(device),
                            q_token_type_ids = x_batch[2].to(device),
                            a_input_ids = x_batch[3].to(device),
                            a_attention_mask = x_batch[4].to(device),
                            a_token_type_ids = x_batch[5].to(device),
                            num_features = x_batch[6].to(device),
                            cat_features = x_batch[7].to(device),
                           )
        predictions = outputs[0]
        test_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output


def add_features(df):
    find = re.compile(r"^[^.]*")
    df['netloc'] = df['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
    df['qa_same_user_page_flag'] = (df['question_user_page']==df['answer_user_page'])*1
    df['question_title_num_words'] = df['question_title'].str.count('\S+')
    df['question_body_num_words'] = df['question_body'].str.count('\S+')
    df['answer_num_words'] = df['answer'].str.count('\S+')
    df['question_vs_answer_length'] = df['question_body_num_words']/df['answer_num_words']
    df['question_title_num_words'] = np.log1p(df['question_title_num_words'])
    df['question_body_num_words'] = np.log1p(df['question_body_num_words'])
    df['answer_num_words'] = np.log1p(df['answer_num_words'])
    df['question_vs_answer_length'] = np.log1p(df['question_vs_answer_length'])
    return df


def custom_loss(logits, labels):
    #q_loss = nn.BCEWithLogitsLoss()(logits[:,:21], labels[:,:21])
    #a_loss = nn.BCEWithLogitsLoss()(logits[:,21:], labels[:,21:])
    #custom_loss = 0.5*q_loss + 0.5*a_loss
    custom_loss = nn.BCEWithLogitsLoss()(logits, labels)
    #loss1 = nn.BCEWithLogitsLoss()(logits[:,0:19], labels[:,0:19])
    #loss2 = nn.BCEWithLogitsLoss()(logits[:,20:], labels[:,20:]) # except index=19
    #custom_loss = loss1 + loss2
    #custom_loss = 0.
    #for i in range(len(loss_sample_weights)):
    #    custom_loss += loss_sample_weights[i] * nn.BCEWithLogitsLoss()(logits[:,i], labels[:,i])
    return custom_loss


#===========================================================
# main
#===========================================================
def main():
    
    with timer('Data Loading'):
        train = pd.read_csv(f"{ROOT}train.csv").fillna("none")
        y_train = train[target_cols].values
        if config.test:
            test = pd.read_csv(f"{ROOT}test.csv").fillna("none")
            submission = pd.read_csv(f"{ROOT}sample_submission.csv")
    
    with timer('Num features'):
        train = add_features(train)
        if config.test:
            test = add_features(test)
        num_features = ['question_title_num_words', 'question_body_num_words', 'answer_num_words', 'question_vs_answer_length']
        train_num = train[num_features].values
        if config.test:
            test_num = test[num_features].values
                
    with timer('Cat features'):
        cat_features = ['netloc', 'category', 'qa_same_user_page_flag']
        ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='return_nan')
        ce_oe.fit(train[cat_features])
        train_cat_df = ce_oe.transform(train[cat_features])
        test_cat_df = ce_oe.transform(test[cat_features]).fillna(0).astype(int)
        train_cat = train_cat_df.values
        test_cat = test_cat_df.values
        cat_dims = []
        for col in cat_features:
            dim = train[col].nunique()
            cat_dims.append((dim+1, dim//2+1)) # for unknown=0
        print(cat_dims)

    if config.train:
        with timer('Create folds'):
            folds = train.copy()

            kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED)
            for fold, (train_index, val_index) in enumerate(kf.split(train.values, y_train)):
                folds.loc[val_index, 'fold'] = int(fold)
            """
            # less gap between CV vs LB with GroupKFold
            # https://www.kaggle.com/ratthachat/quest-cv-analysis-on-different-splitting-methods
            kf = GroupKFold(n_splits=NUM_FOLDS)
            for fold, (train_index, val_index) in enumerate(kf.split(X=train.question_body, groups=train.question_body)):
                folds.loc[val_index, 'fold'] = int(fold)
            """
            folds['fold'] = folds['fold'].astype(int)
            save_cols = [ID] + target_cols + ['fold']
            folds[save_cols].to_csv('folds.csv', index=None)

    with timer('Prepare Bert config'):
        tokenizer = BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt", 
                                                  do_lower_case=True)
        input_categories = ['question_title', 'question_body', 'answer']
        bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
        bert_config = BertConfig.from_json_file(bert_model_config)
        bert_config.num_labels = len(target_cols)
        bert_model = 'bert-base-uncased'
        do_lower_case = 'uncased' in bert_model
        output_model_file = 'bert_pytorch.bin'
    
    if config.train:

        BATCH_SIZE = 8
        if DEBUG:
            epochs = 1
        else:
            epochs = config.epochs
        ACCUM_STEPS = config.accum_steps

        with timer('Train Bert'):
            
            for fold in range(NUM_FOLDS):

                logger.info(f"Current Fold: {fold}")
                train_index = folds[folds.fold != fold].index
                val_index = folds[folds.fold == fold].index

                train_df, val_df = train.iloc[train_index], train.iloc[val_index]
                logger.info(f"Train Shapes: {train_df.shape}")
                logger.info(f"Valid Shapes: {val_df.shape}")
            
                logger.info("Preparing train datasets....")
            
                inputs_train = compute_input_arays(train_df, input_categories, tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH, 
                                                   num_features=train_num[train_index], cat_features=train_cat[train_index])
                outputs_train = compute_output_arrays(train_df, columns=target_cols)
                outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
                lengths_train = np.argmax(inputs_train[0]==0, axis=1)
                lengths_train[lengths_train==0] = inputs_train[0].shape[1]
            
                logger.info("Preparing valid datasets....")
            
                inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH, 
                                                   num_features=train_num[val_index], cat_features=train_cat[val_index])
                outputs_valid = compute_output_arrays(val_df, columns = target_cols)
                outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
                lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
                lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
            
                logger.info("Preparing Dataloaders Datasets....")

                train_set = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
                train_sampler = RandomSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,sampler=train_sampler)
            
                valid_set = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
                valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            
                model = CustomBert.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config, cat_dims=cat_dims)
                model.zero_grad()
                model.to(device)
                torch.cuda.empty_cache()
                if config.freeze : ## This is basically using out of the box bert model while training only the classifier head with our data . 
                    for param in model.bert.parameters():
                        param.requires_grad = False
                model.train()
            
                i = 0
                best_avg_loss = 100.0
                best_score = -1.
                best_param_loss = None
                best_param_score = None
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]        
                optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=4e-5)
                #optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=4e-5)
                #criterion = nn.BCEWithLogitsLoss()
                criterion = custom_loss
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup, num_training_steps=epochs*len(train_loader)//ACCUM_STEPS)
                logger.info("Training....")
            
                for epoch in tqdm(range(epochs)):

                    torch.cuda.empty_cache()
                
                    start_time   = time.time()
                    avg_loss, avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5 = train_model(model, train_loader, optimizer, criterion, scheduler, config)
                    avg_val_loss, score = val_model(model, criterion, valid_loader, val_shape=val_df.shape[0], batch_size=BATCH_SIZE)
                    elapsed_time = time.time() - start_time

                    logger.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t train_loss={:.4f} \t train_loss_1={:.4f} \t train_loss_2={:.4f} \t train_loss_3={:.4f} \t train_loss_4={:.4f}  \t train_loss_5={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
                        epoch+1, epochs, avg_loss, avg_val_loss, avg_loss, avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5, score, elapsed_time))

                    if best_avg_loss > avg_val_loss:
                        i = 0
                        best_avg_loss = avg_val_loss 
                        best_param_loss = model.state_dict()

                    if best_score < score:
                        best_score = score
                        best_param_score = model.state_dict()
                        logger.info('best_param_score_{}_{}.pt'.format(config.expname ,fold))
                        torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname, fold))
                    else:
                        i += 1

            del train_df, val_df, model, optimizer, criterion, scheduler
            del valid_loader, train_loader, valid_set, train_set
            torch.cuda.empty_cache()
            gc.collect()
    
    if config.cv:

        with timer('CV'):

            folds = pd.read_csv(f'{MODEL_DIR}folds.csv')
            results = np.zeros((len(train), len(target_cols)))
            logits = np.zeros((len(train), len(target_cols)))

            for fold in range(NUM_FOLDS):
                
                #train_index = folds[folds.fold != fold].index
                val_index = folds[folds.fold == fold].index
                #train_df, val_df = train.iloc[train_index], train.iloc[val_index]
                val_df = train.iloc[val_index]
                
                inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH, 
                                                   num_features=train_num[val_index], cat_features=train_cat[val_index])
                outputs_valid = compute_output_arrays(val_df, columns = target_cols)
                outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
                lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
                lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
                valid_set = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
                valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, drop_last=False)
                
                model = CustomBert.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config, cat_dims=cat_dims)
                model.zero_grad()
                model.to(device)
                model.load_state_dict(torch.load(f'{MODEL_DIR}best_param_score_{config.expname}_{fold}.pt'))
                result, logit = predict_valid_result(model, valid_loader, len(val_df))  
                results[val_index, :] = result
                logits[val_index, :] = logit 
            
            rho_val = np.mean([spearmanr(logits[:,i], results[:,i]).correlation for i in range(results.shape[1])])
            logger.info(f'CV spearman-rho: {round(rho_val, 5)}')

            oof = pd.DataFrame()
            for i, col in enumerate(target_cols):
                oof[col] = results[:,i]
            oof.to_csv(f'oof_{config.expname}.csv', index=False)
    
    if config.test:

        with timer('Inference'):

            test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH, 
                                              num_features=test_num, cat_features=test_cat)
            lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
            lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
            test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
            test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
            result = np.zeros((len(test), len(target_cols)))

            for fold in range(NUM_FOLDS):
                model = CustomBert.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config, cat_dims=cat_dims)
                model.zero_grad()
                model.to(device)
                model.load_state_dict(torch.load(f'{MODEL_DIR}best_param_score_{config.expname}_{fold}.pt'))
                result += predict_result(model, test_loader, len(test)) 
                if DEBUG:
                    break
                    
            result /= NUM_FOLDS

        with timer('Create submission.csv'):
            submission.loc[:, 'question_asker_intent_understanding':] = result
            submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()



