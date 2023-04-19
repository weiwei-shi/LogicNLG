from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
import logging
import time 
import json,os,re

# 类别: id
sent_type2id_dict = {'count': 0, 'superlative': 1, 'comparative': 2, 'aggregation': 3, 'majority': 4, 'unique': 5, 'ordinal': 6}
# id: 类别
sent_id_type_dict = {0: 'count', 1: 'superlative', 2: 'comparative', 3: 'aggregation', 4: 'majority', 5: 'unique', 6: 'ordinal'}

# 得到attention mask
def get_atten_mask(tokens_ids, pad_index=0):
    return list(map(lambda x: 1 if x != pad_index else 0, tokens_ids))

class NewsDataset(Dataset):

    def __init__(self, file_path, tokenizer: BertTokenizer, max_length=512, device=None):
        sent_type = []
        content = []
        atten_mask = []
        seq_typ_ids = []
        with open(file_path, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                line = line.split('\t')

                if line[0] == "sentence":
                    continue
                sent_type.append(sent_type2id_dict[line[1]])
                token_ids = tokenizer.encode(line[0], max_length=max_length, pad_to_max_length=True)
                content.append(token_ids)
                atten_mask.append(get_atten_mask(token_ids))
                seq_typ_ids.append(tokenizer.create_token_type_ids_from_sequences(token_ids_0=token_ids[1:-1]))

        self.label = torch.from_numpy(np.array(sent_type)).unsqueeze(1).long()
        self.token_ids = torch.from_numpy(np.array(content)).long()
        self.seq_type_ids = torch.from_numpy(np.array(seq_typ_ids)).long()
        self.atten_masks = torch.from_numpy(np.array(atten_mask)).long()
        if device is not None:
            self.label = self.label.to(device)
            self.token_ids = self.token_ids.to(device)
            self.seq_type_ids = self.seq_type_ids.to(device)
            self.atten_masks = self.atten_masks.to(device)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.label[item], self.token_ids[item], self.seq_type_ids[item], self.atten_masks[item]


class NewsDataset2(Dataset):
    def __init__(self, file_path, tokenizer: BertTokenizer, max_length=512, device=None):
        # sent_type = []
        content = []
        atten_mask = []
        seq_typ_ids = []
        with open(file_path, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                # line = line.split('\t')

                if line == "question":
                    continue
                # sent_type.append(sent_type2id_dict[line[1]])
                token_ids = tokenizer.encode(line, max_length=max_length, pad_to_max_length=True)
                content.append(token_ids)
                atten_mask.append(get_atten_mask(token_ids))
                seq_typ_ids.append(tokenizer.create_token_type_ids_from_sequences(token_ids_0=token_ids[1:-1]))

        # self.label = torch.from_numpy(np.array(sent_type)).unsqueeze(1).long()
        self.token_ids = torch.from_numpy(np.array(content)).long()
        self.seq_type_ids = torch.from_numpy(np.array(seq_typ_ids)).long()
        self.atten_masks = torch.from_numpy(np.array(atten_mask)).long()
        if device is not None:
            # self.label = self.label.to(device)
            self.token_ids = self.token_ids.to(device)
            self.seq_type_ids = self.seq_type_ids.to(device)
            self.atten_masks = self.atten_masks.to(device)

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, item):
        return self.token_ids[item], self.seq_type_ids[item], self.atten_masks[item]


def train(train_dataset, model: BertForSequenceClassification, optimizer: AdamW, batch_size):
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    model.train()
    tr_loss = 0.0
    tr_acc = 0
    global_step = 0
    torch.cuda.empty_cache()
    for step, batch in tqdm(enumerate(train_loader)):
        # print(step)
        inputs = {
            'input_ids': batch[1],
            'token_type_ids': batch[2],
            'attention_mask': batch[3],
            'labels': batch[0]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        # print(loss)
        logits = outputs[1]

        tr_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算准确率
        _, pred = logits.max(1)
        number_corr = (pred == batch[0].view(-1)).long().sum().item()
        tr_acc += number_corr
        global_step += 1

    return tr_loss / global_step, tr_acc / len(train_dataset)


def evalate(eval_dataset, model: BertForSequenceClassification, batch_size):
    model.eval()
    eval_sampler = RandomSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    tr_acc = 0
    torch.cuda.empty_cache()
    for step, batch in tqdm(enumerate(eval_loader)):
        inputs = {
            'input_ids': batch[1],
            'token_type_ids': batch[2],
            'attention_mask': batch[3],
            'labels': batch[0]
        }
        outputs = model(**inputs)
        # loss = outputs[0]
        logits = outputs[1]

        # tr_loss += loss.item()

        # 计算准确率
        _, pred = logits.max(1)
        number_corr = (pred == batch[0].view(-1)).long().sum().item()
        tr_acc += number_corr

    return tr_acc / len(eval_dataset)


def predict(test_dataset, model, batch_size):
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
    torch.cuda.empty_cache()

    preds = []
    for step, batch in tqdm(enumerate(test_loader)):
        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'attention_mask': batch[2],
        }
        logits = model(**inputs)
        _, predict = logits[0].max(1)
        s = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in batch[0]]
        i = 0
        for p in predict:
            if step%200==0:
                print('\nsent: {}'.format(s[i]))
                print("type: {}".format(sent_id_type_dict[p.item()]))
                i = i + 1
            preds.append(sent_id_type_dict[p.item()])
    return preds


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def write_json(obj):
    # train_data
    # item_list = []
    # with open('data/train_lm_preprocessed2.json', 'r') as f:
    #     load_dict = json.load(f)
    #     i = 0
    #     for idx, _ in enumerate(load_dict):
    #         entry = load_dict[idx]
    #         entry.append(obj[i])
    #         i = i + 1
    #         item_list.append(entry)

    # with open('data/train_lm_preprocessed3.json', 'w', encoding='utf-8') as f:
    #     json.dump(item_list, f, indent=2)

    # test_data
    item_list = {}
    with open('data/test_lm2.json', 'r') as f:
        load_dict = json.load(f)
        keys = list(load_dict.keys())
        i = 0
        for idx, _ in enumerate(load_dict):
            table_id = keys[idx]
            entry = load_dict[table_id]
            for e in entry:
                # e.append(obj[i])
                e.insert(4, obj[i])
                i = i + 1
            item_list[table_id] = entry

    with open('data/test_lm3.json', 'w', encoding='utf-8') as f:
        json.dump(item_list, f, indent=2)

    # val_data
    # item_list = {}
    # with open('data/test_lm_pos_neg.json', 'r') as f:
    #     load_dict = json.load(f)
    #     keys = list(load_dict.keys())
    #     i = 0
    #     for idx, _ in enumerate(load_dict):
    #         table_id = keys[idx]
    #         entry = load_dict[table_id]
    #         for e in entry:
    #             e["pos"].append(obj[i])
    #             i = i + 1
    #             e["neg"].append(obj[i])
    #             i = i + 1
    #         item_list[table_id] = entry

    # with open('data/test_lm_pos_neg2.json', 'w', encoding='utf-8') as f:
    #     json.dump(item_list, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--lr_rate", default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epoches", default=20)
    parser.add_argument("--load_model", type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    # 创建config
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(sent_type2id_dict))
    # 创建tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 创建分类器
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config).to(device)
    model.to(device)
    # 加载模型
    if args.load_model:
        print(">> Load Model: {}".format(args.load_model))
        # model.load_state_dict(torch.load(args.load_model))
        model = BertForSequenceClassification.from_pretrained(args.load_model, config=config).to(device)

    # 定义优化器和损失函数
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_rate, eps=1e-8)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.do_train:
        logger.info('create train dataset')
        train_dataset = NewsDataset('data/type_data_train.tsv', tokenizer, max_length=args.max_length, device=device)
        logger.info('create eval dataset')
        eval_dataset = NewsDataset('data/type_data_valid.tsv', tokenizer, max_length=args.max_length, device=device)
        best_val_acc = 0.0
        for e in range(1, args.epoches):
            start_time = time.time()
            train_loss, train_acc = train(train_dataset, model, optimizer, args.batch_size)
            eval_acc = evalate(eval_dataset, model, args.batch_size)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            logger.info('Epoch: {:02} | Time: {}m {}s'.format(e, epoch_mins, epoch_secs))
            logger.info(
                'Train Loss: {:.6f} | Train Acc: {:.6f} | Eval Acc: {:.6f}'.format(train_loss, train_acc, eval_acc))
            if eval_acc > best_val_acc:
                best_val_acc = eval_acc
                torch.save(model.state_dict(), 'models/Type_ep_{}.pt'.format(e))

    if args.do_test:
        logger.info('create test dataset')
        test_dataset = NewsDataset2('data/decom_test.tsv', tokenizer, max_length=args.max_length, device=device)
        prediction = predict(test_dataset, model, args.batch_size)
        write_json(prediction)

    '''
    train:
        CUDA_VISIBLE_DEVICES=0 python Classify.py --do_train
    test:
        CUDA_VISIBLE_DEVICES=0 python Classify.py --do_test --load_model models/Type_ep_4.pt
    '''

