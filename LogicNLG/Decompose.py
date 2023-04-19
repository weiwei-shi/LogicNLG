from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
import pandas as pd
import pdb
import os
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_evidence_features(evidence, tokenizer):
    evidence_lst = evidence.split('. ')
    evd_num = len(evidence_lst)
    if evd_num < 2:
        pad_evd(evidence_lst)
    else:
        evidence_lst = evidence_lst[:2]
        evd_num = len(evidence_lst)

    tokens_tmp = []
    for evd in evidence_lst:
        if not evd.endswith('.'):
            evd += '.'
        evd_tmp = tokenizer.tokenize(evd)
        tokens_tmp.append(evd_tmp)
    _truncate_seq_pair_verb(tokens_tmp, args.max_seq_length_evidence - 4)

    tokens = []

    for idx, item in enumerate(tokens_tmp):
        tokens.extend(item) 
    input_ids = tokenizer.convert_tokens_to_ids(tokens) # 词id
    input_mask = [1] * len(input_ids)

    padding = [0] * (args.max_seq_length_evidence - len(input_ids)) # 填充到最大长度
    input_ids += padding
    input_mask += padding

    assert len(input_ids) == args.max_seq_length_evidence
    assert len(input_mask) == args.max_seq_length_evidence
    return input_ids, input_mask

# 如果句子不能分解，进行填充
def pad_evd(evd_lst):
    while len(evd_lst) < 2:
        evd_lst.append("[PAD]")

# 对超出长度的句子进行截断
def _truncate_seq_pair_verb(tokens_tmp, max_length):
    while True:
        total_length = 0
        max_idx = 0
        max_idx_next = 0
        max_len = 0
        for i, tokens in enumerate(tokens_tmp):
            total_length += len(tokens)
            if len(tokens) > max_len:
                max_idx_next = max_idx
                max_len = len(tokens)
                max_idx = i
        if total_length <= max_length:
            break
        if len(tokens_tmp[max_idx]) > 2:
            tokens_tmp[max_idx].pop()
        else:
            tokens_tmp[max_idx_next].pop()


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        #table = pd.read_csv(args.data_dir + '/' + item.table_file).astype(str)
        encoding = self.tokenizer(item.question, pad_to_max_length=True, max_length= self.max_length, return_tensors="pt")
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        evidence = item.evidence
        input_ids, input_mask = get_evidence_features(evidence, tokenizer=self.tokenizer)
        encoding["input_ids_evd"] = torch.tensor(input_ids, dtype=torch.long)
        encoding["attention_mask_evd"] = torch.tensor(input_mask, dtype=torch.long)

        return encoding

    def __len__(self):
        return len(self.data)

class TableDataset2(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        #table = pd.read_csv(args.data_dir + '/' + item.table_file).astype(str)
        encoding = self.tokenizer(item.question, pad_to_max_length=True, max_length= self.max_length, return_tensors="pt")
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        return encoding

    def __len__(self):
        return len(self.data)


def get_dataloader(data_dir, file_name, batch_size, tokenizer, max_length, phase):

    data = pd.read_csv(os.path.join(data_dir, file_name), sep='\t')
    if phase == 'test':
        dataset = TableDataset2(data, tokenizer, max_length)
    else: 
        dataset = TableDataset(data, tokenizer, max_length)
    if phase == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


# def _mkdir(path):
#     if not os.path.exists(path):
#         os.mkdir(path)
#         print(">> mkdir: {}".format(path))


def run_train(device, tokenizer, model, writer, phase="train"):

    model.train()

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr_rate)

    optimizer.zero_grad()

    train_dataloader = get_dataloader(data_dir=args.data_dir, file_name=args.train_file, batch_size=args.batch_size,
                                      tokenizer=tokenizer, max_length=args.max_seq_length_evidence, phase=phase)

    global_step = 0
    best_acc = 0.0

    loss_fct = CrossEntropyLoss()
    optimizer_flag = 0

    # training
    for epoch in range(args.epoch):
        print("start training {}th epoch".format(epoch))

        for idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            input_ids_evd = batch["input_ids_evd"].to(device)
            attention_mask_evd = batch["attention_mask_evd"].to(device)

            y_ids = input_ids_evd[:, :-1].contiguous()
            lm_labels = input_ids_evd[:, 1:].clone().detach()
            lm_labels[input_ids_evd[:, 1:] == tokenizer.pad_token_id] = -100

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            if idx%10 == 0:
                print("Training {} Loss：{}" .format(idx, loss.item()))

            # if idx%500==0:
            #     print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()

        if model == "t5-small":    
            torch.save(model.state_dict(), '{}/Decom_small_ep_{}.pt'.format(args.save_model, epoch))
        else:
            torch.save(model.state_dict(), '{}/Decom_ep_{}.pt'.format(args.save_model, epoch))


def run_eval(device, tokenizer, model, global_step=-1, writer=None, phase=None):
    model.eval()

    data_file = {"dev": "decom_dev.tsv",
                 "test": "decom_test.tsv"}

    eval_dataloader = get_dataloader(data_dir=args.data_dir, file_name=data_file[phase], batch_size=args.eval_batch_size,
                                     tokenizer=tokenizer, max_length=args.max_seq_length_evidence, phase=phase)

    preds = []
    acts = []

    for idx, batch in enumerate(tqdm(eval_dataloader, desc=phase)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        input_ids_evd = batch["input_ids_evd"].to(device)
        attention_mask_evd = batch["attention_mask_evd"].to(device)

        generated_ids = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            max_length=150, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
            )

        p = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        t = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in input_ids_evd]

        if idx%50==0:
            print('Completed {}\n'.format(idx))
            print('Presicton: {}\n'.format(p))
            print('Target: {}\n'.format(t))

        preds.extend(p)
        acts.extend(t)

    return preds, acts

def run_test(device, tokenizer, model, global_step=-1, writer=None, phase=None):
    model.eval()

    data_file = {"dev": "decom_dev.tsv",
                 "test": "decom_test.tsv"}

    test_dataloader = get_dataloader(data_dir=args.data_dir, file_name=data_file[phase], batch_size=args.eval_batch_size,
                                     tokenizer=tokenizer, max_length=args.max_seq_length_evidence, phase=phase)

    preds = []

    for idx, batch in enumerate(tqdm(test_dataloader, desc=phase)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated_ids = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            max_length=150, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
            )

        p = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        if idx%50==0:
            print('Completed {}\n'.format(idx))
            print('Presicton: {}\n'.format(p))

        preds.extend(p)

    final_df = pd.DataFrame({'GeneratedText':preds})
    final_df.to_csv('data/prediction.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str, help="input data dir")
    parser.add_argument("--train_file", default="decom_train.tsv", help="can be train_complex.tsv")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--load_model", type=str)

    parser.add_argument("--lr_rate", default=2e-5, help="5e-5")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epoch", default=5)
    parser.add_argument("--period", default=2000)

    parser.add_argument("--save_model", default="models")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--every', default=50, type=int)
    parser.add_argument("--max_seq_length_evidence", default=64)
    parser.add_argument("--in_dim", default=768)
    parser.add_argument("--mem_dim", default=768)

    args = parser.parse_args()

    device = torch.device('cuda:0')

    # 创建文件夹
    # _mkdir('./outputs')
    # if args.do_train:
    #     _mkdir(args.save_model)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter(os.path.join('tensorboard/decompose'))

    if args.load_model:
        print(">> Load Model: {}".format(args.load_model))
        model.load_state_dict(torch.load(args.load_model))

    print(">> Load data: {}".format(args.data_dir))

    # build pipeline
    if args.do_train:
        run_train(device, tokenizer, model, writer=writer, phase="train")

    if args.do_eval:
        run_eval(device, tokenizer, model, global_step=-1, writer=None, phase="dev")

    if args.do_test:
        run_test(device, tokenizer, model, global_step=-1, writer=None, phase="test")



    '''
    train:
        CUDA_VISIBLE_DEVICES=0 python Decompose.py --do_train
    eval:
        CUDA_VISIBLE_DEVICES=0 python Decompose.py --do_eval --load_model models/Decom_ep_3.pt
    test:
        CUDA_VISIBLE_DEVICES=0 python Decompose.py --do_test --load_model models/Decom_ep_3.pt
    '''