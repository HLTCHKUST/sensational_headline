import torch
import torch.utils.data as data
from torch.autograd import Variable
from utils.global_variables import *
import pickle
import logging

import re
def my_clean(context):
    context = re.sub(r"[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",context)
    context = re.sub(r"[【】╮╯▽╰╭★→「」]+", "",context)
    context = re.sub(r"[！，❤。～《》：（）【】「」？”“；：、]+","",context)
    return context

class Lang:
    def __init__(self, vocab_file=None):
        self.vocab_file = vocab_file
        self.idx2word = {UNK_idx: "UNK", PAD_idx: "PAD", EOS_idx: "EOS", SOS_idx: "SOS"}
        self.n_words = len(self.idx2word)
        self.word2count = dict(zip(self.idx2word.values(), [0] * self.n_words))
        if vocab_file is not None:
            with open(vocab_file,"r") as f:
                for w in f.readlines():
                    self.idx2word[self.n_words] = w.split()[0].strip()
                    self.n_words += 1
                    self.word2count[w.split()[0].strip()] = 0
        self.word2idx = dict(zip(self.idx2word.values(), self.idx2word.keys()))

    def extend(self, vocab_file):
        existing_words = set(self.idx2word.values())
        with open(vocab_file,"r") as f:
            for w in f.readlines():
                if w in existing_words:
                    continue
                self.idx2word[self.n_words] = w.split()[0].strip()
                self.n_words += 1
                self.word2count[w.split()[0].strip()] = 0
        self.word2idx = dict(zip(self.idx2word.values(), self.idx2word.keys()))

    def index_words(self, sent):
        for word in sent.split():
            self.index_word(word)


    def index_word(self, word):
        if word not in self.word2idx:
            if self.vocab_file is not None:
                self.word2count["UNK"] += 1
                return
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1

        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    def __init__(self, x_seq, y_seq, max_q, lang, pointer_gen=False):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.max_q = max_q
        self.vocab_size = lang.n_words
        self.word2idx = lang.word2idx
        self.pointer_gen = pointer_gen

    def __getitem__(self, idx):

        item = {}
        item["input_txt"] = self.x_seq[idx]
        item["input_batch"] = self.process(item["input_txt"], False)
        item["label"] = torch.FloatTensor([1. if self.y_seq[idx] == "1" else 0.]) 
        item["max_q"] = self.max_q

        return item 

    def __len__(self):
        return len(self.x_seq)

    def process_target(self, target_txt, oovs):
        # seq = [self.word2idx[word] if word in self.word2idx and self.word2idx[word] < self.output_vocab_size else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
        seq = []
        for word in target_txt.strip().split():
            if word in self.word2idx:
                seq.append(self.word2idx[word])
            elif word in oovs:
                seq.append(self.vocab_size + oovs.index(word))
            else:
                seq.append(UNK_idx)
        seq.append(EOS_idx)
        seq = torch.LongTensor(seq)
        return seq

    def process_input(self, input_txt):
        seq = []
        oovs = []
        for word in input_txt.strip().split():
            if word in self.word2idx:
                seq.append(self.word2idx[word])
            else:
                if word not in oovs:
                    oovs.append(word)
                seq.append(self.vocab_size + oovs.index(word))
        
        seq = torch.LongTensor(seq)
        return seq, oovs

    def process(self, input_txt, target):
        
        if target:
            # seq = [self.word2idx[word] if word in self.word2idx and self.word2idx[word] < self.output_vocab_size else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
            seq = [self.word2idx[word] if word in self.word2idx else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
        else:
            seq = [self.word2idx[word] if word in self.word2idx else UNK_idx for word in input_txt.strip().split()]
        seq = torch.Tensor(seq)
        return seq

def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if max_len:
            lengths = [len(seq) if len(seq) < max_len[0] else max_len[0] for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max_len[0]).long()
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            if max_len:
                end = min(lengths[i], max_len[0])
            else:
                end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["input_batch"]), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    input_batch, input_lengths = merge(item_info['input_batch'], item_info['max_q'])
    input_batch = Variable(input_batch)
    input_lengths = Variable(torch.LongTensor(input_lengths))
    item_info["label"] = torch.cat(item_info["label"], 0)

    if USE_CUDA:
        input_batch = input_batch.cuda()
        input_lengths = input_lengths.cuda()
        item_info["label"] = item_info["label"].cuda()

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = input_lengths
    d["input_txt"] = item_info["input_txt"]
    d["label"] = item_info["label"]

    return d 

def get_seq(data, lang, batch_size, update_lang, max_q, max_len, pointer_gen, shuffle=True):
    x_seq, y_seq, input_lengths, target_lengths = [], [], [], []
    
    if max_len is not None:
    	data = data[:max_len]
    data = data[:max_len]
    for d in data:
        x_seq.append(d["x"])
        y_seq.append(d["label"])
        if update_lang:
            lang.index_words(d["x"])
    
    dataset = Dataset(x_seq, y_seq, max_q, lang, pointer_gen=pointer_gen)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
    	shuffle=shuffle, collate_fn=collate_fn)
    
    return data_loader

def read_langs(file_name):

    data = []
    with open(file_name, "r") as f:
            for line in f.readlines():
                d = {}
                elements = line.strip().split("\t")
                if len(elements) != 2:
                    continue
                d["x"] = my_clean(elements[0])
                d["label"] = elements[1]
                d["x_len"] = len(d["x"].strip().split())
                data.append(d)

    max_q = max([d["x_len"] for d in data])

    return data, max_q, 0

def prepare_data_seq(batch_size, output_vocab_size, debug=False, shuffle=True, pointer_gen=False):
    f_name = "dataset/sensation/db_pointer.pkl"
    import pickle
    if debug and os.path.exists(f_name):
        with open(f_name, "rb") as f:
            train = pickle.load(f)
            dev = pickle.load(f)
            test = pickle.load(f)
            lang = pickle.load(f)
            max_q = pickle.load(f)
            max_r = pickle.load(f)
    else:
        file_train = "dataset/sensation/train.txt"
        file_dev = "dataset/sensation/dev.txt"
        file_test = "dataset/sensation/test.txt"
        vocab_file = "dataset/lcsts/vocab.dict.{}".format(output_vocab_size)
        
        d_train, max_q_train, max_r_train = read_langs(file_train)
        d_dev, max_q_dev, max_r_dev = read_langs(file_dev)
        d_test, max_q_test, max_r_test = read_langs(file_test)
        
        lang = Lang(vocab_file)
        logging.info("finish loading lang")
        
        max_q = max(max_q_train, max_q_test, max_q_dev)
        max_r = max(max_r_train, max_r_test, max_r_dev)
        max_q = min(max_q, 400)
        logging.info("max_q: {}, max_r: {}".format(max_q, max_r))
        
        logging.info("start get seq for train")
        if debug:
        	max_len = 20000
        else:
        	max_len = None
        
        train = get_seq(d_train, lang, batch_size, True, max_q, max_len, pointer_gen=pointer_gen, shuffle=shuffle)
        logging.info("start get seq for dev")
        dev = get_seq(d_dev, lang, batch_size, False, max_q, max_len, pointer_gen=pointer_gen, shuffle=False)
        logging.info("start get seq for test")
        test = get_seq(d_test, lang, batch_size, False, max_q, max_len, pointer_gen=pointer_gen, shuffle=False)
    if debug and not os.path.exists(f_name):
        with open(f_name, "wb") as f:
            pickle.dump(train, f)
            pickle.dump(dev, f)
            pickle.dump(test, f)
            pickle.dump(lang, f)
            pickle.dump(max_q, f)
            pickle.dump(max_r, f)
    return train, dev, test, lang, max_q, max_r

def input_txt_to_batch(input_txt, lang):

    batch_size = len(input_txt)
    input_txt = [my_clean(" ".join(word_list)).split() for word_list in input_txt]
    input_idxs = [[lang.word2idx[w] if w in lang.word2idx else UNK_idx for w in sent] for sent in input_txt]
    max_len = max([len(txt) for txt in input_txt])
    max_len = 33
    input_idxs = [sent_idxs + [PAD_idx] * (max_len - len(sent_idxs)) for sent_idxs in input_idxs]
    input_batch = Variable(torch.LongTensor(input_idxs))
    if USE_CUDA:
        input_batch = input_batch.cuda()
    return input_batch
