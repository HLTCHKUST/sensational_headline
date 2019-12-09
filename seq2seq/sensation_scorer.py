import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.sensation_config import *
import numpy as np

def get_embedding(word2vec_file, lang):
    word2vector = {}
    with open(word2vec_file) as f:
        lines = f.readlines()
        assert len(lines[0].strip().split())  == 2
        emb_size = int(lines[0].strip().split()[1])
        for line in lines[1:]:
            elements = line.strip().split()
            word2vector[elements[0]] = [float(e) for e in elements[1:]]

    # new_embedding = np.zeros((100, emb_size))
    new_embedding = np.random.randn(lang.n_words, emb_size) * 0.01
    for i in range(lang.n_words):
    # for i in range(100):
        if lang.idx2word[i] in word2vector:
            new_embedding[i] = word2vector[lang.idx2word[i]]

    return new_embedding

def share_embedding(opts, lang):
        embedding = nn.Embedding(lang.n_words, opts['emb_size'], padding_idx=PAD_idx)
        embedding.weight.data.requires_grad = True

        if opts['emb_file'] is not None:
            print("loading pretrained emb")
            pre_embedding = get_embedding(opts['emb_file'], lang)
            embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
            embedding.weight.data.requires_grad = False

        return embedding

class SensationCNN(nn.Module):
    def __init__(self, opts, lang):

        super(SensationCNN, self).__init__()
        embedding_size = opts['emb_size']
        num_filters = opts["num_filters"]
        filter_sizes = [int(s) for s in opts["filter_sizes"].strip().split(',')]
        vocab_size = lang.n_words

        self.embedding = share_embedding(opts, lang)

        self.convs_q = nn.ModuleList(nn.Conv1d(embedding_size, num_filters, filter_size) for filter_size in filter_sizes)
        self.out = nn.Linear(num_filters * len(filter_sizes), 1)

        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.convs_q = self.convs_q.cuda()
            self.out = self.out.cuda()


    def forward(self, input_batch):

        q_embedded = self.embedding(input_batch)
        q_embedded = q_embedded.transpose(1, 2)

        feat_maps_q = [F.relu(conv(q_embedded))  for conv in self.convs_q]
        feat_map2_q = torch.cat([F.max_pool1d(feat_map, feat_map.size(2)).squeeze(-1) for feat_map in feat_maps_q], 1)

        prob = F.sigmoid(self.out(feat_map2_q)).squeeze()
        return prob

    def predict_raw_text(self, raw_txt):
       
        # input_batch = 
        pass
   
    def predict(self, batch):
        
        input_batch = batch["input_batch"]
        label = batch["label"]
        prob = self.forward(input_batch)
        return prob

    def evaluate(self, dev):
        acc = [] 
        for batch in dev:
            input_batch = batch["input_batch"]
            label = batch["label"]
            prob = self.forward(input_batch)
            acc.append(((prob > 0.5).long() == label.long()).float().sum().data[0].cpu() * 1.0 / label.size(0))

        return sum(acc) / len(acc)


    def train_step(self, batch):

        input_batch = batch["input_batch"]
        label = batch["label"]
        prob = self.forward(input_batch) 
        loss =  F.binary_cross_entropy(prob, label)
        loss.backward()
        acc = ((prob > 0.5).long() == label.long()).float().sum() * 1.0 / label.size(0) 

        return loss.data[0], acc
