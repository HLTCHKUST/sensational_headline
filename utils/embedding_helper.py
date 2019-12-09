import pickle
import numpy as np

def get_embedding(emb_file, lang, key="word2id"):
    with open(emb_file['vocab_path'], 'rb') as f:
        if key == "word2id":
             word2id = pickle.load(f, encoding="latin1")[key]
        elif key == "id2word":
            id2word = pickle.load(f, encoding="latin1")
            word2id = dict(zip(id2word.values(), id2word.keys()))
        else:
            raise KeyError("keyerror in get_embddding")    
        

    with open(emb_file['embedding_path'],'rb') as f:
        embedding = pickle.load(f, encoding='latin1')
        print(embedding.shape)
        print(lang.n_words)

    emb_size = embedding.shape[1]
    # new_embedding = np.zeros((100, emb_size))
    new_embedding = np.random.randn(lang.n_words, emb_size) * 0.01
    for i in range(lang.n_words):
    # for i in range(100):
        if lang.idx2word[i] in word2id:
            new_embedding[i] = embedding[word2id[lang.idx2word[i]]]

    return new_embedding
