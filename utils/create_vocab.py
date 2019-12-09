save_valid_file = "../dataset/lcsts/valid.txt"
save_train_file = "../dataset/lcsts/train.txt"


import jieba
with open(save_train_file, "r") as f:
    words = []
    for i, line in enumerate(f.readlines()):
        if i % 10 == 0:
            print(i)
        elements = line.strip().split("\t")
        if len(elements) != 2:
            continue
        headline, article = elements[0], elements[1]
        words.extend(list(jieba.cut(headline, cut_all=False)))
        words.extend(list(jieba.cut(article, cut_all=False)))
from collections import Counter
counter =  Counter(words)
with open("../dataset/lcsts/vocab.dict.50000", "w") as f:
    f.write("".join([ w+"\t"+str(c)+"\n"  for w, c in counter.most_common(50000)]))
