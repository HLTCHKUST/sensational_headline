save_test_file = "./dataset/lcsts/test.txt"
save_valid_file = "./dataset/lcsts/valid.txt"
save_train_file = "./dataset/lcsts/train.txt"


import jieba
def segment(s):
    words = jieba.cut(s, cut_all=False)
    return " ".join(words)


def segment_file(txt_file, save_file):
    with open(save_file, "w") as fw:
      with open(txt_file, "r") as f:
        for line in f.readlines():
            elements = line.strip().split("\t")
            if len(elements) != 2:
                continue
            headline, article = elements[0], elements[1]
            new_h, new_a = segment(headline), segment(article)
            # print(new_h+"\t"+new_a+"\n")
            cont = new_h+"\t"+new_a+"\n"
            fw.write(cont)

segment_file(save_train_file, "./dataset/lcsts/segment_train.txt")
