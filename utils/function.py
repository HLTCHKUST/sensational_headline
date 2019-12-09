import numpy as np

def harmonic_mean(r1, r2, beta):

    return (1. + beta ** 2) * (r1 * r2) / (beta ** 2 * r1 + r2) 

def cosine_similarity(vector1, vector2):
 
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)) + 1e-8)


def load_lexicons(lexicon_path):

    with open(lexicon_path+"/positive.txt", "r") as f:
        pos = [line.strip() for line in f.readlines()]

    with open(lexicon_path+"/negative.txt", "r") as f:
        neg = [line.strip() for line in f.readlines()]

    return set(pos + neg)

def load_va_lexicons(sentiment_lexicon_file):
    word2valence = {}
    word2arousal = {}
    with open(sentiment_lexicon_file, "r") as f:
        for line in f.readlines()[1:]:
            elements = line.strip().split(u",") # No.,Word,Valence_Mean,Valence_SD,Arousal_Mean,Arousal_SD,Frequency
            word = elements[1]
            word2valence[word] = float(elements[2])
            word2arousal[word] = float(elements[4])

    return {"word2valence": word2valence, "word2arousal": word2arousal}
