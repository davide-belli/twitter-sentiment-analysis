import re
import string
from nltk.tokenize import TweetTokenizer

def find_emoticons(sentence):
    smile_emoticons_str = r"""(?:[:=;][oO\-]?[D\)\]])"""
    sad_emoticons_str = r"""(?:[:=;][oO\-']?[\(\[\|])"""
    funny_emoticons_str = r"""(?:[:=;@][oO\-]?[Oo0pP])"""
    sentence = re.sub(smile_emoticons_str, "<smile>", sentence)
    sentence = re.sub(sad_emoticons_str, "<sad>", sentence)
    sentence = re.sub(funny_emoticons_str, "<funny>", sentence)

    return sentence

def preprocess(inp, output):
    sentences=[]
    targets=[]
    tknzr = TweetTokenizer(reduce_len=True)

    with open('./'+inp, 'r', encoding='utf-8') as f:
        for line in f:
            target, tmp = line.strip().split("|_|")
            tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
            tmp = find_emoticons(tmp)
            tmp = re.sub('[\-_"#@(),!*;:.~\[\]]', " ", tmp)
            tmp = tmp.replace("?", "")
            tmp = tmp.replace("&", "")
            tmp = tmp.replace("'s ", " ")
            tmp = tmp.replace("&quot", "")
            tmp = tmp.replace("&amp", "")
            tmp = tmp.replace("'", " ")
            tmp = tmp.replace("\t", " ")
            tmp = tknzr.tokenize(tmp)
            targets.append(target)
            tem = " ".join(tmp)
            sentences.append(tem.lower())

    with open('./'+output, 'w', encoding='utf-8') as f:
        for i in range(len(targets)):
            f.write(targets[i]+"|_|"+sentences[i]+"\n")

train = 'original_train.txt'
valid = 'original_valid.txt'
test = 'original_test.txt'

preprocess(test, 'test.txt')
preprocess(train, 'train.txt')
preprocess(valid, 'valid.txt')
