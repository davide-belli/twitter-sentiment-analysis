import re
import string
from nltk.tokenize import TweetTokenizer

def find_emoticons(sentence):
    smile_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]] # Mouth
        )"""

    sad_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-']? # Nose (optional)
            [\(\[\|] # Mouth
        )"""
    funny_emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [OpP] # Mouth
        )"""
    sentence = re.sub(smile_emoticons_str, "<smile>", sentence)
    sentence = re.sub(sad_emoticons_str, "<sad>", sentence)
    sentence = re.sub(funny_emoticons_str, "<funny>", sentence)

    return sentence

def preprocess(inp, output):
    sentences=[]
    targets=[]
    printable = set(string.printable)
    tknzr = TweetTokenizer(reduce_len=True)
    
    with open('./'+inp, 'r') as f:
        for line in f:
            target, tmp = line.strip().split("|_|")
            tmp = tmp.lower()
            tmp = "".join(filter(lambda x: x in printable, tmp))
            tmp = re.sub("https?:?//[\w/.]+", "<URL>", tmp)
            tmp = find_emoticons(tmp)
            tmp = re.sub('[\-_"#@(),!*;:.]', " ", tmp)
            tmp = tmp.replace("?", "")
            # tmp = tmp.replace("/", "")
            tmp = tmp.replace("&", "")
            tmp = tmp.replace("'s", "")
            tmp = tmp.replace("&quot", "")
            tmp = tmp.replace("&amp", "")
            tmp = tmp.replace("'", " ")
            tmp = tmp.replace("\t", " ")
            # tkn = tknzr.tokenize(tmp)
            # tmp = " ".join(tkn)
            tmp_list = tmp.split()
            tmp = " ".join(tmp_list)
            targets.append(target)
            sentences.append(tmp)
    
    with open('./'+output, 'w') as f:
        for i in range(len(targets)):
            f.write(targets[i]+"|_|"+sentences[i]+"\n")
      
train = 'original_train.txt'
valid = 'original_valid.txt'
test = 'original_test.txt'

preprocess(test, 'test.txt')
preprocess(train, 'train.txt')
preprocess(valid, 'valid.txt')
