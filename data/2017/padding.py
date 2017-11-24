
def data_len(input):
    max_len = 0
    n=0
    with open('./'+input, 'r') as f:
        for line in f:
            n+=1
            print(n)
            target,sentence = line.split("|_|")
            words = sentence.split()
            if len(words)>max_len:
                max_len = len(words)
                max_sentence= words
    print(max_len)
    print(max_sentence)
    return max_len

def add_padding(input,output, max_len):
    sentences=[]
    targets=[]
    #max_len=0
    #max_sentence=""
    with open('./'+input, 'r') as f:
        for line in f:
            target,sentence = line.split("|_|")
            targets.append(target)
            sentences.append(sentence)
            words = sentence.split()
            #print(words)
            #print(len(words),"\n")
            #if len(words)>max_len:
                #max_len = len(words)
                #max_sentence= words
                #print(max_len)
                #print(max_sentence)

    print(max_len)
    #print(max_sentence[5], max_sentence[-1])
    for s in range(len(sentences)):
        sentences[s]=sentences[s][:-2]
        sentences[s]+=" <eos>"
        words = sentences[s].split()
        for i in range(max_len-len(words)+1):
            sentences[s]+=" <pad>"
        sentences[s]+="\n"
    with open('./'+output,'w') as f:
        for i in range(len(targets)):
            f.write(targets[i]+"|_|"+sentences[i])
      
train = 'unpad_train.txt'
valid = 'unpad_valid.txt'
test = 'unpad_test.txt'
a = data_len(train)
b = data_len(valid)
c = data_len(test)
max_len = max(a, b, c)

add_padding(train,'train.txt', max_len)
add_padding(valid,'valid.txt', max_len)
add_padding(test,'test.txt', max_len)
#add_padding('unpad_2017.txt','2017.txt')
