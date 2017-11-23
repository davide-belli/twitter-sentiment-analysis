
def add_padding(input,output):
    sentences=[]
    targets=[]
    max_len=0
    max_sentence=""
    with open('./'+input, 'r') as f:
        for line in f:
            target,sentence = line.split("|_|")
            targets.append(target)
            sentences.append(sentence)
            words = sentence.split()
            #print(words)
            #print(len(words),"\n")
            if len(words)>max_len:
                max_len = len(words)
                max_sentence= words
                #print(max_len)
                #print(max_sentence)

    print(max_len)
    print(max_sentence)
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
            
#add_padding('unpad_train.txt','train.txt')
#add_padding('unpad_valid.txt','valid.txt')
#add_padding('unpad_test.txt','test.txt')
add_padding('unpad_2017.txt','2017.txt')
