
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
            if len(sentence)>max_len:
                max_len = len(sentence)
                max_sentence= sentence

    print(max_len)
    print(max_sentence)
    print("a-"+sentences[0][-1])
    for s in range(len(sentences)):
        sentences[s]=sentences[s][:-2]
        sentences[s]+=" <eos>"
        while len(sentences[s])<max_len:
            sentences[s]+=" <pad>"
        sentences[s]+="\n"
    with open('./'+output,'w') as f:
        for i in range(len(targets)):
            f.write(targets[i]+"|_|"+sentences[i])
            
add_padding('unpad_train.txt','train.txt')
add_padding('unpad_valid.txt','valid.txt')
add_padding('unpad_test.txt','test.txt')
