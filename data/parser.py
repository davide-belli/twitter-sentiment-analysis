
def parse(input,output):
    # sentences=[]
    # targets=[]
    lines=[]
    with open('./'+input, 'r') as f:
        for line in f:
            _,target,sentence = line.split("\t") #NOTE in 2013 there is one more ID at beginning of each tweet
            # targets.append(target)
            # sentences.append(sentence)
            lines.append(target+"|_|"+sentence)
            print(lines[-1])
    
    with open('./'+output,'w') as f:
        for line in lines:
            f.write(line)
            
#parse('source_train.tsv','unpad_train.txt')
#parse('source_valid.tsv','unpad_valid.txt')
#parse('source_test.tsv','unpad_test.txt')
parse('source_2017.tsv','unpad_2017.txt')