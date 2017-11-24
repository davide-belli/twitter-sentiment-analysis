
def parse(input,output):
    # sentences=[]
    # targets=[]
    lines=[]
    n=0
    with open('./'+input, 'r') as f:
        for line in f:
            #print(line)
            n+=1
            print(n)
            if input == 'source_test.tsv':
                _,target,sentence= line.strip().split("\t") 
            else:
                _,target,sentence = line.strip().split("\t") #NOTE in 2013 there is one more ID at beginning of each tweet
            # targets.append(target)
            # sentences.append(sentence)
            lines.append(target+"|_|"+sentence+"\n")
            print(sentence)
    
    with open('./'+output,'w') as f:
        for line in lines:
            f.write(line)
       
train = 'source_train.tsv'
valid = 'source_valid.tsv'
test = 'source_test.tsv'

parse(train,'unpad_train.txt')
parse(valid,'unpad_valid.txt')
parse(test,'unpad_test.txt')
#parse('source_2017.tsv','unpad_2017.txt')