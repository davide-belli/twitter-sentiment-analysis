
def parse(input,output):
    # sentences=[]
    # targets=[]
    lines=[]
    with open('./'+input, 'r') as f:
        for line in f:
            _,_,target,sentence = line.split("\t")
            # targets.append(target)
            # sentences.append(sentence)
            lines.append(target+"|_|"+sentence)
            print(lines[-1])
    
    with open('./'+output,'w') as f:
        for line in lines:
            f.write(line)
            
parse('source_train.tsv','unpad_train.txt')
parse('source_valid.tsv','unpad_valid.txt')
parse('source_test.tsv','unpad_test.txt')