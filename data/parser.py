
import os.path

dataset_folder = './dataset'
original = 'original/'
parsed = 'parsed/'




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
            
            _, target, sentence = line.strip().split("\t")
            
            # targets.append(target)
            # sentences.append(sentence)
            lines.append(target+"\t"+sentence+"\n")
            print(sentence)
    
    with open('./'+output,'w') as f:
        for line in lines:
            f.write(line)
      
      
if not os.path.exists(os.path.join(dataset_folder, parsed)):
    os.makedirs(os.path.join(dataset_folder, parsed))
    
for data in ['train', 'test', 'valid']:
    orig = os.path.join(dataset_folder, original, data+'.tsv')
    pars = os.path.join(dataset_folder, parsed, data+'.txt')
    parse(orig, pars)

#parse('source_2017.tsv','unpad_2017.txt')