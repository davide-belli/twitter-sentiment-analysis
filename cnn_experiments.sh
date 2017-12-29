#!/bin/bash

source activate nlp1


echo "CNN emb word"
#python main.py --plot --epochs 20 --lr 0.001 --lrdecay 0.2 --lamb 0.0 --model CNN --batch_size 6 --last --pause_value 5 --initial ./data/embeddings.csv

python main.py --plot --epochs 30 --lr 0.0001 --lamb 0.0001 --model CNN --batch_size 20 --last --pause_value 5 --initial ./data/embeddings.csv

echo "CNN no_emb word"
python main.py --plot --epochs 30 --lr 0.0001 --lamb 0.0001 --model CNN --batch_size 20 --last


echo "CNN emb last"
python main.py --plot --epochs 300 --lr 0.0005 --lamb 0.0001 --model CNN --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 10

echo "CNN no_emb last"
python main.py --plot --epochs 300 --lr 0.01 --lamb 0.001 --model CNN --batch_size 100 --last

