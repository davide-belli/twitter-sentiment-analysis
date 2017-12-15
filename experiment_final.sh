#!/bin/bash

source activate nlp1

echo "RAN no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 20
echo "RAN_BI no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 20
echo "LSTM_BI no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM_BIDIR --batch_size 20
echo "LSTM_REV no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM_REV --batch_size 20
echo "LSTM no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM --batch_size 20
echo "GRU no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model GRU --batch_size 20
echo "RNN_TANH no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RNN_TANH --batch_size 20
echo "CNN no_emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model CNN --batch_size 20


echo "RAN emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN --batch_size 20 --pre_trained
echo "RAN_BI emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 20 --pre_trained
echo "LSTM_BI emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM_BIDIR --batch_size 20 --pre_trained
echo "LSTM_REV emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM_REV --batch_size 20 --pre_trained
echo "LSTM emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model LSTM --batch_size 20 --pre_trained
echo "GRU emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model GRU --batch_size 20 --pre_trained
echo "RNN_TANH emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model RNN_TANH --batch_size 20 --pre_trained
echo "CNN emb word"
python main.py --cuda --plot --epochs 100 --lr 0.01 --lamb 0.1 --model CNN --batch_size 20 --pre_trained


echo "RAN no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 --last
echo "RAN_BI no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 100 --last
echo "LSTM_BI no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_BIDIR --batch_size 100 --last
echo "LSTM_REV no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_REV --batch_size 100 --last
echo "LSTM no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM --batch_size 100 --last
echo "GRU no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model GRU --batch_size 100 --last
echo "RNN_TANH no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RNN_TANH --batch_size 100 --last
echo "CNN no_emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model CNN --batch_size 100 --last


echo "RAN nemb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RAN --batch_size 100 --last --pre_trained
echo "RAN_BI emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RAN_BIDIR --batch_size 100 --last --pre_trained
echo "LSTM_BI emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_BIDIR --batch_size 100 --last --pre_trained
echo "LSTM_REV emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM_REV --batch_size 100 --last --pre_trained
echo "LSTM emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model LSTM --batch_size 100 --last --pre_trained
echo "GRU emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model GRU --batch_size 100 --last --pre_trained
echo "RNN_TANH emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model RNN_TANH --batch_size 100 --last --pre_trained
echo "CNN emb last"
python main.py --cuda --plot --epochs 1000 --lr 0.01 --lamb 0.1 --model CNN --batch_size 100 --last --pre_trained
