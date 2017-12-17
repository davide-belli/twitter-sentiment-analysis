#!/bin/bash

source activate nlp1



echo "RAN emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RAN --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "RAN_BI emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RAN_BIDIR --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "LSTM_BI emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM_BIDIR --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "LSTM_REV emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM_REV --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "LSTM emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "GRU emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model GRU --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "RNN_TANH emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RNN_TANH --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "RNN_RELU emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RNN_RELU --batch_size 20 --initial ./data/embeddings.csv --pause_value 5
echo "CNN emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model CNN --batch_size 20 --initial ./data/embeddings.csv --pause_value 5


echo "RAN no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RAN --batch_size 20
echo "RAN_BI no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RAN_BIDIR --batch_size 20
echo "LSTM_BI no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM_BIDIR --batch_size 20
echo "LSTM_REV no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM_REV --batch_size 20
echo "LSTM no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model LSTM --batch_size 20
echo "GRU no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model GRU --batch_size 20
echo "RNN_TANH no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RNN_TANH --batch_size 20
echo "RNN_RELU no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model RNN_RELU --batch_size 20
echo "CNN no_emb word"
python main.py --cuda --plot --epochs 30 --lr 0.005 --lamb 0.0 --model CNN --batch_size 20


echo "RAN no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RAN --batch_size 100 --last
echo "RAN_BI no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RAN_BIDIR --batch_size 100 --last
echo "LSTM_BI no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM_BIDIR --batch_size 100 --last
echo "LSTM_REV no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM_REV --batch_size 100 --last
echo "LSTM no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM --batch_size 100 --last
echo "GRU no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model GRU --batch_size 100 --last
echo "RNN_TANH no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RNN_TANH --batch_size 100 --last
echo "RNN_RELU no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RNN_RELU --batch_size 100 --last
echo "CNN no_emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model CNN --batch_size 100 --last


echo "RAN nemb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RAN --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "RAN_BI emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RAN_BIDIR --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "LSTM_BI emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM_BIDIR --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "LSTM_REV emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM_REV --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "LSTM emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model LSTM --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "GRU emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model GRU --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "RNN_TANH emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RNN_TANH --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "RNN_RELU emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model RNN_RELU --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
echo "CNN emb last"
python main.py --cuda --plot --epochs 300 --lr 0.005 --lamb 0.0 --model CNN --batch_size 100 --last --initial ./data/embeddings.csv --pause_value 5
