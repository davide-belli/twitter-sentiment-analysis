# twitter-sentiment-analysis
### Comparing various Neural Models in Sentiment Analysis tasks on Twitter datasets
Project for the course Natural Language Processing 1 @ University of Amsterdam

[Davide Belli](https://github.com/davide-belli), [Gabriele Cesa](https://github.com/Gabri95), [Linda Petrini](https://github.com/LindaPetrini)

[Paper](https://github.com/davide-belli/twitter-sentiment-analysis/blob/master/documents/language-models-twitter.pdf)



```
python main.py --cuda --epochs 20 
```
See experiment_final.sh for more examples

### Command line arguments description for 'main.py':

* '--data'            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default='./data/dataset/preprocessed/'        'location of the data corpus'
* '--model'           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default='LSTM'                               
  1. RNN_TANH
  2. RNN_RELU
  3. LSTM
  4. GRU
  5. LSTM_BIDIR
  6. LSTM_REV
  7. RAN
  8. RAN_BIDIR
  9. CNN
* '--emsize'          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=300,                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'size of word embeddings'
* '--nhid'            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=200,                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'number of hidden units per layer'
* '--nlayers'         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=2,                                   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'number of layers'
* '--nreduced'        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=30,                                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'number of units in the reduced layer'
* '--lr',             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0.01,                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'initial learning rate'
* '--lamb',           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0.1,                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'lambda for L2 regularization (weight decay)'
* '--lrdecay',        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0.0,                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'learning rate decay parameter for Adagrad'
* '--clip',           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0.25,                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'gradient clipping'
* '--epochs'          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=40,                                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'upper epoch limit'
* '--batch_size'      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=20,                                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'batch size'
* '--bptt'            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=35,                                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'sequence length'
* '--dropout',       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0.5,                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'dropout applied to layers (0 = no dropout)'
* '--seed'            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=1111,                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'random seed'
* '--log-interval'    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=50,                                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'report interval'
* '--save'            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default='model.pt'                           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'path to save the final model'
* '--recallsave'      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default='model_recall.pt'                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'path to save the final model'
* '--pause_value'     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=0,                                   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'not optimise embeddings for the first 5 epochs'
* '--initial'         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;default=None,                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'path to embedding file. If not set they are initialized randomly'

### Booleans
* '--tied' = 'tie the word embedding and softmax weights'
* '--plot' = 'plot confusion matrix'
* '--last' = 'backpropagate at the end of a tweet'
* '--pre' = 'use preprocessed data'
* '--pause' = 'not optimise embeddings for the first 5 epochs'
* '--cuda' = 'use CUDA'
* '--shuffle' = 'shuffle train data every epoch'
