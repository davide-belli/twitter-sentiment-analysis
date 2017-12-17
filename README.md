# twitter-sentiment-analysis
Comparing various Neural Models in Sentiment Analysis tasks on Twitter datasets

python main.py --cuda --epochs 20
(run 20 epochs)

Command line arguments description for 'main.py':

'--data'            default='./data/dataset/preprocessed/'        'location of the data corpus'
'--model'           default='LSTM'                               'current net (RNN_TANH, RNN_RELU, LSTM, GRU, LSTM_BIDIR, LSTM_REV, RAN, RAN_BIDIR, CNN)'
'--emsize'          default=300,                                 'size of word embeddings'
'--nhid'            default=200,                                 'number of hidden units per layer'
'--nlayers'         default=2,                                   'number of layers'
'--nreduced'        default=30,                                  'number of units in the reduced layer'
'--lr',             default=0.01,                                'initial learning rate'
'--lamb',           default=0.1,                                 'lambda for L2 regularization (weight decay)'
'--lrdecay',        default=0.0,                                 'learning rate decay parameter for Adagrad'
'--clip',           default=0.25,                                'gradient clipping'
'--epochs'          default=40,                                  'upper epoch limit'
'--batch_size'      default=20,                                  'batch size'
'--bptt'            default=35,                                  'sequence length'
'--dropout',        default=0.5,                                 'dropout applied to layers (0 = no dropout)'
'--seed'            default=1111,                                'random seed'
'--log-interval'    default=50,                                  'report interval'
'--save'            default='model.pt'                           'path to save the final model'
'--recallsave'      default='model_recall.pt'                    'path to save the final model'
'--pause_value'     default=0,                                   'not optimise embeddings for the first 5 epochs'
'--initial'         default=None,                                'path to embedding file. If not set they are initialized randomly'

Booleans
'--tied' = 'tie the word embedding and softmax weights'
'--plot' = 'plot confusion matrix'
'--last' = 'backpropagate at the end of a tweet'
'--pre' = 'use preprocessed data'
'--pause' = 'not optimise embeddings for the first 5 epochs'
'--cuda' = 'use CUDA'
'--shuffle' = 'shuffle train data every epoch'
