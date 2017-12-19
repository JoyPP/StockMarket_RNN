import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.input_size = args.input_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.batch_size = args.batch_size


        self.lstm = nn.LSTM(args.input_size, args.lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_size, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        # sequence: (batch_size, window_size, feature_dim)
        self.hidden = self.init_hidden()
        _, self.hidden = self.lstm(inputs, self.hidden)
        output = self.softmax(self.fc1(self.hidden[0][0]))
        return output

    def init_hidden(self):
        result = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_size)),
                Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_size)))
        return result



class CNN_LSTMModel(nn.Module):
    def __init__(self, args):
        super(CNN_LSTMModel, self).__init__()
        self.args = args

        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, kernel_size=(K, args.feature_dim)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, args.input_size)

        self.lstm = nn.LSTM(args.input_size, args.lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(args.lstm_hidden_size, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # x: Variable(batch_size, window_size, seq_len, feature_dim)
        batch_size, windown_size = x.size(0), x.size(1)
        x = Variable(x.data.resize_(x.size(0)*x.size(1), x.size(2), x.size(3)))

        x = x.unsqueeze(1)  # (batch_size*window_size, 1, seq_len, feature_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size*window_size, Co, seq_len)]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(batch_size*window_size, Co)]*len(Ks)

        x = torch.cat(x, 1)     # (batch_size*window_size, len(Ks) * Co)
        x = self.dropout(x)     # (batch_size*window_size, len(Ks) * Co)
        x = self.fc1(x)    # (batch_size*window_size, input_size)

        self.hidden = self.init_hidden()
        x = Variable(x.data.resize_(batch_size, windown_size, x.size(1))) # (batch_size, window_size, input_size)
        _, self.hidden = self.lstm(x, self.hidden) # (1, window_size, input_size) # in this case, window_size is the seq_len of lstm
        output = self.softmax(self.fc2(self.hidden[0][0]))
        return output

    def init_hidden(self):
        result = (Variable(torch.zeros(1, self.args.batch_size, self.args.lstm_hidden_size)),
                Variable(torch.zeros(1, self.args.batch_size, self.args.lstm_hidden_size)))
        return result




class GRU_LSTMModel(nn.Module):
    def __init__(self, args):
        super(GRU_LSTMModel, self).__init__()
        self.args = args

        self.gru = nn.GRU(args.input_size, args.gru_hidden_size, batch_first=True)

        self.lstm = nn.LSTM(args.gru_hidden_size, args.lstm_hidden_size)
        self.fc2 = nn.Linear(args.lstm_hidden_size, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # x: Variable(batch_size, window_size, seq_len, feature_dim)
        batch_size, windown_size = x.size(0), x.size(1)
        gru_output = None   # would be (window_size, batch_size, gru_hidden_size)
        self.gru_hidden = self.init_gru_hidden()
        for w in range(windown_size):
            _, gru_hidden = self.gru(x[:,w], self.gru_hidden)
            if gru_output is None:
                gru_output = gru_hidden
            else:
                gru_output = torch.cat((gru_output, gru_hidden),0)

        self.lstm_hidden = self.init_lstm_hidden()
        _, self.lstm_hidden = self.lstm(gru_output, self.lstm_hidden) # gru_output: (window_size, batch_size, gru_hidden_size) # in this case, window_size is the seq_len of lstm
        output = self.softmax(self.fc2(self.lstm_hidden[0][0]))
        return output

    def init_gru_hidden(self):
        result = Variable(torch.zeros(1, self.args.batch_size, self.args.gru_hidden_size))
        return result

    def init_lstm_hidden(self):
        result = (Variable(torch.zeros(1, self.args.batch_size, self.args.lstm_hidden_size)),
                Variable(torch.zeros(1, self.args.batch_size, self.args.lstm_hidden_size)))
        return result

