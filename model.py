import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size


        self.lstm = nn.LSTM(args.input_size, args.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_size, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        # sequence: (window_size, batch_size, max_len, feature_dim)
        self.hidden = self.init_hidden()
        _, self.hidden = self.lstm(inputs, self.hidden)
        output = self.softmax(self.fc1(self.hidden[0][0]))
        return output

    def init_hidden(self):
        result = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_size)))
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

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(args.hidden_size, 3)
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
        result = (Variable(torch.zeros(1, self.args.batch_size, self.args.hidden_size)),
                Variable(torch.zeros(1, self.args.batch_size, self.args.hidden_size)))
        return result

