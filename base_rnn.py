import os
import cPickle
import time
import fasttext
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torch.autograd import Variable
#from utils import progress_bar
from data_loader import data_loader, fasttext_model_pretraining


def train_nopadding(net, use_cuda=False):
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        targets = Variable(torch.LongTensor(targets))
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        print 'loss = ',loss
        start_time =  time.time()
        loss.backward()
        print 'spent time backward = ', time.time() - start_time
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'batch_idx = %d, loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, train_loss/(batch_idx+1), 100.0*correct/total, correct, total)
        #progress_bar(batch_idx, len(train_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_nopadding(net, use_cuda=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        targets = Variable(torch.LongTensor(targets))
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'batch_idx = %d, loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, test_loss/(batch_idx+1), 100.0*correct/total, correct, total)

        #progress_bar(batch_idx, len(test_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def train_padding(net, use_cuda=False):
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    # inputs: (window_size, batch_size, max_len, feature_dim)
    for batch_idx, (inputs, seq_lens, sort_list, targets) in enumerate(train_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        targets = Variable(torch.LongTensor(targets))
        outputs = net(inputs, seq_lens, sort_list)
        loss = criterion(outputs, targets)
        #print 'loss = ',loss
        #start_time = time.time()
        loss.backward()
        #print 'spent time backward = ', time.time() - start_time
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'batch_idx = %d, loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, train_loss/(batch_idx+1), 100.0*correct/total, correct, total)
    return 1.0 * correct / total
        #progress_bar(batch_idx, len(train_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_padding(net, use_cuda=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, seq_lens, sort_list, targets) in enumerate(test_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        targets = Variable(torch.LongTensor(targets))
        outputs = net(inputs, seq_lens, sort_list)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'batch_idx = %d, loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, test_loss/(batch_idx+1), 100.0*correct/total, correct, total)
    return 1.0*correct/total
        #progress_bar(batch_idx, len(test_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        #self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, sequences, seq_lens, sort_lists):
        # sequence: (window_size, batch_size, max_len, feature_dim)
        self.hidden = self.init_hidden()
        for seq, seq_len, sort_list in zip(sequences, seq_lens, sort_lists):
            pack = nn.utils.rnn.pack_padded_sequence(Variable(torch.Tensor(seq)), seq_len, batch_first=True)
            # transfer the self.hidden with same order of seq
            tmp1 = np.zeros(self.hidden[0].size(), dtype=np.float32)
            tmp2 = np.zeros(self.hidden[1].size(), dtype=np.float32)
            np.take(self.hidden[0].data.numpy(), sort_list, axis=1, out=tmp1)
            np.take(self.hidden[1].data.numpy(), sort_list, axis=1, out=tmp2)
            self.hidden = (Variable(torch.from_numpy(tmp1)), Variable(torch.from_numpy(tmp2)))
            # calculate the hidden state
            _, self.hidden = self.lstm(pack, self.hidden)
            # return back to the normal order
            sort2 = np.argsort(sort_list)
            np.take(self.hidden[0].data.numpy(), sort2, axis=1, out=tmp1)
            np.take(self.hidden[1].data.numpy(), sort2, axis=1, out=tmp2)
            self.hidden = (Variable(torch.from_numpy(tmp1)), Variable(torch.from_numpy(tmp2)))

        output = self.softmax(self.linear(self.hidden[0][0]))
        return output

    def init_hidden(self):
        result = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_size)))
        if use_cuda:
            return result.cuda()
        return result


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--max_epoch', default=0, type=int, help='max_epoch')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    '''

    use_cuda = False
    batch_size = 16
    windown_size = 10

    # get fastText model
    if not os.path.exists('model.bin'):
        model = fasttext_model_pretraining()
    else:
        model = fasttext.load_model('model.bin')

    # Load data
    symbol = 'BHP'
    print 'preparing data for ', symbol
    global train_dataset, test_dataset
    if os.path.exists(symbol+'_train.pkl'):
        print 'read datasets from pkl files.'
        with open(symbol+'_train.pkl', 'r') as f:
            train_dataset = cPickle.load(f)
        with open(symbol + '_test.pkl', 'r') as f:
            test_dataset = cPickle.load(f)
    else:
        train_dataset, test_dataset = data_loader(symbol, model, directory='dataset/', batch_size=batch_size, window_size=window_size)
        print 'saving dataset into pkl files.'
        with open(symbol+'_train.pkl', 'w') as f:
            cPickle.dump(train_dataset, f)
        with open(symbol+'_test.pkl', 'w') as f:
            cPickle.dump(test_dataset, f)

    #train_dataset = train_dataset[:10]
    print '#sample in training dataset is ', len(train_dataset)
    print '#sample in test dataset is ', len(test_dataset)

    # Check the save_dir exists or not
    # if not os.path.exists(args.save_dir):
    #    os.makedirs(args.save_dir)

    # build model
    print 'Building model...'
    net = LSTMModel(input_size=100, hidden_size=200, output_size=3, batch_size=batch_size)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    train_acc, test_acc = [], []
    for epoch in range(0, 100):
        print 'Epoch %d' % epoch
        epoch_start = time.time()
        train_acc.append(train_padding(net))
        test_acc.append(test_padding(net))
        print 'Each epoch costs ', time.time() - epoch_start

    l1, = plt.plot(range(100), train_acc, 'r', label='train')
    l2, = plt.plot(range(100), test_acc, 'b', label='test')
    plt.legend(handles=[l1, l2])
    plt.title(symbol)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(symbol + '.png')

    print 'Finish...'































