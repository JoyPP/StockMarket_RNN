import os
import cPickle
import time
import fasttext
import argparse

import torch
import torch.backends.cudnn as cudnn

from data_loader import data_loader, fasttext_model_pretraining
from model import LSTMModel, CNN_LSTMModel, GRU_LSTMModel
from plot import plot
from train import train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch  Training')
    parser.add_argument('--feature_dim', default=100, type=int, help='feature dimension')
    parser.add_argument('--input_size', default=100, type=int, help='input size of lstm model')
    parser.add_argument('--gru_hidden_size', default=150, type=int, help='hidden size of gru layer')
    parser.add_argument('--lstm_hidden_size', default=200, type=int, help='hidden size of lstm layer')
    parser.add_argument('--max_epoch', default=100, type=int, help='max_epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--window_size', default=10, type=int, help='window size')
    parser.add_argument('--kernel_num', default=32, type=int, help='number of each kernel')
    parser.add_argument('--kernel_sizes', default='3,4,5', type=str, help='kernel sizes')
    parser.add_argument('--time_interval', default=7, type=int, help='prediction time interval')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout (0 = no dropout) [default: 0.5]')
    parser.add_argument('--data_directory', default='dataset/', type=str, help='data directory')
    parser.add_argument('--cuda_able', action='store_true', help='enables cuda')
    args = parser.parse_args()

    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    use_cuda = torch.cuda.is_available() and args.cuda_able
    #use_cuda = False
    print 'use_cuda = ', use_cuda

    # get fastText model
    if not os.path.exists('model.bin'):
        model = fasttext_model_pretraining()
    else:
        model = fasttext.load_model('model.bin')

    # get symbols
    if os.path.exists(args.data_directory) and os.path.isdir(args.data_directory):
        symbols = [f[:-5] for f in os.listdir(args.data_directory) if f.endswith('.xlsx')]
    global train_dataset, test_dataset
    train_dataset, test_dataset, dataset_path = data_loader(model, args)
    print '#batch in training dataset is ', len(train_dataset)
    print '#batch in test dataset is ', len(test_dataset)

    # build model
    print 'Building model...'
    net = GRU_LSTMModel(args)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True


    train_acc, test_acc = [], []
    for epoch in range(0, args.max_epoch):
        print 'Epoch %d' % epoch
        epoch_start = time.time()
        train_dataset, test_dataset
        train_acc.append(train(net, train_dataset, use_cuda, args))
        test_acc.append(test(net, test_dataset, use_cuda))
        print 'Each epoch costs ', time.time() - epoch_start

    # save the model
    lstm_model_path = dataset_path + 'lstm.model'
    with open(lstm_model_path, 'w') as f:
        cPickle.dump(net, f)

    # plot the accuracy and save it into dataset_path
    plot(train_acc, test_acc, dataset_path)

    print 'Finish...'

    '''

    args.feature_dim = 5
    args.input_size = 10
    args.hidden_size = 20
    args.kernel_num = 7
    args.batch_size = 3

    seq_len = 11



    net = CNN_LSTMModel(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    x = Variable(torch.rand(args.batch_size, seq_len, args.feature_dim))
    out = net(x)


'''

















