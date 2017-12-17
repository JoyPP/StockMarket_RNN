import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable


def train(net, train_dataset, use_cuda, args):
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(torch.Tensor(inputs)), Variable(torch.LongTensor(targets))
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        #print 'loss = ',loss
        #start_time =  time.time()
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

def test(net, test_dataset, use_cuda):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_dataset):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(torch.Tensor(inputs)), Variable(torch.LongTensor(targets))
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'batch_idx = %d, loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, test_loss/(batch_idx+1), 100.0*correct/total, correct, total)
    return 1.0*correct/total
        #progress_bar(batch_idx, len(test_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

