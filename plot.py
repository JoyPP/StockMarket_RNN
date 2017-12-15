import matplotlib.pyplot as plt

def plot(train_acc, test_acc, pkl_path):
    '''
    plot accuracy of training and test data and save it into pkl_path
    :param train_acc:
    :param test_acc:
    :param pkl_path: path to save the fig
    :return:
    '''
    epoches = len(train_acc)
    l1, = plt.plot(range(epoches), train_acc, 'r', label='train')
    l2, = plt.plot(range(epoches), test_acc, 'b', label='test')
    plt.legend(handles=[l1, l2])
    plt.title('Accuracy for all data')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(pkl_path + 'All.png')