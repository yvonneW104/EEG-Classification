import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNBase
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class FC_Net(nn.Module):
    def __init__(self, input_size=22*1000, name='FC_Net'):
        super(FC_Net, self).__init__()
        self.model_name = name
        self.fc1 = nn.Linear(input_size, 4)
        #self.fc2 = nn.Linear(1024, 4)
        #self.fc3 = nn.Linear(256, 64)
        #self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = F.relu(self.fc2(x))

        #x = F.sigmoid(self.fc3(x))
        #x = F.softmax(self.fc4(x), dim=1)
        x = F.softmax(x, dim=1)
        return x

    def get_name(self):
        return self.model_name


class CnnNet(nn.Module):
    def __init__(self, model_name='CnnNet'):
        super(CnnNet, self).__init__()
        self.model_name = model_name
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        self.fc1 = nn.Linear(496, 4)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        x = x.view(x.size(0), -1)
        x = F.softmax(self.fc1(x),dim=1)
        return x

    def get_name(self):
        return self.model_name


class CnnNet1d(nn.Module):
    def __init__(self, model_name='CnnNet1d'):

        super(CnnNet1d, self).__init__()
        #1000
        self.model_name = model_name
        self.conv1 = nn.Conv1d(22, 32, 37, padding=0, stride=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.pooling1 = nn.MaxPool1d(2)
        #482
        self.conv2 = nn.Conv1d(32, 64, 31, padding=0, stride=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.pooling2 = nn.MaxPool1d(2)
        #226
        self.conv3 = nn.Conv1d(64, 128, 27, padding=0, stride=1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.pooling3 = nn.MaxPool1d(2)
        100
        # self.conv4 = nn.Conv1d(128, 256, 25, padding=0, stride=1)
        # self.batchnorm4 = nn.BatchNorm1d(256)
        # self.pooling4 = nn.MaxPool1d(2)
        # 38
        # self.conv5 = nn.Conv1d(256, 512, 9, padding=0, stride=1)
        # self.batchnorm5 = nn.BatchNorm1d(512)
        # self.pooling5 = nn.MaxPool1d(2)
        # 7
        # self.fc1 = nn.Linear(512*15, 4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.2)
        x = self.pooling2(x)

        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.2)
        x = self.pooling2(x)

        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.2)
        x = self.pooling3(x)

        # x = F.relu(self.conv4(x))
        # x = self.batchnorm4(x)
        # x = F.dropout(x, 0.2)
        # x = self.pooling4(x)
        #
        # x = F.relu(self.conv5(x))
        # x = self.batchnorm5(x)
        # x = F.dropout(x, 0.2)
        # x = self.pooling5(x)
        #
        # x = x.view(x.size(0), -1)
        # x = F.softmax(self.fc1(x), dim=1)
        return x

    def get_name(self):
        return self.model_name


class LSTM_net(nn.Module):
    def __init__(self, name='LSTM_net'):
        super(LSTM_net, self).__init__()
        self.hidden_size = 64
        self.rnn = nn.LSTM(
            input_size=22,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fcnet = FC_Net(input_size = self.hidden_size*1000)
        self.name = name

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x, (h_n, h_c) = self.rnn(x)
        #print('rout: ', x.shape)

        x_reshape = x.contiguous().view(x.size(0), -1)
        x = self.fcnet.forward(x_reshape)

        return x

    def get_name(self):
        return self.name


class LSTM_CNN_net(nn.Module):
    def __init__(self, model_name='LSTM_CNN_net'):
        super(LSTM_CNN_net, self).__init__()
        self.hidden_size = 22
        self.rnn = nn.LSTM(
            input_size=22,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.cnn = CnnNet1d()
        self.model_name = model_name

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x, (h_n, h_c) = self.rnn(x)
        #print('rout: ', x.shape)
        x = x.permute(0, 2, 1)
        #x_reshape = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        #print('reshape_size: ', x_reshape.size())
        x = self.cnn.forward(x)

        return x

    def get_name(self):
        return self.model_name


class CRNN(nn.Module):
    def __init__(self, model_name='CRNN'):
        super(CRNN, self).__init__()
        self.model_name = model_name
        self.conv = CnnNet1d()

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.out1 = nn.Linear(128*100, 4)


    def forward(self, x):

        x = self.conv.forward(x)
        x = x.permute(0, 2, 1)
        r_out, _ = self.rnn(x, None)
        r_out = F.dropout(r_out, 0.25)
        out = F.softmax(self.out1(r_out.view(r_out.size(0), -1)))
        return out

    def get_name(self):
        return self.model_name