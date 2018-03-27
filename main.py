import preprocess
import torch
from network import *
from Solver import *
import torch.nn as nn
import os

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def main():
    data_dir = 'project_datasets'
    #The shape of data may need to be changed to [-1, 1, 22, 1000] depending on your network

    data = preprocess.load_data(data_dir, data_size=9, data_shape=[-1, 1, 22, 1000], do_nrom=False)
    print('X_train: ', data['X_train'].shape, ' y_train: ', data['y_train'].shape)
    print('X_val: ', data['X_val'].shape, '  y_val: ', data['y_val'].shape)
    print('X_test: ', data['X_test'].shape, '  y_test: ', data['y_test'].shape)

    load = False
    training = True
    reg_rate = 0.1


    #model = LSTM_CNN_net(model_name='LSTM_CNN_net')
    #model = LSTM_CNN_net(model_name='LSTM_Net')
    #model = CnnNet(model_name='Cnn_Net')
    #model = CnnNet1d('CnnNet1d')
    model = CnnNet('Cnn_Net')
    #model = CRNN()
    model_name = model.get_name()

    print('Using model: ', model_name)
    #mode = 'train'
    mode = 'test'

    if mode is 'test':
        load = True
        training = False

    save_path = os.path.join('model', model_name)
    test_data_path = os.path.join(save_path, 'test_data.npy')
    model_path = os.path.join(save_path, 'network_model.dat')

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if load:
        if os.path.isfile(model_path):
            model = torch.load(model_path)
            print('load model from ', model_path)
        else:
            print('can not find model file')
            exit(0)
        if os.path.isfile(test_data_path):
            test_data = np.load(test_data_path)
            data['X_test'] = test_data.item().get('X')
            data['y_test'] = test_data.item().get('y')
            print('Load test data  ', 'X_test: ', data['X_test'].shape, '  y_test: ', data['y_test'].shape)
        else:
            print('Can not find test data')
            exit(0)

    else:
        test_data = {'X': data['X_test'], 'y': data['y_test']}
        np.save(test_data_path, test_data)
        print('save test data in', test_data_path)

    if use_cuda:
        print('cuda is available')
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0.95)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    loss_function = nn.MSELoss()
    #loss_function = nn.CrossEntropyLoss()

    solver = Solver(model=model, optim=optimizer, loss_function=loss_function, data=data, batch_size=200, epoch=100
                    , save_path=save_path, reg_rate=reg_rate)
    if training:
        print('========================= Start training ==========================')
        solver.train(random_choice=True)
        print('========================= Finish training ==========================')

    test_acc = solver.get_test_acc()

    print('test accuracy is %.4f' % test_acc)
    val_acc_his = np.load(os.path.join(save_path, 'val_acc_his.npy'))
    train_acc_his = np.load(os.path.join(save_path, 'train_acc_his.npy'))
    print('train_acc = ', np.max(train_acc_his))
    print('val_acc_his = ', np.max(val_acc_his))
    solver.print_figure()


if __name__ == '__main__':
    main()
