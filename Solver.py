from network import *
import time
import copy
import numpy as np
import random
import os
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import torch.nn.init as init

class Solver(object):
    def __init__(self, model, optim, loss_function, data, batch_size, epoch, reg_rate=0.2,
                 save_path=os.path.join('model', 'new_model')):

        self.model = model
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']
        self.batch_size = batch_size
        self.epoch_num = epoch
        self.num_class = 4
        self.reg_rate = reg_rate
        self.optim = optim
        self.loss = loss_function
        self.save_path = save_path
        self.best_model = copy.deepcopy(model)

        self.train_acc_his = []
        self.val_acc_his = []
        self.train_loss_his = []
        self.save_check_epoch = 20

        self.initial_model()

    def initial_model(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                init.constant(m.bias, 0)

    def calculate_acc(self, y_pred_vector, y_label):
        y_pred = np.argmax(y_pred_vector.cpu().data.numpy(), axis=1)
        accuracy = np.sum(y_pred == y_label) / float(y_label.size)
        return accuracy

    def get_val_acc(self):
        y_pred_vector = self.model.forward(Variable(Tensor(self.X_val)))
        val_acc = self.calculate_acc(y_pred_vector, self.y_val)
        return val_acc

    def get_test_acc(self):
        y_pred_vector = self.best_model.forward(Variable(Tensor(self.X_test)))
        test_acc = self.calculate_acc(y_pred_vector, self.y_test)
        return test_acc

    def get_reg_loss(self):
        reg_loss = 0.0
        if self.reg_rate:
            for W in self.model.parameters():
                reg_loss += torch.sum(W * W)
        return reg_loss*self.reg_rate

    def train(self, random_choice=False):
        best_val_accuracy = 0.0
        data_len = self.y_train.size
        batch_num = data_len // self.batch_size

        plt.ion()
        for epoch in range(0, self.epoch_num):
            epoch_start_time = time.time()
            for batch_epoch in range(0, batch_num):

                if random_choice:
                    choice_idnex = random.sample(range(0, data_len), self.batch_size)
                    batch_train = self.X_train[choice_idnex]
                    batch_label = self.y_train[choice_idnex]
                else:
                    batch_epoch_start = self.batch_size * batch_epoch
                    batch_epoch_end = batch_epoch_start + self.batch_size

                    batch_train = self.X_train[batch_epoch_start: batch_epoch_end]
                    batch_label = self.y_train[batch_epoch_start: batch_epoch_end]

                onehot_label = np.zeros((self.batch_size, self.num_class))
                onehot_label[np.arange(self.batch_size), batch_label] = 1

                batch_train, batch_label_onehot = Variable(Tensor(batch_train)), Variable(Tensor(onehot_label))
                y_pred_vector = self.model.forward(batch_train)

                reg_loss = self.get_reg_loss()
                train_loss = self.loss(y_pred_vector, batch_label_onehot) + reg_loss

                val_acc = self.get_val_acc()
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    self.best_model = copy.deepcopy(self.model)

                self.optim.zero_grad()
                train_acc = self.calculate_acc(y_pred_vector, batch_label)
                train_loss.backward()
                self.optim.step()

                self.train_acc_his.append(train_acc)
                self.val_acc_his.append(val_acc)
                self.train_loss_his.append(float(train_loss))

            if not epoch % self.save_check_epoch:
                self.save_model()
            self.print_figure()
            epoch_end_time = time.time()

            print('Epoch: ', epoch,
                  ' |training loss %.4f' % train_loss,
                  ' |training accuracy %.4f' % train_acc,
                  ' |cost %.4f seconds' % (epoch_end_time - epoch_start_time),
                  ' |best validation acc : %.4f' % best_val_accuracy)
        plt.ioff()
        self.save_training_his()
        self.save_figure()
        self.save_model()

    def print_figure(self):
        plt.figure(1, figsize=(20, 10))
        plt.clf()
        plt.title('Accuracy over steps')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.plot(self.train_acc_his)
        plt.plot(self.val_acc_his)
        plt.legend(['train_accuracy', 'val_accuracy'])

        plt.figure(2, figsize=(20, 10))
        plt.clf()
        plt.title('Training loss over steps')
        plt.xlabel('iterations')
        plt.ylabel('training loss')
        plt.plot(self.train_loss_his)
        plt.legend(['training_loss'])
        plt.pause(0.1)
        plt.show()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        save_file = os.path.join(self.save_path, 'network_model.dat')
        torch.save(self.best_model, save_file)
        print('best model saved in ', save_file)

    def save_figure(self):

        plt.figure(1, figsize=(20, 10))
        plt.clf()
        plt.title('Accuracy over epoch')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(self.train_acc_his)
        plt.plot(self.val_acc_his)
        plt.legend(['train_accuracy', 'test_accuracy'])
        plt.savefig(os.path.join(self.save_path, 'accuracy.png'))

        plt.figure(2, figsize=(20, 10))
        plt.clf()
        plt.title('Training loss over epco')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        plt.plot(self.train_loss_his)
        plt.legend(['training_loss'])
        plt.savefig(os.path.join(self.save_path, 'loss.png'))

    def save_training_his(self):
        np.save(os.path.join(self.save_path, 'train_acc_his'), self.train_acc_his)
        np.save(os.path.join(self.save_path, 'val_acc_his'), self.val_acc_his)
        np.save(os.path.join(self.save_path, 'training_los_his'), self.train_loss_his)