import os
import h5py
import numpy as np


def load_data(data_dir, data_size=9, train_proportion=0.8, num_test=50, data_shape=[-1, 1, 22, 1000], do_nrom=False):
    data = {}
    for i in range(data_size):
        data_name = [fn for fn in os.listdir(data_dir) if fn.startswith('A0%d'%(i+1)) and fn.endswith('.mat')]
        #assert len(data_name) == 1

        data_path = os.path.join(data_dir, data_name[0])
        A0xT = h5py.File(data_path, 'r')
        X_A0xT = np.copy(A0xT['image'])[:, 0:22, :]
        y_A0xT = np.copy(A0xT['type'])[0, 0:X_A0xT.shape[0]:1]

        # remove trails contains nan
        index = [c for c in range(X_A0xT.shape[0]) if ~np.any(np.isnan(X_A0xT[c]))]
        X_A0xT = X_A0xT[index]
        y_A0xT = y_A0xT[index]
        y_A0xT = np.asarray(y_A0xT, dtype=np.int32)
        
        y_A0xT = y_A0xT-769

        # split train, val, test
        if i == 0:
            data = split_data(X_A0xT, y_A0xT, train_proportion, num_test)
        else:
            sub_data = split_data(X_A0xT, y_A0xT, train_proportion, num_test)
            for key, value in data.items():
                data[key] = np.concatenate((data[key], sub_data[key]), axis=0)

    # normalize dataset
    if do_nrom:
        full_dataset = np.concatenate((data['X_train'], data['X_val'], data['X_test']), axis=0)
        mean = np.mean(full_dataset, axis=0)
        std = np.std(full_dataset, axis=0)
        data['X_train'] = (data['X_train'] - mean) / std
        data['X_val'] = (data['X_val'] - mean) / std
        data['X_test'] = (data['X_test'] - mean) / std

    # shuffle train data
    data['X_train'], data['y_train'] = shuffle(data['X_train'], data['y_train'])

    data['X_train'] = data['X_train'].reshape((data_shape))
    data['X_val'] = data['X_val'].reshape((data_shape))
    data['X_test'] = data['X_test'].reshape((data_shape))


    return data

def split_data(X, y, train_proportion=0.8, num_test=50):
    num_train = int(train_proportion * (y.shape[0] - num_test))
    X, y = shuffle(X, y)
    X_train, X_val, X_test = X[num_test:num_train+num_test,:,:], X[num_train+num_test:,:,:], X[:num_test,:,:]
    y_train, y_val, y_test = y[num_test:num_train+num_test], y[num_train+num_test:], y[:num_test]

    sub_data = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test}

    return sub_data

def shuffle(X, y):
    index = np.arange(y.shape[0])
    np.random.shuffle(index)
    X = X[index,:,:]
    y = y[index]
    return X, y


