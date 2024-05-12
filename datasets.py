import torch
from torch.utils.data import Dataset
from torch import from_numpy, fft
import numpy as np
import h5py

def get_openset_cis_lora(dataset_path=None,
                         devices_range=None,
                         datasets_indices_old=None,
                         samples_file=None
                         ):  # (1,8192,2)
    train_file = 'Train/dataset_training_no_aug.h5'  # 1-30 1000  # 'Test/dataset_residential.h5' #31-40 400
    test_files = ['Test/dataset_seen_devices.h5',  # 1-30 400
                  'Test/dataset_residential.h5',  # 31-40
                  'Test/dataset_rogue.h5',  # 41-45 200
                  'Test/dataset_other_device_type.h5']  # 46-60 400
    
    if (devices_range['val_uuc_max'] is not None and devices_range['val_uuc_max'] >= 60) or (
            devices_range['test_uuc_max'] is not None and devices_range['test_uuc_max'] >= 60):
        raise ValueError('[dataset] cis dataset only support device id < 60')
    data_shape = (1, 8192, 2)
    # load samples
    if samples_file is not None:
        f_sample = h5py.File(samples_file, "r")
        x_sample = np.array(f_sample['X']).astype(np.float32)
        y_sample = np.array(f_sample['Y']).astype(np.int64)
        data_num_sample = len(y_sample)
        data_num_sample_train = int(0.7 * data_num_sample)
        data_num_sample_val = data_num_sample - data_num_sample_train
        if samples_file.split('/')[-1][:5] == 'multi':
            assert y_sample.max() == 2 * (devices_range['known_max'] - devices_range['known_min'] + 1) - 1
        else:
            pass
            # assert y_sample.max()==y_sample.min()
        print('[DATASET] load samples from', samples_file,
              f', Sample num: {data_num_sample}, y range:{y_sample.min()}-{y_sample.max()}')
        f_sample.close()

    data_num = {'train': 0, 'test': 0, 'val': 0, 'val_0': 0, 'val_1': 0, 'test_0': 0, 'test_1': 0}
    data = {}
    need_val_uuc = True if devices_range['val_uuc_min'] is not None else False
    need_test_uuc = True if devices_range['test_uuc_min'] is not None else False

    label_start = 1
    datasets_indices = {}
    device_id_start = min(
        [devices_range['known_min'], devices_range['val_uuc_min'] if need_val_uuc else devices_range['known_min'],
         devices_range['test_uuc_min'] if need_test_uuc else devices_range['known_min']])
    with h5py.File(dataset_path + train_file, "r") as f:
        subset_num = f['label'].shape[1]
        subset_x = np.array(f['data']).astype(np.float32)
        subset_x = np.concatenate((-subset_x[:, 8192:].reshape((subset_num, 1, 8192, 1)),
                                   subset_x[:, :8192].reshape((subset_num, 1, 8192, 1))), axis=3)
        subset_y = np.array(f['label'][0, :]).astype(np.int64) - label_start

        train_indices = np.where((subset_y >= devices_range['known_min']) &
                                 (subset_y <= devices_range['known_max']))[0]
        data['train_y'] = subset_y[train_indices] - device_id_start
        data['train_x'] = subset_x[train_indices]
        data_num['train'] = len(train_indices)
    # data_test={}
    test_num = 0
    for test_idx in range(len(test_files)):
        test_file = test_files[test_idx]
        with h5py.File(dataset_path + test_file, "r") as f:
            test_num += f['label'].shape[1]
    test_x = np.zeros((test_num, data_shape[0], data_shape[1], data_shape[2]), dtype=np.float32)
    test_y = np.zeros((test_num,), dtype=np.int64)
    test_start = 0
    for test_idx in range(len(test_files)):
        test_file = test_files[test_idx]
        with h5py.File(dataset_path + test_file, "r") as f:
            subset_num = f['label'].shape[1]
            subset_x = np.array(f['data']).astype(np.float32)
            subset_x = np.concatenate((-subset_x[:, 8192:].reshape((subset_num, 1, 8192, 1)),
                                       subset_x[:, :8192].reshape((subset_num, 1, 8192, 1))), axis=3)
            subset_y = np.array(f['label'][0, :]).astype(np.int64) - label_start
            test_x[test_start:test_start + subset_num] = subset_x
            test_y[test_start:test_start + subset_num] = subset_y
            test_start += subset_num
    assert test_start == test_num
    indices_known = np.where((test_y >= devices_range['known_min']) &
                             (test_y <= devices_range['known_max']))[0]
    subset_known_indices = np.arange(len(indices_known))
    np.random.shuffle(subset_known_indices)
    known_val_num = int(0.5 * len(indices_known))
    known_test_num = len(indices_known) - known_val_num
    if datasets_indices_old is None:
        datasets_indices['val_0'] = indices_known[subset_known_indices[:known_val_num]]
        datasets_indices['test_0'] = indices_known[subset_known_indices[known_val_num:]]
    else:
        datasets_indices['val_0'] = datasets_indices_old['val_0']
        datasets_indices['test_0'] = datasets_indices_old['test_0']
    if need_val_uuc:
        datasets_indices['val_1'] = \
        np.where((test_y >= devices_range['val_uuc_min']) & (test_y <= devices_range['val_uuc_max']))[0]
    if need_test_uuc:
        datasets_indices['test_1'] = \
        np.where((test_y >= devices_range['test_uuc_min']) & (test_y <= devices_range['test_uuc_max']))[0]
    data_num['val_0'] = known_val_num
    if need_val_uuc:
        data_num['val_1'] = len(datasets_indices['val_1'])
    data_num['test_0'] = known_test_num
    if need_test_uuc:
        data_num['test_1'] = len(datasets_indices['test_1'])
    for s in ['val_0', 'val_1', 'test_0', 'test_1']:
        if (s != 'val_1' or need_val_uuc) and (s != 'test_1' or need_test_uuc):
            data[s + '_x'] = test_x[datasets_indices[s]]
            data[s + '_y'] = test_y[datasets_indices[s]] - device_id_start
    if samples_file is not None:
        data_num['train'] += data_num_sample_train
        data_num['val_0'] += data_num_sample_val
        # todo label of samples
        data['train_x'] = np.concatenate((data['train_x'], x_sample[:data_num_sample_train]), axis=0)
        data['val_0_x'] = np.concatenate((data['val_0_x'], x_sample[data_num_sample_train:]), axis=0)
        data['train_y'] = np.concatenate((data['train_y'], y_sample[:data_num_sample_train]), axis=0)
        data['val_0_y'] = np.concatenate((data['val_0_y'], y_sample[data_num_sample_train:]), axis=0)
    for s in ['x', 'y']:
        if need_val_uuc:
            data[f'val_{s}'] = np.concatenate((data[f'val_0_{s}'], data[f'val_1_{s}']), axis=0)
            del data[f'val_0_{s}'], data[f'val_1_{s}']
        else:
            data[f'val_{s}'] = data[f'val_0_{s}']
        if need_test_uuc:
            data[f'test_{s}'] = np.concatenate((data[f'test_0_{s}'], data[f'test_1_{s}']), axis=0)
            del data[f'test_0_{s}'], data[f'test_1_{s}']
        else:
            data[f'test_{s}'] = data[f'test_0_{s}']
    classnum = devices_range['known_max'] - devices_range['known_min'] + 1
    if samples_file is not None:
        classnum += 1
    data_num['val'] = data_num['val_0'] + data_num['val_1']
    data_num['test'] = data_num['test_0'] + data_num['test_1']
    print('[DATASET]', data_num)
    print(f'[DATASET] known class num: {classnum}, train label range: (', data['train_y'].min(), data['train_y'].max(),
          '), val label max:', data['val_y'].max(), ', test label max:', data['test_y'].max(),
          f'device id start:{device_id_start}')
    f.close()
    return data, classnum, datasets_indices


def get_datasets(dataset_path=None, devices_range=None, data_shape=(1, 16384, 2), datasets_indices=None, samples_file=None, da=None):
    '''r
    datasets_indices: not None if load old dataset split
    samples_file: not None if required load sampled datas as unseen data in training dataset
    '''
    closet_class_num = devices_range['known_max'] - devices_range['known_min'] + 1

    data, train_class_num, datasets_indices = get_openset_cis_lora(dataset_path=dataset_path,
                                                                   devices_range=devices_range,
                                                                   datasets_indices_old=datasets_indices,
                                                                   samples_file=samples_file)
    test_dataset = sliceDataset(data['test_x'], data['test_y'], data_shape=data_shape, da=da, test=True, closet_class_num=closet_class_num)
    if len(da.keys())>0:
        train_dataset = sliceDataset(data['train_x'], data['train_y'], data_shape=data_shape, da=da, test=True, closet_class_num=closet_class_num)
    else:
        train_dataset = sliceDataset(data['train_x'], data['train_y'], data_shape=data_shape, da=da, closet_class_num=closet_class_num)
    val_dataset = sliceDataset(data['val_x'], data['val_y'], data_shape=data_shape, da=da, closet_class_num=closet_class_num)

    return train_dataset, val_dataset, test_dataset, train_class_num, datasets_indices


class sliceDataset(Dataset):
    def __init__(self, X, Y, data_shape=(256, 64, 2), downsample=1, da={}, start=0, test=False, closet_class_num=0):
        self.X = X
        self.Y = Y
        self.slice_len = data_shape[1]
        self.downsample = downsample
        self.sample_num = len(Y)
        self.da = da
        self.data_len = data_shape[0] * data_shape[1]
        self.channels = self.data_len // self.slice_len // self.downsample
        self.da = da
        self.start = start
        self.test = test
        self.closet_class_num = closet_class_num

    def update_data_len(self, data_len):
        self.data_len = int(data_len)

    def __len__(self):
        return self.sample_num

    def set_out_format(self, snr=99):
        self.snr = snr

    def __getitem__(self, index):
        wave = self.X[index]
        label = self.Y[index]
        wave = wave / np.std(wave)

        if 'pa' in self.da.keys():
            wave = panolinear(wave, self.da['pa'])
        wave = wave / np.std(wave)
        wave = from_numpy(wave)
        indices = np.arange(self.start, self.start + self.data_len, self.downsample)
        wave = wave[:, indices, :]
        wave = wave.reshape((self.channels, self.slice_len, 2))
        pred_label = label % self.closet_class_num
        sim_label = label // self.closet_class_num
        if self.test:
            y = label
        else:
            y = from_numpy(np.array([pred_label, sim_label], dtype=np.int64))
        return wave, y



def panolinear(x, a=1.5):
    amp = np.random.uniform(a, a + 0.3, (4))
    amp = 10 ** (-amp)
    phase = np.random.uniform(low=0.0, high=2 * np.pi, size=(4))
    pa = amp * np.exp(1j * phase)
    wave = x[:, :, 0] + 1j * x[:, :, 1]
    # x+x^3 + x[n-1] + x[n-1]^3 + +x^2+ x[n]*x[n-1] +x[n-1]^2
    wave = wave + pa[0] * wave ** 3 + pa[1] * np.concatenate([wave[:, 0:1] * 0, wave[:, :-1]], axis=1) + pa[
        2] * np.concatenate([wave[:, 0:1] * 0, wave[:, :-1]], axis=1) ** 3
    x = np.concatenate([wave.real[:, :, np.newaxis], wave.imag[:, :, np.newaxis]], axis=2)
    # x = x + pa[0]*x*x +pa[1]*x*x*x
    return x





if __name__ == '__main__':
    devices_range = {'known_min': 0, 'known_max': 9,
                     'val_uuc_min': 10, 'val_uuc_max': 39,
                     'test_uuc_min': 40, 'test_uuc_max': 59}
    train_dataset, val_dataset, test_dataset, train_class_num, datasets_indices = get_datasets(
        data_shape=(1, 256 * 32, 2), devices_range=devices_range, dataset='cislora')