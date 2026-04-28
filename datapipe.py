version = 2 # 将每个频带视为每个视图

import os 
import numpy as np 
import torch
from torch_geometric.data import Data, Batch, InMemoryDataset, HeteroData
from torch_geometric.utils import dense_to_sparse  # 如果需要生成边
from tqdm import tqdm
import scipy.io as sio
import glob  
from sklearn.model_selection import StratifiedKFold, KFold


cross_number = 15 # faced数据集受试者数量很多，进行交叉验证
classes = 4 # Num. of classes 

def to_categorical(y, num_classes=None, dtype='float32'): 
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None):
        self.stage = stage #Train or test
        self.subjects = subjects  
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        
        #super(EmotionDataset, self).__init__(root, transform, pre_transform)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, self.stage, self.subjects, self.sub_i)]
    def download(self):
        pass
    
    def process(self): 
        data_list = [] 
        # process by samples
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)): 
            data = HeteroData()
            num_nodes = self.X[0][sample_id].shape[0]
            for v in range(5):
                node_type = f'view_{v}'  # 节点类型
                x_v = self.X[v][sample_id, :, :]  # [num_nodes, num_features]
                data[node_type].x = torch.FloatTensor(x_v)  # 为每个节点类型设置 x
                data[node_type].num_nodes = num_nodes  # 显式设置节点数
            data.y = torch.FloatTensor(self.Y[sample_id, :])
            data_list.append(data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data 

def kfold_indices(labels, n_splits=5, stratified=True, shuffle=True, random_state=42):
    """
    返回 [(train_idx, test_idx), ...] 列表
    labels 可以是向量或one-hot矩阵
    """
    y = np.array(labels)
    if y.ndim > 1:
        y_for_split = np.argmax(y, axis=1)
    else:
        y_for_split = y
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(np.zeros(len(y_for_split)), y_for_split))


def get_all_folders(path):
    folders = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            folders.append(os.path.join(root, dir_name))
    return folders


def get_data(data_name, data_path, labels_path):
    if data_name == 'seed':
        label = sio.loadmat(labels_path)['label']
        files = get_all_folders(data_path)
        fa_ll = []
        for f in files:
            fa_ll.append(sorted(glob.glob(f+'/'+'*_*')))
        fa_ll = sum(fa_ll, [])
        sublist = set()
        for f in fa_ll:
            sublist.add(f.split('/')[-1].split('_')[0])
    elif data_name == 'seed4':
        label1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
        label2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
        label3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        files = get_all_folders(data_path)
        fa_ll = []
        for f in files:
            fa_ll.append(sorted(glob.glob(f+'/'+'*_*')))
        fa_ll = sum(fa_ll, [])
        sublist = set()
        for f in fa_ll:
            sublist.add(f.split('/')[-1].split('_')[0])
    elif data_name == 'faced':
        label = sio.loadmat(labels_path)['label']
        files = sorted(glob.glob(data_path+'sub*'))
        sublist = set()
        for f in files:
            sublist.add(f.split('/')[-1].split('.')[0].split('b')[1])
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    sublist = sorted(list(sublist))
    print(sublist)
    sub_mov = [] 
    sub_label = []
    for sub_i in range(len(sublist)):   # 遍历15个受试者
        sub = sublist[sub_i]
        mov_data = 0
        if data_name =='faced':
            sub_files = glob.glob(data_path+'sub'+sub+'*')
            for f in sub_files:
                print(f)
                mov_data = np.load(f, allow_pickle=True)   # (28, 32, 30, 5)（视频数，通道数，时间点，频带特征数）
                mov_data = mov_data.transpose(0, 1, 3, 2)  # (28, 32, 5, 30)
                mov_data = normalize(mov_data)
        else:
            sub_files = []
            for i in range(1, 4):
                sub_files.append(glob.glob(data_path+str(i)+'/'+sub+'_'+'*'))  # 将每个受试者的3次实验数据路径存入sub_files
            sub_files = sum(sub_files, [])  # 展平列表
            sesseion = []
            for f in sub_files:       # 3 session
                print(f)
                data = sio.loadmat(f, verify_compressed_data_integrity=False)
                keys = data.keys()
                de_mov = [k for k in keys if 'de_LDS' in k]  # de_LDS de_movingAve
                mov_datai = []  # one session
                for t in range(24):   # N:  SEED=15 SEEDIV=24
                    temp_data = data[de_mov[t]].transpose(0,2,1)   
                    data_length  = temp_data.shape[-1]
                    mov_i = np.zeros((62, 5, 64))  # SEED 265 IV 64
                    mov_i[:,:,:data_length] = temp_data
                    mov_datai.append(mov_i)
                mov_datai = np.array(mov_datai)  # (N, 62, 5, time_len)
                sesseion.append(mov_datai)
            mov_data = np.vstack(sesseion)   # 将单个人的三次实验数据合并(3*15, 62, 5, time_len)
            mov_data = normalize(mov_data)   # 数据归一化
        sub_mov.append(mov_data)         # 将所有受试者的数据存入sub_mov 15*(3*15, 62, 5, time_len)
        sub_label.append(np.hstack([label1, label2, label3]).squeeze())  # 将所有受试者的标签存入sub_label 15*(3*15,)只用一个session时
        # sub_label.append(np.hstack([label]).squeeze())
    sub_mov = np.array(sub_mov)   # (15, 45, 62, 265*5)
    sub_label = np.array(sub_label) # (15, 45)
    return sub_mov, sub_label

def build_dataset(cross_num, data_name, data_path, labels_path, view_num=5):
    mov_coefs, labels = get_data(data_name, data_path, labels_path) # 受试者个数，每个受试者的视频数，通道数，频带特征数*时间点数
    used_coefs = mov_coefs
    splits = kfold_indices(labels, n_splits=cross_num)
    print(splits)
    if data_name == 'seed' or data_name == 'seed4':
        for train_index, test_index in splits:
            path = '/home/mhyu/yuminghao/code/MVEEG2/{}/processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                        data_name, version, 'Train', cross_num, test_index[0])
            print(path)
            print('Building train and test dataset')
            train_X = dict()
            test_X = dict()
            if data_name == 'seed':
                for v in range(view_num):
                    train_X[v] = used_coefs[train_index,:,:,v,:].reshape(-1, 62, 265)
                    test_X[v] = used_coefs[test_index,:,:,v,:].reshape(-1, 62, 265)
            elif data_name == 'seed4':
                for v in range(view_num):
                    train_X[v] = used_coefs[train_index,:,:,v,:].reshape(-1, 62, 64)
                    test_X[v] = used_coefs[test_index,:,:,v,:].reshape(-1, 62, 64)
            else:
                raise ValueError('Data name not recognized.')
            Y = labels[train_index].reshape(-1)
            testY = labels[test_index].reshape(-1) 
            #get labels
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)#
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)

            train_dataset = EmotionDataset('Train', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), cross_num, test_index[0], train_X, Y)
            test_dataset = EmotionDataset('Test', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), cross_num, test_index[0], test_X, testY)
    elif data_name == 'faced':
        for sub_i in range(cross_num):
            path = '/home/mhyu/yuminghao/code/MVEEG2/{}/processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                    data_name, version, 'Train', cross_num, sub_i)
            print(path)
            train_idx, test_idx = splits[sub_i]
            test_index = test_idx
            train_index = train_idx
            
            print('Building train and test dataset')
            train_X = dict()
            test_X = dict()
            for v in range(view_num):
                train_X[v] = used_coefs[train_index,:,:,v,:].reshape(-1, 32, 30)
                test_X[v] = used_coefs[test_index,:,:,v,:].reshape(-1, 32, 30)
            #get train & test
            Y = labels[train_index].reshape(-1)
            testY = labels[test_index].reshape(-1) 
            #get labels
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)#
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)

            train_dataset = EmotionDataset('Train', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), cross_num, sub_i, train_X, Y)
            test_dataset = EmotionDataset('Test', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), cross_num, sub_i, test_X, testY)
            print('Dataset is built.')
            
def get_dataset(subjects, sub_i, data_name):
    path = '/home/mhyu/yuminghao/code/MVEEG2/{}/processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
            data_name, version, 'Train', subjects, sub_i)
    print(path)
    if not os.path.exists(path): 
        _data_path = '/home/mhyu/yuminghao/code/datasets/SEED/ExtractedFeatures/'
        # _data_path = '/home/mhyu/yuminghao/code/datasets/FACED/EEG_Features/DE/'
        _labels_path = '/home/mhyu/yuminghao/code/datasets/SEED/ExtractedFeatures/label.mat'
        # _labels_path = "/home/mhyu/yuminghao/code/datasets/FACED/labels9.mat"
        # get_data(data_name='seed', data_path=_data_path, labels_path=_labels_path)
        build_dataset(cross_num=15, data_name='seed', data_path=_data_path, labels_path=_labels_path, view_num=5)
        # raise IOError('Train dataset is not exist!')

    train_dataset = EmotionDataset('Train', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), subjects, sub_i)
    test_dataset = EmotionDataset('Test', '/home/mhyu/yuminghao/code/MVEEG2/{}/'.format(data_name), subjects, sub_i)

    return train_dataset, test_dataset

if __name__ == '__main__':
    # _data_path = '/home/mhyu/yuminghao/code/datasets/SEED/ExtractedFeatures/'
    _data_path = '/home/mhyu/yuminghao/code/datasets/SEEDIV/eeg_feature_smooth/'
    # _data_path = '/home/mhyu/yuminghao/code/datasets/FACED/EEG_Features/DE/'
    _labels_path = '/home/mhyu/yuminghao/code/datasets/SEED/ExtractedFeatures/label.mat'
    _labels_path = '/home/mhyu/yuminghao/code/datasets/SEEDIV/eeg_feature_smooth/'
    # _labels_path = "/home/mhyu/yuminghao/code/datasets/FACED/labels9.mat"
    # get_data(data_name='seed', data_path=_data_path, labels_path=_labels_path)
    build_dataset(cross_num=15, data_name='seed4', data_path=_data_path, labels_path=_labels_path, view_num=5)
    
