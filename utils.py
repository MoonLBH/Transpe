from sklearn import metrics
import torch
import torch.utils
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
from keras.preprocessing import text,sequence
from sklearn import metrics
from scipy import stats
import os

def position_encoding(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

def one_hot_data(X):
    vectorizer = text.Tokenizer( lower=False, split=" ", num_words=None, char_level=True )
    vectorizer.fit_on_texts( X )
    # construct a new vocabulary
    alphabet = "AGCT"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    word_index = {k:v for k,v in char_dict.items()}
    word_index["PAD"] = 0
    vectorizer.word_index = word_index.copy()
    index_word = {v:k for k,v in word_index.items()}
    X = vectorizer.texts_to_sequences(X)
    X = [[w for w in x] for x in X]
    X = sequence.pad_sequences(X)
    return X

def data_prepare(train = True, valid=False, test = False):

    if train:
        token_train = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/train/tr_rna_sequence_tensor.pt')
        y_train = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/train/tr_target_tensor.pt')
        seg_train = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/train/tr_seg.pt')
        num = len(token_train)
        x_train = torch.Tensor(num,2,47)
        for i in range(num):
            x_train[i] = torch.stack((token_train[i],seg_train[i]),dim=0)
        train_data = Data.TensorDataset(x_train,y_train)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
        return train_loader

    if valid:
        token_valid = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/valid/va_rna_sequence_tensor.pt')
        y_valid = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/valid/va_target_tensor.pt')
        seg_valid = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/valid/va_seg.pt')
        num = len(token_valid)
        x_valid = torch.Tensor(num,2,47)
        for i in range(num):
            x_valid[i] = torch.stack((token_valid[i],seg_valid[i]),dim=0)
        valid_data = Data.TensorDataset(x_valid,y_valid)
        valid_loader = Data.DataLoader(dataset=valid_data, batch_size=64, shuffle=True, num_workers=2)
        return valid_loader

    if test:
        token_test = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/test/te_rna_sequence_tensor.pt')
        y_test = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/test/te_target_tensor.pt')
        seg_test = torch.load('/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/lib1/test/te_seg.pt')
        num = len(token_test)
        x_test = torch.Tensor(num,2,47)
        for i in range(num):
            x_test[i] = torch.stack((token_test[i],seg_test[i]),dim=0)
        test_data = Data.TensorDataset(x_test,y_test)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=num, shuffle=False, num_workers=2)
        return test_loader

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    pearson_r,_ = stats.pearsonr(y_true, y_pred)
    spearman_r,_ = stats.spearmanr(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson_r,
        'Spearman': spearman_r
    }

def get_test_dataset(cells):
    '''
    cell : ['Endo', 'HCT116', 'MDA']
    '''
    test_loader_dict = {}
    for cell in cells:
        if cell == 'Endo':
            path = '/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/Endo'
            mode = ['Endo-BR1-TR1','Endo-BR1-TR2','Endo-BR2-TR1','Endo-BR2-TR2','Endo-BR2-TR3','Endo-BR3']
        else:
            path = '/data/liuxiuqin/libohao/Bohao/pe/Data/DeepPE/hct_AN_mda'
            mode = ['HCT-BR1-TR1','HCT-BR1-TR2','HCT-BR2-TR1','HCT-BR2-TR2','MDA-BR1-TR1','MDA-BR1-TR2','MDA-BR2-TR1','MDA-BR2-TR2']
        
        for m in mode:
            data_path_x = os.path.join(path,'{}_X.pt'.format(m))
            data_path_y = os.path.join(path,'{}_Y.pt'.format(m))
            data_path_seg = os.path.join(path,'{}_seg.pt'.format(m))

            token_test = torch.load(data_path_x)
            y_test = torch.load(data_path_y)
            seg_test = torch.load(data_path_seg)
            num = len(token_test)
            x_test = torch.Tensor(num,2,47)
            for i in range(num):
                x_test[i] = torch.stack((token_test[i],seg_test[i]),dim=0)
            test_data = Data.TensorDataset(x_test,y_test)
            test_loader = Data.DataLoader(dataset=test_data, batch_size=num, shuffle=False, num_workers=2)
            test_loader_dict[m] = test_loader
            
    return test_loader_dict

def get_data(save_dir, train = True, valid=False, test = False):

    if train:
        token_train = torch.load(os.path.join(save_dir,'tr_rna_sequence_tensor.pt'))
        y_train = torch.load(os.path.join(save_dir,'tr_target_tensor.pt'))
        seg_train = torch.load(os.path.join(save_dir,'tr_seg.pt'))
        num = len(token_train)
        x_train = torch.Tensor(num,2,47)
        for i in range(num):
            x_train[i] = torch.stack((token_train[i],seg_train[i]),dim=0)
        train_data = Data.TensorDataset(x_train,y_train)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
        return train_loader

    if valid:
        token_valid = torch.load(os.path.join(save_dir,'va_rna_sequence_tensor.pt'))
        y_valid = torch.load(os.path.join(save_dir,'va_target_tensor.pt'))
        seg_valid = torch.load(os.path.join(save_dir,'va_seg.pt'))
        num = len(token_valid)
        x_valid = torch.Tensor(num,2,47)
        for i in range(num):
            x_valid[i] = torch.stack((token_valid[i],seg_valid[i]),dim=0)
        valid_data = Data.TensorDataset(x_valid,y_valid)
        valid_loader = Data.DataLoader(dataset=valid_data, batch_size=64, shuffle=True, num_workers=2)
        return valid_loader

    if test:
        token_test = torch.load(os.path.join(save_dir,'te_rna_sequence_tensor.pt'))
        y_test = torch.load(os.path.join(save_dir,'te_target_tensor.pt'))
        seg_test = torch.load(os.path.join(save_dir,'te_seg.pt'))
        num = len(token_test)
        x_test = torch.Tensor(num,2,47)
        for i in range(num):
            x_test[i] = torch.stack((token_test[i],seg_test[i]),dim=0)
        test_data = Data.TensorDataset(x_test,y_test)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=num, shuffle=False, num_workers=2)
        return test_loader