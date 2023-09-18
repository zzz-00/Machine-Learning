import gc
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # 进度条


# 固定随机数生成器的种子
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 预处理数据
def load_feat(path):
    feat = torch.load(path)  # 加载数据，用于打开feat的pt文件
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)  # repeat()函数可以对张量重复扩张，两个参数时第一个为行的倍数，第二个为列的个数
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


# 拼接2k+1个语音帧
def concat_feat(x, concat_n):  # 传入feat和要拼接的帧的数量
    assert concat_n % 2 == 1  # concat_n must be odd
    if concat_n < 2:  # 不拼接
        return x
    seq_len, feature_dim = x.size(0), x.size(1)  # 行数表示有多少条语素，列数表示有多少特征
    x = x.repeat(1, concat_n)  # 行数不变，列数乘concat_n次
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim，改变tensor格式为第0维：3，第1维语音条数，第2维特征个数
    mid = (concat_n // 2)  # 向下取整
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)  # 取后面一个frame
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)  # 取前面一个frame

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8):
    # 类别数，不能更改
    class_num = 41  # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':  # 两种模式
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}  # 定义标签字典
    if mode == 'train':
        # 拼接文件路径./libriphone/train_labels.txt，一行一行地读训练标签，./表示当前目录下的xxx
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')  # 把所有换行和空格去掉
            label_dict[line[0]] = [int(p) for p in line[1:]]  # 生成字典

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.shuffle(usage_list)  # 打乱原列表
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)  # 设置tensor size，共n个frame，每个frame有39个特征
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):  # 进度条
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)  # 当前frame中含有多少语音条
        feat = concat_feat(feat, concat_nframes)  # 拼接操作
        if mode == 'train':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
        print(y.shape)
        return X, y
    else:
        return X


# 定义数据集，获取dataset对象
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)  # 转为LongTensor类型
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]  # 获取该行数据
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# 模型
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),  # 在2D和3D输入数据上应用批量正则化
            nn.Dropout(0.25)  # 缓解过拟合
        )

    def forward(self, x):
        x = self.block(x)
        return x


# 定义分类网络
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],  # 生成一个由hidden_layers个BasicBlock构成的列表；*用来调取列表中内容
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):  # 对父类的forward进行重写
        x = self.fc(x)
        return x


# Hyper-parameters
# data prarameters
# TODO: change the value of "concat_nframes" for medium baseline
concat_nframes = 15  # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75  # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 114514  # random seed
batch_size = 512  # batch size
num_epoch = 40  # the number of training epoch
learning_rate = 1e-4  # learning rate
model_path = './model.ckpt'  # the path where the checkpoint will be saved

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 10  # the number of hidden layers
hidden_dim = 512  # the hidden dim

# 加载数据
same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes,
                                   train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 训练
# 初始化模型, 定义损失函数, 优化器
# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch  # 一个batch分为特征和结果列
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()  # 计算梯度，反向传播
        optimizer.step()  # 更新数据

        _, train_pred = torch.max(outputs, 1)  # 返回每个语音（数据）的预测概率值最大的类别，val_pred表示在outputs中[]中的位置，第一个返回值是概率的具体数值
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    # validation
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # 去除梯度信息
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1)  # 获取概率值最大的类别
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
            val_loss += loss.item()

    print(
        f'[{epoch + 1:03d}/{num_epoch:03d}] Train Acc: {train_acc / len(train_set):3.5f} Loss: {train_loss / len(train_loader):3.5f} | Val Acc: {val_acc / len(val_set):3.5f} loss: {val_loss / len(val_loader):3.5f}')

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc / len(val_set):.5f}')

# 回收不需要的数据
del train_set, val_set
del train_loader, val_loader
gc.collect()

# 测试
# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))  # 加载模型

pred = np.array([], dtype=np.int32)  # 初始化空数组

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# 保存测试结果
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
