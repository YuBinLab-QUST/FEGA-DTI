import os
import random
import numpy as np
import torch
import dgl
import logging
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.nn as nn

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def graph_collate_func(x):
    feature_vectors,feature,coor,d, v_p, protein_mask,y = zip(*x)
    #将feature、coor转换为张量
    feature = [torch.tensor(f, dtype=torch.float32) for f in feature if f.size != 0]
    coor = [torch.tensor(c, dtype=torch.float32) for c in coor if c.size != 0]
    # 假设你希望将所有特征和坐标堆叠在一起
    feature = torch.cat(feature,dim=0)  # 形状为 (批大小, 特征维度)
    coor = torch.cat(coor,dim=0)  # 形状为 (批大小, 坐标维度, 3)
    feature = feature.unsqueeze(0)
    coor = coor.unsqueeze(0)
    feature = feature.mean(dim=2, keepdim=True)#通道降维
    # 使用 AdaptiveAvgPool2d 将空间维度从 2216 降到 64
    adaptive_pool = torch.nn.AdaptiveAvgPool2d((64, 1))
    feature = adaptive_pool(feature)  # 现在 feature 的形状为 (1, 64, 1)
    #  将通道维度从 1 扩展到 128
    feature = feature.repeat(1, 1, 128)  # 现在 feature 的形状为 (1, 64, 128)
    coor = coor.unsqueeze(1)
    coor = F.interpolate(coor, size=(64, 3), mode='nearest')  # 空间维度下采样, 得到 (1, 64, 3)
    coor = coor.squeeze(1)
    d = dgl.batch(d)
    v_p = torch.stack(v_p, dim=0)
    protein_mask = torch.tensor(np.array(protein_mask))
    y = torch.tensor(np.array(y))
    feature_vectors = torch.cat(feature_vectors, dim=0)
    feature_vectors = feature_vectors.unsqueeze(2)  # 变为 (32, 600, 1)
    feature_vectors = feature_vectors.repeat(1, 1, 128)  # 变为 (32, 600, 128)
    # 使用自适应平均池化
    pool = nn.AdaptiveAvgPool2d((354, 128))
    pooled_feature_vectors = pool(feature_vectors.permute(0, 2, 1).unsqueeze(3))
    # 先调整维度顺序，将中间维度（354）移到前面，方便后续操作
    reshaped_feature_vectors = pooled_feature_vectors.permute(0, 2, 1, 3)
    feature_vectors = reshaped_feature_vectors[:, :, :, 0]
    return feature_vectors,feature,coor,d, v_p, protein_mask,y
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
