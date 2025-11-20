import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem

from functools import lru_cache
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
def one_of_k_encoding_unk(x, allowable_set):
    """
    将输入 x 编码为 one-hot 向量。如果 x 不在允许集合中，则将其编码为最后一个元素（'Unknown'）。
    """
    if x not in allowable_set:
        x = 'Unknown'
    return [x == s for s in allowable_set]

def one_of_k_encoding(x, allowable_set):
    """
    将输入 x 编码为 one-hot 向量。
    """
    return [1 if x == s else 0 for s in allowable_set]
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                                          'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
                                          'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

@lru_cache(maxsize=128)
def get_coors_batch(smiles_list):
    all_features = []
    all_coordinates = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES:{smile}")
            mol = Chem.AddHs(mol)
            num_confs = 1
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
            if len(ids) == 0:
                raise ValueError(f"Embedding failed for SMILES:{smile}")
            if AllChem.UFFOptimizeMolecule(mol, confId=ids[0]) == -1:
                raise ValueError(f"UFF optimization failed for SMILES:{smile}")
            c_size = mol.GetNumAtoms()
            features = []
            coordinates = []
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                feature = atom_features(atom)
                features.append(feature / sum(feature))
                pos = mol.GetConformer().GetAtomPosition(atom_idx)
                coordinates.append(np.array(pos))
            all_features.append(np.array(features))
            all_coordinates.append(np.array(coordinates))
        except Exception as e:
            all_features.append(np.zeros((0,)))
            all_coordinates.append(np.zeros((0, 3)))
    return np.array(all_features), np.array(all_coordinates)

class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df,drug_features, protein_features, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.all_feature_vectors = drug_features  # 直接使用传入的特征字典
        self.all_protein_features = protein_features
        unique_smiles = df['SMILES'].unique()
        self.smiles_graph_cache = {}
        for smi in unique_smiles:
            graph = self.fc(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            self.smiles_graph_cache[smi] = graph
    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        drug_smi = self.df['SMILES'][index]
        feature_vectors = self.all_feature_vectors[drug_smi].unsqueeze(0)  # 获取对应索引的编码结果

        v_d = self.df.iloc[index]['SMILES']
        #获取药物分子原子的三位坐标
        #feature,coor = get_coors(v_d)
        feature, coor = get_coors_batch((v_d,))
        feature = feature[0]  # 取出批次结果中的对应元素（因为只传入了一个SMILES）
        coor = coor[0]
        #v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        v_d = self.smiles_graph_cache[drug_smi]
        v_d = v_d.clone()  # ⚠️ 注意：clone 避免 in-place 操作污染缓存
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        protein_seq = self.df['Protein'][index]
        #获取蛋白质的特征
        v_p = self.df.iloc[index]['Protein']
        pro_len = len(v_p)
        y = self.df.iloc[index]["Y"]
        #处理蛋白质掩码
        protein_max = 1200
        protein_mask = np.zeros(protein_max)

        v_p = self.all_protein_features[protein_seq].unsqueeze(0).repeat(1200, 1)  # 获取对应索引的编码结果并重复


        if pro_len > protein_max:
            protein_mask[:] = 1
        else:
            protein_mask[:pro_len] = 1
        return feature_vectors,feature,coor,v_d, v_p, protein_mask, y

class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches