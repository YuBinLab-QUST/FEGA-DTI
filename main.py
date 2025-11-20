from models import BINDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser(description="FEGA-DTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='Human')
parser.add_argument('--split', default='', type=str, metavar='S', help="split task", choices=['random', 'random1', 'random2', 'random3', 'random4'])
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)

    output_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, f'{args.data}/{args.split}')
    mkdir(output_dir)

    print("start...")
    print(f"dataset: {args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}\n")

    ### === 路径设置 === ###
    dataset_root = '/root/FEGA-DTI-main/FEGA-DTI/datasets'
    dataset_path = os.path.join(dataset_root, args.data)
    split_path = os.path.join(dataset_path, str(args.split)) if args.split else dataset_path

    train_path = os.path.join(split_path, 'train.csv')
    val_path = os.path.join(split_path, "val.csv")
    test_path = os.path.join(split_path, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    ### === 加载缓存的 .pt 特征文件 === ###
    drug_feature_path = os.path.join(dataset_root, f'{args.data}_chemberta.pt')
    prot_feature_path = os.path.join(dataset_root, f'{args.data}_protbert.pt')

    print(f"Loading cached drug features from: {drug_feature_path}")
    print(f"Loading cached protein features from: {prot_feature_path}")

    drug_features = torch.load(drug_feature_path)
    prot_features = torch.load(prot_feature_path)

    ### === 构建 Dataset 实例，传入特征 === ###
    train_dataset = DTIDataset(df_train.index.values, df_train, drug_features, prot_features)
    val_dataset = DTIDataset(df_val.index.values, df_val, drug_features, prot_features)
    test_dataset = DTIDataset(df_test.index.values, df_test, drug_features, prot_features)

    print(f'train_dataset: {len(train_dataset)}')

    params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE,
        'shuffle': True,
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': True,
        'collate_fn': graph_collate_func
    }

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = BINDTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, args.data, args.split, **cfg)
    result = trainer.train()

    with open(os.path.join(output_dir, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(output_dir, "config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {output_dir}")
    print('\nend...')

    return result

if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s")
