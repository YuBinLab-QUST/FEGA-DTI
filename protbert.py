import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# 初始化 ProtBERT 模型
path = "/root/BINDTI-main/BINDTI/prot_bert"
prot_tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
prot_encoder = BertModel.from_pretrained(path)
prot_encoder.eval().cuda()

# 定义映射层（从 1024 -> 128）
linear_layer = nn.Linear(1024, 128)
linear_layer.eval().cuda()

# 如果你已经训练好并保存过映射层，可以用这行加载：
# linear_layer.load_state_dict(torch.load("protein_linear_layer.pt"))

@torch.no_grad()
def get_prot_embedding(sequence: str) -> torch.Tensor:
    """对单条蛋白质序列编码 + 线性映射"""
    sequence = ' '.join(list(sequence.strip()))
    inputs = prot_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1200)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = prot_encoder(**inputs)
    pooled = outputs.last_hidden_state.mean(dim=1)      # shape: (1, 1024)
    mapped = linear_layer(pooled)                        # shape: (1, 128)
    return mapped.squeeze(0).cpu().to(torch.float32)     # shape: (128,)

def process_protein_dataset(dataset_path, dataset_name, save_dir):
    all_sequences = set()

    for split in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(os.path.join(dataset_path, split))
        if 'Protein' not in df.columns:
            raise ValueError(f"列 'Protein' 在 {split} 文件中未找到")
        all_sequences.update(df['Protein'].astype(str).str.strip())

    embedding_dict = {}
    for seq in tqdm(all_sequences, desc=f"Processing proteins: {dataset_name}", ncols=120):
        try:
            emb = get_prot_embedding(seq)
            embedding_dict[seq] = emb
        except Exception as e:
            print(f"❌ Failed to process protein {seq[:10]}...: {e}")

    save_path = os.path.join(save_dir, f"{dataset_name}_protbert.pt")
    torch.save(embedding_dict, save_path)
    print(f"✅ Saved protein embeddings to {save_path}")

# 四个数据集路径
datasets = {
    #"bindingDB": "/root/BINDTI-main/BINDTI/datasets/bindingDB/random",
    #"biosnap": "/root/BINDTI-main/BINDTI/datasets/biosnap",
    #"Celegans": "/root/BINDTI-main/BINDTI/datasets/Celegans",
    #"Davis": "/root/BINDTI-main/BINDTI/datasets/Davis",
    #"human":"/root/BINDTI-main/BINDTI/datasets/human"
    #"cold_drug":"/root/BINDTI-main/BINDTI/datasets/cold_drug"
    "Hum":"/root/BINDTI-main/BINDTI/datasets/Hum"
}

save_root = "/root/BINDTI-main/BINDTI/datasets"

# 执行处理
for name, path in datasets.items():
    process_protein_dataset(path, name, save_root)
