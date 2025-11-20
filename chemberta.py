import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 初始化 ChemBERTa 模型
model_name = "/root/BINDTI-main/BINDTI/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval().cuda()  # 使用 GPU（如果可用）

@torch.no_grad()
def get_chemberta_embedding(smiles: str):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 表征
    return embedding.squeeze(0).cpu()

def process_dataset(dataset_path, dataset_name, save_dir):
    all_smiles = set()

    # 收集 train / val / test 中所有唯一 SMILES
    for split in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(os.path.join(dataset_path, split))
        if 'SMILES' not in df.columns:
            raise ValueError(f"列 'SMILES' 在 {split} 文件中未找到")
        all_smiles.update(df['SMILES'].astype(str).str.strip())

    # 提取 ChemBERTa 表征
    embedding_dict = {}
    for smiles in tqdm(all_smiles, desc=f"Processing {dataset_name}", ncols=120):
        try:
            emb = get_chemberta_embedding(smiles)
            embedding_dict[smiles] = emb
        except Exception as e:
            print(f"❌ Failed to process SMILES {smiles}: {e}")

    # 保存为 .pt 文件
    save_path = os.path.join(save_dir, f"{dataset_name}_chemberta.pt")
    torch.save(embedding_dict, save_path)
    print(f"✅ Saved embeddings to {save_path}")

# 数据集路径
datasets = {
    #"bindingDB": "/root/BINDTI-main/BINDTI/datasets/bindingDB/random",
    #"biosnap": "/root/BINDTI-main/BINDTI/datasets/biosnap",
    #"Celegans": "/root/BINDTI-main/BINDTI/datasets/Celegans",
    #"Davis": "/root/BINDTI-main/BINDTI/datasets/Davis"
    #"human":"/root/BINDTI-main/BINDTI/datasets/human/random3",
     #"cold_drug":"/root/BINDTI-main/BINDTI/datasets/cold_drug"
    "Hum":"/root/BINDTI-main/BINDTI/datasets/Hum"
}

# 特征保存路径
save_root = "/root/BINDTI-main/BINDTI/datasets"

# 依次处理每个数据集
for name, path in datasets.items():
    process_dataset(path, name, save_root)
