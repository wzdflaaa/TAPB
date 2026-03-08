import argparse
import os
import pickle
from pickle import UnpicklingError
import warnings
from time import time
import pandas as pd
import torch
from omegaconf import OmegaConf
from dataloader.dataloader import DTIDataset, get_dataLoader
from transformers import AutoTokenizer
from models.tapb import TAPB
from trainer import Trainer
from utils.utils import set_seed, mkdir, load_config_file
from preparation import generate_esm2_feature

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="TAPB for DTI prediction")
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster', 'augmented'])
args = parser.parse_args()

TRAIN_CONFIG_PATH = 'configs/train_config.yaml'
MODEL_CONFIG_PATH = 'configs/model_config.yaml'
print(f"Running on: {device}", end="\n\n")

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    train_config = load_config_file(TRAIN_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)
    set_seed(seed=config.TRAIN.SEED)
    output_path = f"./results/{args.data}/{args.split}/{config.TRAIN.OUTPUT_DIR}{config.TRAIN.SEED}"
    mkdir(output_path)
    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))
    mol_path = 'models/drug/molformer'

    if args.split == 'cluster':
        train_path = os.path.join(dataFolder, 'source_train_with_id.csv')
        val_path = os.path.join(dataFolder, "target_test_with_id.csv")
        test_path = os.path.join(dataFolder, "target_test_with_id.csv")
    else:
        train_path = os.path.join(dataFolder, 'train_with_id.csv')
        val_path = os.path.join(dataFolder, "val_with_id.csv")
        test_path = os.path.join(dataFolder, "test_with_id.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    protein_path = os.path.join(dataFolder, config.TRAIN.PR_PATH)

    """
    旧版本可能将蛋白特征存储为列表，按 pr_id排序。
            -当读取内存--esm-2特征失败时则调用函数重新生成特征文件，并且在读取后检查数据格式
    检查加载的数据是否为列表类型：
    如果是，则通过合并训练集、验证集和测试集的 DataFrame（df_train、df_val、df_test）
    提取唯一的 pr_id，并将列表转换为字典，键为 pr_id，值为对应特征。
    这种处理确保了旧版本特征文件的向后兼容性，因为旧版本可能将蛋白质特征存储为与唯一 pr_id 顺序对应的列表。"""

    def load_or_build_protein_features():
        if not os.path.isfile(protein_path):
            generate_esm2_feature(config, args.data, args.split)

        try:
            with open(protein_path, 'rb') as protein_f:
                protein_features = pickle.load(protein_f)
        except (UnpicklingError, EOFError, OSError, ValueError):
            print(f"Warning: corrupted protein feature file detected at {protein_path}. Regenerating...")
            if os.path.exists(protein_path):
                os.remove(protein_path)
            generate_esm2_feature(config, args.data, args.split)
            with open(protein_path, 'rb') as protein_f:
                protein_features = pickle.load(protein_f)

        # Backward compatibility: old feature files may store a list ordered by unique pr_id.
        if isinstance(protein_features, list):
            all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
            unique_pr_ids = all_df['pr_id'].unique()
            protein_features = {pid: feature for pid, feature in zip(unique_pr_ids, protein_features)}

        if not isinstance(protein_features, dict):
            raise TypeError(f"Unsupported protein feature format: {type(protein_features)}")

        return protein_features

# add:Baseline TAPB (no confounder/CAM/backdoor path)
    model = TAPB(model_configs=model_configs).to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    pr_f = load_or_build_protein_features()

    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    val_dataset = DTIDataset(df_val.index.values, df_val, pr_f)
    test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)

    drug_tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
    
    bz = config.TRAIN.BATCH_SIZE
    
    MLM = config.TRAIN.MLM
    
    #去除了mutation和random_deletion，专注于masking--- CAM和数据增强(do(T))的相关参数
    train_dataloader = get_dataLoader(bz, train_dataset, drug_tokenizer, shuffle=True, MLM=MLM,
                                      mask_rate=config.TRAIN.MASK_PROBABILITY,
                                      target_random_deletion_ratio=0,
                                      mutation_rate=0)
    val_dataloader = get_dataLoader(bz, val_dataset, drug_tokenizer)
    test_dataloader = get_dataLoader(bz, test_dataset, drug_tokenizer)


    trainer = Trainer(model, opt, device, train_dataloader, val_dataloader, test_dataloader, output_path, config)
    result, best_epoch = trainer.train()


    with open(os.path.join(output_path, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print()

    return result, best_epoch


if __name__ == '__main__':
    s = time()
    result, best_epoch = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
