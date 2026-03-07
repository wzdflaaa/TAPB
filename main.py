import argparse
import os
import pickle
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

    if not os.path.isfile(protein_path):
        generate_esm2_feature(config, args.data, args.split)

# add:Baseline TAPB (no confounder/CAM/backdoor path)
    model = TAPB(model_configs=model_configs).to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    protein_f = open(protein_path, 'rb')
    pr_f = pickle.load(protein_f)
    
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
