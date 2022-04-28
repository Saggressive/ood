import torch
import os
import sys
os.chdir("../")
sys.path.append("./")
from processdata.pre_process import tokenizer_process,process,get_label_dict
from processdata.subwords_enhance import get_four_fold,get_two
from model.ood_model import BertForSequenceClassification
from model.datatset import synthesis_data,DataCollator
from datasets import load_dataset
from transformers import BertTokenizer,DataCollatorWithPadding
from transformers import AdamW,get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader,WeightedRandomSampler
from tqdm import tqdm
from model.model_evaluate import evaluate_base,get_best_acc
import shutil
import json
from torch.nn.functional import softmax
import tensorboard_logger as tb_logger
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def main_test(config):
    # labels_dict = get_label_dict(config["data_path"], config["know_rate"])
    labels_dict_path = config["save_dir"] + os.sep + 'dict.txt'
    with open(labels_dict_path,"r") as f:
        labels_dict=json.load(f)
    print(labels_dict)
    tokenizer = BertTokenizer.from_pretrained(config["save_dir"])
    model = torch.load(config["model_save_path"])
    for name,par in model.named_parameters():
        print(name)

if __name__=="__main__":
    save_dir = "./best_model_idea_binary"
    config = {
        "pretrained_path": "./bert-base-uncased",
        "data_path": "./oos",
        "know_rate": 0.25,
        "train_script_path": "./dataset/oos_data",
        "neg_script_path": "./dataset/neg_data",
        "epoch": 200,
        "accumulation_steps": 1,
        "batch_size": 96,
        "num_labels": 2,
        "save_dir": save_dir,
        "model_save_path": save_dir + os.sep + "best.pt",
        "is_save": True,
        "config_save_path": save_dir + os.sep + "ood_config.txt",
        "val_result_path": save_dir + os.sep + "result_val.txt",
        "test_result_path": save_dir + os.sep + "result_test.txt",
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "linear_lr": 2e-4,
        "linear_decay": 1e-4,
        "use_balance": False,
        "tb_folder": "./tb_folder_idea_binary",
        "tmp": 0.2,
        "val_tmp": 1,
        "use_neg": True,
        "token_length": 64,
        "alpha": 1.0,
        "beta": 0.0,
        "neg_multiple": 2
    }
    main_test(config)