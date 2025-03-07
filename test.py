from prettytable import PrettyTable
import os
import torch, os
import numpy as np
import time
import os.path as op

from datasets import build_dataloader 
from processor.processor import InferenceModule
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from thop import profile
device = 'cuda' if torch.cuda.is_available()  else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="McRSA Test")
    dataset_dir = r''# dataset path
    root_dir = r'' # pretrained mpdel path

    pretrain_set = 'RSTPReid' # [CUHK-PEDES, ICFG-PEDES, RSTPReid, Flickr_30k, MSCOCO]
    premodel = os.path.join(root_dir, pretrain_set)

    weight_name = 'McRSA'
    Pretrain_Lib = {pretrain_set: weight_name}  
    weigt_fileName = Pretrain_Lib[pretrain_set]
    test_path = os.path.join(premodel, weigt_fileName)

    path_test = os.path.join(premodel, test_path + '/' + 'configs.yaml')
    parser.add_argument("--config_file", default=path_test)
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.dataset_name = pretrain_set 
    args.test_batch_size = 64

    args.root_dir = dataset_dir
    args.output_dir = os.path.join(root_dir, test_path)
    args.training = False
    logger = setup_logger('McRSA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    InferenceModule(model, test_img_loader, test_txt_loader)
