import argparse


def get_args():
    parser = argparse.ArgumentParser(description="McRSA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default=r"./logs")
    parser.add_argument("--log_period", default=200)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") 
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') 
    parser.add_argument("--img_aug", default=False, action='store_true')
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    
    parser.add_argument("--loss_names", default='ppe+fsa+mcu',
                        help="which loss to use ['ppe', 'fsa', 'mcu']")
    
    ######################## img-txt settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=30)#60
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="RSTPReid", 
                        help="[CUHK-PEDES, ICFG-PEDES, RSTPReid, Flickr_30k, MSCOCO]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default= r"F:\DataSet\CrossModalRetrieval\Image_Text")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args(args = [])

    return args
