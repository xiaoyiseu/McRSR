import argparse
import datetime
import json
import os
import random
import time
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
import spacy
#allow_tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchvision as tv
import utils
from dataset import create_dataset, create_sampler, create_loader
from models.model_person_search import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import warnings
warnings.filterwarnings("ignore")


# CUDA_VISIBLE_DEVICES = int(os.environ['LOCAL_RANK'])
# torch.distributed.barrier(device_ids=int(os.environ["LOCAL_RANK"]))

import torch
from thop import profile
import utils

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pitm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_prd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mrtd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mae', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_attribute', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('flops', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('params', utils.SmoothedValue(window_size=1, fmt='{value:.2f}M'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    computed_flops = False  # 只计算一次 FLOPs

    for i, (image1, image2, text1, text2, idx, replace) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        replace = replace.to(device, non_blocking=True)

        text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
        text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)

        attribute_masks = []
        for text_ids in text_input2.input_ids:
            attribute_mask = get_attribute_mask(text_ids, tokenizer)
            attribute_masks.append(attribute_mask)

        attribute_masks = torch.stack(attribute_masks)
        text_input2.attribute_masks = attribute_masks.to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1.0, i / len(data_loader))

        # =================== 计算 FLOPs（仅在第一个 batch 计算） ====================
        if not computed_flops:
            inputs = (image1, image2, text_input1, text_input2, alpha, idx, replace)
            total_flops, total_params = profile(model, inputs=inputs, verbose=False)
            total_params /= 1e6  # 转换为百万（M）参数
            total_flops /= 1e9  # 计算 **每个样本的 FLOPs**
            total_flops /= image1.shape[0]  # 计算 **每个样本的 FLOPs**
            computed_flops = True  # 只计算一次
        else:
            total_flops, total_params = 0, 0  # 避免额外计算


        # =================== 前向传播 ====================
        loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_mae, loss_attribute = model(
            image1, image2, text_input1, text_input2, alpha=alpha, idx=idx, replace=replace)

        loss = sum(config['weights'][j] * los for j, los in enumerate(
            (loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_mae, loss_attribute)))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =================== 记录损失 & FLOPs ====================
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_pitm=loss_pitm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_prd=loss_prd.item())
        metric_logger.update(loss_mrtd=loss_mrtd.item())
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(loss_attribute=loss_attribute.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if computed_flops:
            metric_logger.update(flops=total_flops)
            metric_logger.update(params=total_params)

        # Warm-up 调度
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

                          
# def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
#     # train
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_pitm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_prd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_mrtd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_mae', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     metric_logger.add_meter('loss_attribute', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     header = 'Train Epoch: [{}]'.format(epoch)
#     print_freq = 50
#     step_size = 100
#     warmup_iterations = warmup_steps * step_size
#     for i, (image1, image2, text1, text2, idx, replace) in enumerate(
#             metric_logger.log_every(data_loader, print_freq, header)):
#         image1 = image1.to(device, non_blocking=True)
#         image2 = image2.to(device, non_blocking=True)
#         idx = idx.to(device, non_blocking=True)
#         replace = replace.to(device, non_blocking=True)

#         text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
#         text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)

#         attribute_masks=[]
#         for text_ids in text_input2.input_ids:
#             attribute_mask = get_attribute_mask(text_ids, tokenizer)
#             attribute_masks.append(attribute_mask)

#         attribute_masks=torch.stack(attribute_masks)
#         text_input2.attribute_masks=attribute_masks.to(device)


#         if epoch > 0 or not config['warm_up']:
#             alpha = config['alpha']
#         else:
#             alpha = config['alpha'] * min(1.0, i / len(data_loader))
#         loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_mae, loss_attribute = model(image1, image2, text_input1, text_input2,
#                                                                   alpha=alpha, idx=idx, replace=replace)
#         loss = 0.
#         for j, los in enumerate((loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_mae, loss_attribute)):
#             loss += config['weights'][j] * los
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         metric_logger.update(loss_cl=loss_cl.item())
#         metric_logger.update(loss_pitm=loss_pitm.item())
#         metric_logger.update(loss_mlm=loss_mlm.item())
#         metric_logger.update(loss_prd=loss_prd.item())
#         metric_logger.update(loss_mrtd=loss_mrtd.item())
#         metric_logger.update(loss_mae=loss_mae.item())
#         metric_logger.update(loss_attribute=loss_attribute.item())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
#             scheduler.step(i // step_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())
#     return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
   

nlp = spacy.load('en_core_web_sm')

def get_attribute_mask(text_ids, tokenizer):
    size = text_ids.size(0)
    text = " ".join([tokenizer.decode([token_id]) for token_id in text_ids[1:]])

    doc = nlp(text)
    attribute_mask = torch.zeros(size)
    text_words=text.split()

    idx_words=0
    char_count=0

    count=1
    for chunk in doc.noun_chunks:
        pos=[]
        adj_founded=False # just to be sure
        for tok in chunk:
            if tok.pos_ == "NOUN":
                # noun = tok.text
                #manage she's to be split in she is
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words+1) # +1 to account for [CLS]
            if tok.pos_ == "ADJ":
                adj_founded=True
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words+1) # +1 to account for [CLS]
                
        if len(pos)>1 and adj_founded:
            attribute_mask[pos]=count
            count+=1
    return attribute_mask

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # evaluate
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    # extract text features
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 32
    
    text_inputs_ids = [] # new version
    text_atts_new_version = []
    text_feats = []
    text_embeds = []
    text_atts = []
    

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation = True,  max_length=config["max_words"], return_tensors="pt").to(device)

        text_inputs_ids.append(text_input.input_ids) # new version
        text_atts_new_version.append(text_input.attention_mask) 

        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(torch.cat([text_input.attention_mask], dim=1))

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    text_inputs_ids = torch.cat(text_inputs_ids, dim=0) # new version
    text_atts_new_version = torch.cat(text_atts_new_version, dim=0) # new version

    # extract image features
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # compute the feature similarity score for all image-text pairs
    sims_matrix = text_embeds @ image_embeds.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx.cpu()]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.text_encoder.bert(text_inputs_ids[start + i].repeat(config['k_test'], 1),
                                         attention_mask=text_atts_new_version[start + i].repeat(config['k_test'], 1),
                                         encoder_hidden_states=encoder_output.to(device),
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='multi_modal'
                                         )
    
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1] # 128

        score_matrix_t2i[start + i, topk_idx] = score
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item(),
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       }
    
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    print(args)
    print(config)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    # Dataset
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('ps', config)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],#[0,0,0],#
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    print('Note:',args.text_encoder)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)


    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_log = ''

    # Model
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)
    # Optimizer and learning rate scheduler
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']
        else:
            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)
        if epoch >= config['eval_epoch'] or args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_t2i, test_dataset.img2person, test_dataset.txt2person, args.eval_mAP)
                print('Test:', test_result, '\n')
                if args.evaluate:
                    log_stats = {'epoch': epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()}
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    log_stats = {'epoch': epoch,
                                 **{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'test_{k}': v for k, v in test_result.items()},
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_epoch%02d.pth' % epoch))
                    if test_result['r1'] > best:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = test_result['r1']
                        best_epoch = epoch
                        best_log = log_stats
        if args.evaluate:
            break
        dist.barrier()
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch: {best_epoch} / {max_epoch}\n")
            f.write(f"{best_log}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/PS_cuhk_pedes.yaml')
    parser.add_argument('--output_dir', default='output/cuhk-pedes')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_mAP', action='store_true', help='whether to evaluate mAP')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
