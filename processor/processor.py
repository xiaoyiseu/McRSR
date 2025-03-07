import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from thop import profile

def dynamic_scaling(loss_dict, epsilon=1e-8):
    mean_magnitude = sum(v.mean().item() for v in loss_dict.values()) / len(loss_dict)
    return {k: v * (mean_magnitude / (v.mean().item() + epsilon)) for k, v in loss_dict.items()}

def TrainModule(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    device, logger = "cuda", logging.getLogger("McRSA.train")
    arguments = {}
    arguments["num_epoch"] = args.num_epoch
    arguments["iteration"] = 0

    logger.info('start training')
    tb_writer, best_top1, patience, patience_counter = SummaryWriter(log_dir=args.output_dir), 0.0, 5, 0
    stages, current_stage = args.loss_names.split('+'), 1
    stage_converged = {f"{stage}_loss": False for stage in stages}

    for epoch in range(start_epoch, args.num_epoch + 1):
        start_time, meters = time.time(), {k: AverageMeter() for k in ["loss"] + [f"{s}_loss" for s in stages]}
        model.train()
        current_loss_name = f"{stages[current_stage - 1]}_loss"

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if epoch == start_epoch and n_iter == 0:
                total_flops, params = profile(model, inputs=(batch,))
                batch_size = batch["images"].shape[0]  
                flops = total_flops / batch_size  
                logger.info(f"FLOPs: {flops / 1e9:.3f} GFLOPs, Params: {params / 1e6:.3f} M")

            ret = model(batch)
            total_loss = sum(v for k, v in ret.items() if "loss" in k)
            meters["loss"].update(total_loss.item(), batch["images"].shape[0])

            for k in stages:
                if f"{k}_loss" in ret:
                    meters[f"{k}_loss"].update(ret[f"{k}_loss"].item(), batch["images"].shape[0])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % args.log_period == 0:
                logger.info(
                    f"Epoch[{epoch}/{args.num_epoch}] Iter[{n_iter + 1}/{len(train_loader)}], " +
                    ", ".join(f"{k}: {v.avg:.4f}" for k, v in meters.items() if v.avg > 0) +
                    f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                )

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            logger.info(
                f"Epoch {epoch} done. Time per batch: {(time.time() - start_time) / len(train_loader):.3f}s"
            )

        if epoch % args.eval_period == 0 and get_rank() == 0:
            # if model.module.eval():
            top1 = evaluator.eval(model.eval() )
            torch.cuda.empty_cache()
            if best_top1 < top1:
                best_top1, patience_counter = top1, 0
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stage_converged[current_loss_name] = True
                logger.info(f"{current_loss_name} converged at epoch {epoch}")
                patience_counter = 0
                if current_stage < len(stages):
                    current_stage += 1
                else:
                    logger.info("All stages converged. Stopping training.")
                    break
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

def InferenceModule(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("McRSA.test")
    logger.info("Enter inferencing")

    start_time = time.time()

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
    total_time = time.time() - start_time
    total_samples = len(test_txt_loader.dataset) + len(test_img_loader.dataset)
    throughput = total_samples / total_time  # 吞吐量（samples/second）
    latency = total_time / total_samples  # 延迟（seconds/sample）
    logger.info(f"\n[计算效率统计]")
    logger.info(f"总耗时: {total_time:.2f} second")
    logger.info(f"吞吐量: {throughput:.2f} samples/second")
    logger.info(f"平均延迟: {latency:.4f} seconds/sample")
    return top1
