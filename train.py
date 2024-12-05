import math
import os
import time
import argparse
import torch
import numpy as np
import torch.distributed as dist
from pathlib import Path, PurePath
from mmcv import Config
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from data.build_dataset import DatasetsBuilder
from modeling.model import EncoderDecoder
from modeling.semi_module import SemiModule
from utils import CheckpointSaver, Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--workdir', default=None, type=str)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.workdir:
        cfg.workdir = args.workdir
    gpu_num = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    logger = Logger(cfg).logger
    logger.info('Config path: {}'.format(args.config))
    tb_writer = SummaryWriter(str(PurePath(cfg.workdir, 'tensorboard')))
    ckpt_saver = CheckpointSaver(cfg.workdir, save_top_k=3)

    data_cfg = cfg.data
    data_cfg.gpu_num = gpu_num
    DataBuilder = DatasetsBuilder(data_cfg)
    train_loader, train_unlabel_loader = DataBuilder.train_dataloader_ddp()
    test_loader = DataBuilder.test_dataloaders(data_cfg.test_subset)

    model = EncoderDecoder(cfg.model).cuda()
    model.to(device)
    
    semi_module = SemiModule(cfg.model, model, device, local_rank)

    base_lr = cfg.optimizer.lr
    lr = base_lr * math.sqrt(gpu_num) * data_cfg.samples_per_gpu / 256. if cfg.optimizer.adjust_lr else base_lr
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    if cfg.scheduler.type == 'OneCycle':
        sched = OneCycleLR(optim, lr, cfg.max_iter, pct_start=cfg.scheduler.warmup_pct, cycle_momentum=False)
    elif cfg.scheduler.type == 'Cosine':
        sched = CosineAnnealingLR(optim, cfg.max_iter, eta_min=0, last_epoch=-1, verbose=False)

    if cfg.model.pretrained and Path(cfg.model.pretrained).exists():
        checkpoint = torch.load(cfg.model.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
        logger.info('Pretrain model loaded: {}'.format(cfg.model.pretrained))
    else:
        logger.info('Pretrain model not found')

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    start_time = time.time()
    epoch = 0
    unlabel_epoch = 0
    trainiter = iter(train_loader)
    train_unlabel_iter = None if train_unlabel_loader is None else iter(train_unlabel_loader)
    
    for step in range(cfg.max_iter):
        optim.zero_grad()

        try:
            _, img, simple_img, text = trainiter.next()
        except StopIteration:
            epoch += 1
            DataBuilder.train_sampler_set_epoch(epoch)

            trainiter = iter(train_loader)
            _, img, simple_img, text = trainiter.next()
        
        img = img.cuda()
        simple_img = simple_img.cuda()
        loss, metrics, tgt_out, proj_logits, proj_rec_feat_cua = model(img, text, simple_img, training=True, with_simple_img=True, unlabel=False)
        
        student_data_dict = {
            'tgt_out': tgt_out,
            'text': text,
            'proj_logits': proj_logits,
            'proj_rec_feat_cua': proj_rec_feat_cua,
        }
        contrast_loss, contrast_metrics = semi_module(None, simple_img, model, step, student_data_dict, update_ema=True)
        
        loss += contrast_loss
        metrics.update(contrast_metrics)
        
        if train_unlabel_iter is not None:
            try:
                _, img, simple_img, text = train_unlabel_iter.next()
            except StopIteration:
                unlabel_epoch += 1
                DataBuilder.train_unlabel_sampler_set_epoch(unlabel_epoch)
                train_unlabel_iter = iter(train_unlabel_loader)
                _, img, simple_img, text = train_unlabel_iter.next()

            img = img.cuda()
            simple_img = simple_img.cuda()
            semi_loss, semi_metrics = semi_module(img, simple_img, model, step, None, update_ema=False)

            loss += semi_loss
            metrics.update(semi_metrics)

        loss.backward()
        optim.step()
        sched.step()

        if ((step + 1) % cfg.write_log_interval == 0) and local_rank == 0:
            time_used = time.time() - start_time
            time_h = time_used // 3600
            time_m = (time_used - time_h * 3600) // 60
            time_s = time_used % 60

            eta = time_used * (cfg.max_iter - step) / (step + 1)
            eta_h = eta // 3600
            eta_m = (eta - eta_h * 3600) // 60
            eta_s = eta % 60
            sub_loss_str = ' | '.join(['{} loss:{:<.4f}'.format(k, v) for k, v in metrics.items()])
            loss_str = 'Iter: {}/{} | {} | lr:{:<.4g} | Time:{:2.0f}:{:2.0f}:{:2.0f} | ETA:{:2.0f}:{:2.0f}:{:2.0f}'.format(
                step + 1, cfg.max_iter, sub_loss_str, optim.param_groups[0]['lr'], time_h, time_m, time_s, eta_h, eta_m, eta_s)
            logger.info(loss_str)
            tb_writer.add_scalar('train/loss', loss.item(), step)
            for loss_key, loss_value in metrics.items():
                tb_writer.add_scalar('train/{}'.format(loss_key), loss_value.item(), step)

        if ((step + 1) % cfg.val_interval == 0) and local_rank == 0:
            model.eval()
            with torch.no_grad():
                num_list = np.zeros(len(test_loader))
                acc_list = np.zeros(len(test_loader))
                ned_list = np.zeros(len(test_loader))
                conf_list = np.zeros(len(test_loader))
                len_list = np.zeros(len(test_loader))
                for idx, (name, test_dataset) in enumerate(test_loader.items()):
                    num_samples = 0
                    correct = 0
                    ned = 0
                    confidence = 0
                    label_length = 0
                    for _, img1, img2, text in test_dataset:
                        img1 = img1.cuda()
                        img2 = img2.cuda()
                        img = torch.cat((img1, img2), dim=0)
                        tmp_num_samples, tmp_correct, tmp_ned, tmp_confidence, tmp_label_length, rec_pred = model(
                            img, rec_labels=text, training=False, with_simple_img=False, unlabel=False, test_with_proj=cfg.model.test_with_proj, test_with_reverse=True)

                        num_samples += tmp_num_samples
                        correct += np.sum(tmp_correct)
                        ned += np.sum(tmp_ned)
                        confidence += np.sum(tmp_confidence)
                        label_length += np.sum(tmp_label_length)

                    accuracy = 100 * correct / num_samples
                    mean_ned = 100 * (1 - ned / num_samples)
                    mean_conf = 100 * confidence / num_samples
                    mean_label_length = label_length / num_samples

                    num_list[idx] = num_samples
                    acc_list[idx] = accuracy
                    ned_list[idx] = mean_ned
                    conf_list[idx] = mean_conf
                    len_list[idx] = mean_label_length
                    eval_str = '{:<14} | samples:{:<6} | accuracy:{:<.2f} | NED:{:<.4f} | confidence:{:<.4f} | label length:{:<.4f} '.format(
                        name, num_samples, accuracy, mean_ned, mean_conf, mean_label_length)
                    logger.info(eval_str)
                    tb_writer.add_scalar('eval/{}'.format(name), accuracy, step+1)

                total_num = int(num_list.sum())
                ave_acc = (acc_list * num_list).sum() / total_num
                ave_ned = (ned_list * num_list).sum() / total_num
                ave_conf = (conf_list * num_list).sum() / total_num
                ave_len = (len_list * num_list).sum() / total_num
                eval_str = '{:<14} | samples:{:<6} | accuracy:{:<.2f} | NED:{:<.4f} | confidence:{:<.4f} | label length:{:<.4f} '.format(
                    'Average', total_num, ave_acc, ave_ned, ave_conf, ave_len)
                logger.info(eval_str)
                tb_writer.add_scalar('eval/average', ave_acc, step+1)
                if ckpt_saver.save(model, optim, step+1, ave_acc, last=False):
                    logger.info('Save iter={}-acc={:.2f}.ckpt'.format(step+1, ave_acc))
            model.train()

    if local_rank == 0:
        _, best_models = ckpt_saver.save(model, optim, last=True)
        save_str = 'Best saved models: ' + ' | '.join([tmp for tmp in best_models.keys()])
        logger.info(save_str)


if __name__ == '__main__':
    main()
