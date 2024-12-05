import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .model import EncoderDecoder


class SemiModule(nn.Module):
    def __init__(self, cfg, student_model, device, local_rank, ema_alpha=0.999):
        super().__init__()
        self.cfg = cfg
        self.conf_thresh = cfg.conf_thresh
        self.kl_thresh = cfg.kl_thresh
        self.temperature = cfg.temperature
        
        self.teacher_model = EncoderDecoder(cfg).cuda()
        self.teacher_model.to(device)
        self.teacher_model = torch.nn.parallel.DistributedDataParallel(
            self.teacher_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        self.teacher_model.train()
        
        for param_t, param_s in zip(self.teacher_model.parameters(), student_model.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for buffer_t, buffer_s in zip(self.teacher_model.buffers(), student_model.buffers()):
            buffer_t.copy_(buffer_s)
            
    def _update_ema_variables(self, student_model, iteration, alpha=0.999):
        for param_t, param_s in zip(self.teacher_model.parameters(), student_model.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1. - alpha)
        for buffer_t, buffer_s in zip(self.teacher_model.buffers(), student_model.buffers()):
            buffer_t.copy_(buffer_s)
        
    def forward(self, img, simple_img, student_model, iteration, student_data_dict=None, update_ema=False):
        if update_ema:
            self._update_ema_variables(student_model, iteration, alpha=0.999)
        
        if student_data_dict:
            proj_logits = student_data_dict['proj_logits']
            tgt_out = student_data_dict['tgt_out']
            text = student_data_dict['text']
            proj_feat_cua = student_data_dict['proj_rec_feat_cua']
            
            self.teacher_model.eval()
            with torch.no_grad():
                simple_logits, proj_simple_feat_cua, _  = self.teacher_model(simple_img, text, training=True, with_simple_img=False, unlabel=True, test_with_proj=self.cfg.test_with_proj)
            self.teacher_model.train()
            
            bs = simple_img.shape[0]
            metrics = {}
            
            kl_loss = self.cfg.loss_weight['kl'] * char_kl_loss(proj_logits, simple_logits.detach(), tgt_out, self.kl_thresh)
            metrics['KL_Loss'] = kl_loss
            cua = self.cfg.loss_weight['cua'] * cua_loss(proj_feat_cua, proj_simple_feat_cua.detach(), tgt_out, temperature=self.temperature)
            metrics['CUA_Loss'] = cua
            
        else:
            self.teacher_model.eval()
            with torch.no_grad():
                simple_logits, proj_simple_feat_cua, confidence, preds = self.teacher_model(simple_img, training=False, with_simple_img=True, unlabel=True, test_with_proj=self.cfg.test_with_proj)
            self.teacher_model.train()

            logits, proj_feat_cua, pseudo_tgt_out = student_model(img, preds, simple_img=None, training=True, with_simple_img=False, unlabel=True, test_with_proj=True)

            tmp_len = min(simple_logits.shape[1], logits.shape[1])
            simple_logits = simple_logits[:, :tmp_len, :]
            logits = logits[:, :tmp_len, :]
            pseudo_tgt_out = pseudo_tgt_out[:, :tmp_len]
            proj_feat_cua = proj_feat_cua[:, :tmp_len, :]
            proj_simple_feat_cua = proj_simple_feat_cua[:, :tmp_len, :]
            
            bs = img.shape[0]
            metrics = {}
      
            semi_kl_loss = self.cfg.loss_weight['semi_kl'] * char_kl_loss(logits, simple_logits.detach(), pseudo_tgt_out, self.kl_thresh)
            metrics['Semi_KL_Loss'] = semi_kl_loss
            semi_cua_loss = self.cfg.loss_weight['semi_cua'] * cua_loss(
                proj_feat_cua, proj_simple_feat_cua.detach(), pseudo_tgt_out, self.temperature, confidence, self.conf_thresh)
            metrics['Semi_CUA_Loss'] = semi_cua_loss

        loss = 0
        for k, v in metrics.items():
            loss += v
        
        return loss, metrics


def cua_loss(normal, simple, targets, temperature=0.1, confidence=None, thresh=0):
    normal = F.normalize(normal, dim=-1)
    simple = F.normalize(simple, dim=-1)
    bs, n, c = normal.shape

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    length = torch.tensor(n, device=normal.device)
    gathered_length = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(gathered_length, length)
    max_length = max(gathered_length).item()

    extended_targets = F.pad(targets, (0, max_length-n), value=targets.max())
    gathered_targets = [torch.zeros_like(extended_targets) for _ in range(world_size)]
    dist.all_gather(gathered_targets, extended_targets)
    gathered_targets = torch.cat(gathered_targets, dim=0)

    extended_normal = F.pad(normal, (0, 0, 0, max_length-n), value=0)
    extended_simple = F.pad(simple, (0, 0, 0, max_length-n), value=0)
    gathered_simple = [torch.zeros_like(extended_simple) for _ in range(world_size)]
    dist.all_gather(gathered_simple, extended_simple)
    gathered_simple = torch.cat(gathered_simple, dim=0)

    normal = extended_normal.reshape(bs * max_length, c)
    simple = gathered_simple.reshape(world_size * bs * max_length, c)
    targets = gathered_targets.reshape(-1)

    pad_idx = targets == 38
    eos_idx = targets == 0
    if confidence is not None:
        confidence = torch.tensor(confidence, device=targets.device)
        gathered_confidence = [torch.zeros_like(confidence) for _ in range(world_size)]
        dist.all_gather(gathered_confidence, confidence)
        gathered_confidence = torch.cat(gathered_confidence, dim=0)

        low_conf_idx = (gathered_confidence < thresh).unsqueeze(1).repeat(1, max_length).reshape(-1)
        valid_idx = (~(pad_idx | low_conf_idx)).to(torch.float32)
    else:
        valid_idx = (~pad_idx).to(torch.float32)
    local_valid_idx = valid_idx.chunk(world_size, dim=0)[rank]
    positive_mask = (targets[None,] == targets[None,].T).to(torch.float32)
    local_positive_mask = positive_mask.chunk(world_size, dim=0)[rank]

    logits = torch.einsum('ij,kj->ik', [normal, simple]) / temperature
    deno1= torch.exp(-logits) * (local_positive_mask * valid_idx[None, :]) + torch.exp(logits) * ((1 - local_positive_mask) * valid_idx[None, :])
    deno1 = deno1.sum(dim=1, keepdim=True)
    deno2 = torch.exp(logits) - torch.exp(-logits)
    deno = deno2 + deno1
    deno[deno<0] = 0

    log_prob = logits - torch.log(deno + 1e-6)
    log_prob = (log_prob * local_positive_mask).sum(dim=1)
    log_prob = log_prob / torch.einsum('ij->i', [local_positive_mask])
    loss = torch.einsum('i,i->', [-log_prob, local_valid_idx]) / (local_valid_idx.sum() + 1e-6)
    
    return loss


def char_kl_loss(logits, simple_logits, targets, thresh=0.5, temperature=0.4):
    simple_score, simple_index = simple_logits.log_softmax(dim=-1).max(dim=-1)
                    
    pad_idx = targets == targets.max()
    simple_score[pad_idx] = 0
    simple_sample_prob = simple_score.sum(dim=-1).exp()
    simple_mask = simple_sample_prob >= thresh
    simple_conf_mask = simple_mask.view(-1, 1).repeat(1, simple_logits.shape[1])
    simple_final_mask = (~pad_idx & simple_conf_mask)
    simple_conf_ratio = simple_mask.to(torch.float).mean()

    kl_loss = torch.tensor(0, device=logits.device)
    if simple_conf_ratio > 0:
        simple_logits_soft = (simple_logits / temperature).softmax(dim=-1)
        rec_logits_logsoft = F.log_softmax(logits, dim=-1)
        kl_loss = simple_conf_ratio * F.kl_div(
            rec_logits_logsoft[simple_final_mask], 
            simple_logits_soft[simple_final_mask], 
            reduction='batchmean', 
            log_target=False)
        
    return kl_loss
