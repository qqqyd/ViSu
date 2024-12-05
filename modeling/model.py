import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.helpers import named_apply
from functools import partial
from nltk import edit_distance
from data.lmdb_dataset import CharsetAdapter
from .vit import ViT
from .decoder import Decoder
from .utils import Tokenizer, TokenEmbedding

class EncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_length = cfg.max_length
        embed_dim = cfg.decode_head.embed_dim
        
        self.tokenizer = Tokenizer(cfg.charset_train)
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id
        self.charset_adapter = CharsetAdapter(cfg.charset_test)
        
        self.backbone = ViT(**cfg.backbone)
        self.rec_decoder = Decoder(**cfg.decode_head)
        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=cfg.decode_head.dropout)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        nn.init.trunc_normal_(self.pos_queries, std=.02)
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_head_cua = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        named_apply(partial(self.init_weights, exclude=['backbone']), self)

    def init_weights(self, module, name='', exclude=()):
        if any(map(name.startswith, exclude)):
            return
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, img):
        return self.backbone(img)
    
    def rec_decode(self, 
                   tgt, 
                   memory, 
                   tgt_mask=None, 
                   tgt_padding_mask=None, 
                   tgt_query=None, 
                   tgt_query_mask=None):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.rec_decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)
    
    def forward(self, 
                img, 
                rec_labels=None, 
                simple_img=None, 
                training=False, 
                with_simple_img=True, 
                unlabel=False, 
                test_with_proj=True, 
                test_with_reverse=False):
        bs = img.shape[0]
        
        if training:
            rec_feat = self.encode(img)
            if with_simple_img:
                simple_feat = self.encode(simple_img)
            
            tgt = self.tokenizer.encode(rec_labels).cuda()
            num_steps = min(self.max_length + 1, tgt.shape[1] - 1)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
            tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf')), 1).cuda()
            rec_out = self.rec_decode(tgt_in, rec_feat, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            
            if unlabel:
                proj_rec_feat = self.proj_head(rec_out)
                logits = self.head(proj_rec_feat) if test_with_proj else self.head(rec_out)
                proj_rec_feat_cua = self.proj_head_cua(rec_out)

                return logits, proj_rec_feat_cua, tgt_out
            
            elif with_simple_img:
                simple_rec_out = self.rec_decode(tgt_in, simple_feat, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
                simple_logits = self.head(simple_rec_out)
                logits = self.head(rec_out)

                proj_rec_feat = self.proj_head(rec_out)
                proj_logits = self.head(proj_rec_feat)
                proj_rec_feat_cua = self.proj_head_cua(rec_out)

                metrics = {}

                simple_rec_loss = F.cross_entropy(simple_logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
                metrics['Simple_Rec'] = self.cfg.loss_weight['simple_rec'] * simple_rec_loss
                rec_loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
                metrics['Rec'] = self.cfg.loss_weight['rec'] * rec_loss
                loss = 0 * F.cross_entropy(proj_logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)  # dummy forward
                for k, v in metrics.items():
                    loss += v
                
                return loss, metrics, tgt_out, proj_logits, proj_rec_feat_cua
        else:
            num_steps = self.max_length
            num_samples = img.shape[0] // 2 if test_with_reverse else img.shape[0]
            
            correct = []
            ned = []
            confidence = []
            label_length = []

            rec_feat = self.encode(img)

            bs = img.shape[0]
            pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
            tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf')), 1).cuda()
            
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long).cuda()
            tgt_in[:, 0] = self.bos_id
            
            logits = []
            proj_feat_cua = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = self.rec_decode(tgt_in[:, :j], rec_feat, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j], tgt_query_mask=query_mask[i:j, :j])
                proj_tgt_out = self.proj_head(tgt_out)
                proj_tgt_out_cua = self.proj_head_cua(tgt_out)
                proj_feat_cua.append(proj_tgt_out_cua)
                if test_with_proj:
                    p_i = self.head(proj_tgt_out)
                else:
                    p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)
            proj_feat_cua = torch.cat(proj_feat_cua, dim=1)
            
            probs = logits.softmax(-1)
            preds, probs = self.tokenizer.decode(probs)

            if test_with_reverse:
                confs = np.array([prob.prod().item() for prob in probs])
                conf1, conf2 = np.split(confs, 2)
                preds = [(preds[int(bias) * num_samples + idx], confs[int(bias) * num_samples + idx]) for idx, bias in enumerate(conf1 < conf2)]
            else:
                preds = [(preds[idx], prob.prod().item()) for idx, prob in enumerate(probs)]

            if rec_labels is None:
                for pred, conf in preds:
                    confidence.append(conf)
                    pred = self.charset_adapter(pred)
                    ned.append(0)
                    correct.append(0)
                    label_length.append(len(pred))
            else:

                for (pred, conf), gt in zip(preds, rec_labels):
                    confidence.append(conf)
                    pred = self.charset_adapter(pred)
                    ned.append(edit_distance(pred, gt) / max(len(pred), len(gt)) if max(len(pred), len(gt)) > 0 else 0)
                    correct.append(int(pred == gt))
                    label_length.append(len(pred))

            if unlabel:
                return logits, proj_feat_cua, np.array(confidence), np.array([tmp[0] for tmp in preds])
            else:
                return num_samples, np.array(correct), np.array(ned), np.array(confidence), np.array(label_length), np.array(preds)
