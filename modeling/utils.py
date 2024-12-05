import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


class Tokenizer():
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset):
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens):
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids, join=True):
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    def encode(self, labels):
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
    
    def decode(self, token_dists, raw=False):
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

    def _filter(self, probs, ids):
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


class CTCTokenizer():
    BLANK = '[B]'

    def __init__(self, charset):
        # BLANK uses index == 0 by default
        specials_first = (self.BLANK,)
        specials_last = ()
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.blank_id = self._stoi[self.BLANK]

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens):
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids, join):
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    def encode(self, labels):
        # We use a padded representation since we don't want to use CUDNN's CTC implementation
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)
    
    def decode(self, token_dists, raw=False):
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

    def _filter(self, probs, ids):
        # Best path decoding:
        ids = list(zip(*groupby(ids.tolist())))[0]  # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]  # Remove BLANKs
        # `probs` is just pass-through since all positions are considered part of the path
        return probs, ids

    
class TokenEmbedding(nn.Module):
    def __init__(self, charset_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)