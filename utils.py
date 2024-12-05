import logging
import torch
from pathlib import Path, PurePath


class Logger:
    def __init__(self, cfg, filename='log.txt'):
        self.log_dir = cfg.workdir
        if not Path(self.log_dir).exists():
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.logger = self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger('message')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(PurePath(self.log_dir, self.filename))
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger


class CheckpointSaver:
    def __init__(self, save_path, save_top_k=3):
        self.save_path = save_path
        self.save_top_k = save_top_k
        self.best_acc = {}
        
    def save(self, model, solver, step=0, acc=0, last=False):
        if last:
            torch.save(model.module.state_dict(), str(PurePath(self.save_path, 'last.ckpt')))
            return True, self.best_acc
        
        ckpt_name = 'iter={}-acc={:.2f}.ckpt'.format(step, acc)
        if len(self.best_acc) < self.save_top_k:
            self.best_acc[ckpt_name] = acc
            torch.save(model.module.state_dict(), str(PurePath(self.save_path, ckpt_name)))
            return True
        
        cur_min_acc = min(self.best_acc.values())
        if acc >= cur_min_acc:
            for k, v in self.best_acc.items():
                if v == cur_min_acc:
                    assert self.best_acc.pop(k, None) is not None
                    (Path(self.save_path) / k).unlink()
                    
                    self.best_acc[ckpt_name] = acc
                    torch.save(model.module.state_dict(), str(PurePath(self.save_path, ckpt_name)))
                    assert len(self.best_acc) == self.save_top_k
                    return True
     
        return False