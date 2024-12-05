import logging
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from data.lmdb_dataset import LmdbDataset

logger = logging.getLogger('message')

class DatasetsBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dict = {
            'lmdb': LmdbDataset,
        }
        
        self.train_data = cfg.train if isinstance(cfg.train, list) else [cfg.train]
        self.test_data = cfg.test if isinstance(cfg.test, list) else [cfg.test]
        self.train_unlabel_data = None
        if 'train_unlabel' in cfg and cfg.semi_train:
            self.train_unlabel_data = cfg.train_unlabel if isinstance(cfg.train_unlabel, list) else [cfg.train_unlabel]

    @staticmethod
    def get_transform(img_size, augment=False, rotation=0, resize=True, weak_augment=False):
        transforms = []
        if weak_augment:
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05))
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        if resize:
            transforms.append(T.Resize(img_size, T.InterpolationMode.BICUBIC))
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

    def build_train_dataset(self):
        rec_transform = self.get_transform(self.cfg.data_shape, augment=True, resize=True, weak_augment=False)
        simple_transform = self.get_transform(self.cfg.data_shape, augment=False, resize=True, weak_augment=True)
        flip_prob = 0.5

        datasets = []
        for cur_dataset in self.train_data:
            tmp_name = cur_dataset.get('name', None)
            dataset_name = cur_dataset['data_root'].split('/')[-1] if tmp_name is None else tmp_name
            if cur_dataset.get('type', 'lmdb') == 'lmdb':
                dataset = LmdbDataset(cur_dataset['data_root'], self.cfg.charset_train, self.cfg.max_length, 
                                      rec_transform=rec_transform, simple_transform=None, filter_long=True, 
                                      filter_empty=True, flip_prob=flip_prob, rot2horizon=self.cfg.rot2horizon, 
                                      font_path=self.cfg.font_path, training=True, ar_thresh=self.cfg.ar_thresh)
                datasets.append(dataset)
                logger.info('Load train {}: {} samples'.format(dataset_name, len(dataset)))
            else:
                logger.error('Unsupported dataset type {}'.format(cur_dataset['type']))
        self.train_dataset = ConcatDataset(datasets)
        
        if self.train_unlabel_data is None:
            self.train_unlabel_dataset = None
        else:
            unlabel_datasets = []
            for cur_dataset in self.train_unlabel_data:
                tmp_name = cur_dataset.get('name', None)
                dataset_name = cur_dataset['data_root'].split('/')[-1] if tmp_name is None else tmp_name
                if cur_dataset.get('type', 'lmdb') == 'lmdb':
                    dataset = LmdbDataset(cur_dataset['data_root'], self.cfg.charset_train, self.cfg.max_length, unlabelled=True, 
                                          rec_transform=rec_transform, simple_transform=simple_transform, filter_long=True, 
                                          filter_empty=True, flip_prob=flip_prob, rot2horizon=self.cfg.rot2horizon, 
                                          font_path=self.cfg.font_path, training=True, ar_thresh=self.cfg.ar_thresh)
                    unlabel_datasets.append(dataset)
                    logger.info('Load unlabel train {}: {} samples'.format(dataset_name, len(dataset)))
                else:
                    logger.error('Unsupported dataset type {}'.format(cur_dataset['type']))
                self.train_unlabel_dataset = ConcatDataset(unlabel_datasets)

        return self.train_dataset, self.train_unlabel_dataset
    
    def build_test_dataset(self, subset=None):
        rec_transform = self.get_transform(self.cfg.data_shape, augment=False, resize=True)
        
        datasets = {}
        for cur_dataset in self.test_data:
            tmp_name = cur_dataset.get('name', None)
            dataset_name = cur_dataset['data_root'].split('/')[-1] if tmp_name is None else tmp_name
            if subset is None or dataset_name in subset:
                if cur_dataset.get('type', 'lmdb') == 'lmdb':
                    datasets[dataset_name] = LmdbDataset(
                        cur_dataset['data_root'], self.cfg.charset_test, self.cfg.max_length, rec_transform=rec_transform, 
                        filter_long=False, filter_empty=False, flip_prob=0, rot2horizon=self.cfg.rot2horizon, training=False,
                        ar_thresh=self.cfg.ar_thresh)
                    logger.info('Load test {}: {} samples'.format(dataset_name, len(datasets[dataset_name])))
                else:
                    logger.error('Unsupported dataset type {}'.format(cur_dataset['type']))
        self.test_dataset = datasets

        return self.test_dataset

    def train_dataloader(self):
        self.build_train_dataset()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.samples_per_gpu * self.cfg.gpu_num,
            num_workers=self.cfg.workers_per_gpu * self.cfg.gpu_num,
            persistent_workers=self.cfg.workers_per_gpu > 0,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

    def test_dataloaders(self, subset):
        self.build_test_dataset(subset)
        return {k: DataLoader(v, 
                              batch_size=self.cfg.samples_per_gpu,
                              num_workers=self.cfg.workers_per_gpu,
                              pin_memory=True)
                for k, v in self.test_dataset.items()}
        
    def train_dataloader_ddp(self):
        self.build_train_dataset()
        self.train_sampler = DistributedSampler(self.train_dataset)
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.samples_per_gpu,
            num_workers=self.cfg.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler)
        if self.train_unlabel_dataset is None:
            train_unlabel_dataloader = None
        else:
            self.train_unlabel_sampler = DistributedSampler(self.train_unlabel_dataset)
            train_unlabel_dataloader = None if self.train_unlabel_dataset is None else DataLoader(
                dataset=self.train_unlabel_dataset,
                batch_size=self.cfg.samples_per_gpu,
                num_workers=self.cfg.workers_per_gpu,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
                sampler=self.train_unlabel_sampler)
        
        return train_dataloader, train_unlabel_dataloader

    def train_sampler_set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
    
    def train_unlabel_sampler_set_epoch(self, epoch):
        self.train_unlabel_sampler.set_epoch(epoch)
        