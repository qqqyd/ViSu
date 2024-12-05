import io
import re
import unicodedata
import lmdb
import random
import data.standard_text as standard_text
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CharsetAdapter:
    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        label = re.sub(self.unsupported, '', label)
        return label


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self, root, charset, max_label_len, min_image_dim=0, remove_whitespace=True, 
                 normalize_unicode=True, unlabelled=False, rec_transform=None, simple_transform=None,
                 filter_long=True, filter_empty=True, flip_prob=0, rot2horizon=False, font_path=None,
                 training=False, ar_thresh=1.0):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.rec_transform = rec_transform
        self.simple_transform = simple_transform
        self.filter_long = filter_long
        self.filter_empty = filter_empty
        self.flip_prob = flip_prob
        self.rot2horizon = rot2horizon
        self.training = training
        self.ar_thresh = ar_thresh
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)
        self.std_text = standard_text.StdText(Path(font_path), charset) if self.training else None

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if self.filter_long and (len(label) > max_label_len):
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if self.filter_empty and (not label):
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if min(w, h) < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            index += 1
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]
            
        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')
        
        flip_flag = random.random() < self.flip_prob
        if self.unlabelled:
            if flip_flag:
                img = img.transpose(Image.ROTATE_180)
            
            simple_img = self.simple_transform(img)
            img = self.rec_transform(img)
            
            return '{:09d}'.format(index), img, simple_img, label
        elif self.training:
            simple_img = self.std_text.draw_text(label)
            simple_img = Image.fromarray(simple_img)
            if flip_flag:
                simple_img = simple_img.transpose(Image.ROTATE_180)
            if self.rec_transform is not None:
                simple_img = self.rec_transform(simple_img)

            if self.rot2horizon and img.size[0] / img.size[1] < self.ar_thresh:
                img = img.transpose(Image.ROTATE_270)
            if flip_flag:
                img = img.transpose(Image.ROTATE_180)
            if self.rec_transform is not None:
                img = self.rec_transform(img)
                
            return '{:09d}'.format(index), img, simple_img, label
        else:       
            if self.rot2horizon and img.size[0] / img.size[1] < self.ar_thresh:
                img1 = img.transpose(Image.ROTATE_270)
            else:
                img1 = img
            img2 = img.transpose(Image.ROTATE_180)
            if self.rec_transform is not None:
                img1 = self.rec_transform(img1)
                img2 = self.rec_transform(img2)
            
            return '{:09d}'.format(index), img1, img2, label
