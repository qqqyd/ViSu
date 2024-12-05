import argparse
import numpy as np
import torch
from pathlib import Path
from mmcv import Config
from tqdm import tqdm
from data.build_dataset import DatasetsBuilder
from modeling.model import EncoderDecoder
from utils import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--with_proj", action="store_true")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint_path = Path(args.checkpoint)
    cfg.workdir = str(checkpoint_path.parent.absolute())
    cfg.model.test_with_proj = args.with_proj
    
    logger = Logger(cfg, str(checkpoint_path.name) + '.eval').logger
    logger.info('Config path: {}'.format(args.config))

    data_cfg = cfg.data
    DataBuilder = DatasetsBuilder(data_cfg)
    test_loader = DataBuilder.test_dataloaders(data_cfg.test_subset)

    model = EncoderDecoder(cfg.model).cuda()
    assert checkpoint_path.exists()
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    
    model.load_state_dict(checkpoint)
    logger.info('Model loaded: {}'.format(str(checkpoint_path)))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of parameters: {}M".format(num_params / 1e6))
    model.eval()
    
    num_list = np.zeros(len(test_loader))
    acc_list = np.zeros(len(test_loader))
    ned_list = np.zeros(len(test_loader))
    conf_list = np.zeros(len(test_loader))
    len_list = np.zeros(len(test_loader))
    
    for idx, (dataset_name, test_dataset) in enumerate(test_loader.items()):
        with torch.no_grad():
            num_samples = 0
            correct = 0
            ned = 0
            confidence = 0
            label_length = 0
            
            for filename, img1, img2, text in tqdm(test_dataset):
                img1 = img1.cuda()
                img2 = img2.cuda()
                img = torch.cat((img1, img2), dim=0)
                tmp_num_samples, tmp_correct, tmp_ned, tmp_confidence, tmp_label_length, rec_pred = model(
                    img, rec_labels=text, training=False, with_simple_img=False, test_with_proj=cfg.model.test_with_proj, test_with_reverse=True)

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
            dataset_name, num_samples, accuracy, mean_ned, mean_conf, mean_label_length)
        logger.info(eval_str)

    total_num = int(num_list.sum())
    ave_acc = (acc_list * num_list).sum() / total_num
    ave_ned = (ned_list * num_list).sum() / total_num
    ave_conf = (conf_list * num_list).sum() / total_num
    ave_len = (len_list * num_list).sum() / total_num
    eval_str = '{:<14} | samples:{:<6} | accuracy:{:<.2f} | NED:{:<.4f} | confidence:{:<.4f} | label length:{:<.4f} '.format(
            'Average', total_num, ave_acc, ave_ned, ave_conf, ave_len)
    logger.info(eval_str)


if __name__ == '__main__':
    main()
