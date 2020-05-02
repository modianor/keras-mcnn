# -*- coding:utf-8 -*-
"""
   File Name:     create_heatmaps.py
   Description:   create gt heatmaps of test set
   Author:        steven.yi
   date:          2019/04/17
"""
import argparse
import os

import numpy as np

from config import current_config as cfg
from utils.data_loader import DataLoader
from utils.heatmap import save_heatmap


def main(args):
    dataset = args.dataset  # 'A' or 'B'
    output_dir = os.path.join(cfg.HM_GT_PATH, 'Part_{}'.format(dataset))

    for _dir in [cfg.HM_GT_PATH, output_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    test_path = cfg.TEST_PATH.format(dataset)
    test_gt_path = cfg.TEST_GT_PATH.format(dataset)
    # load data
    data_loader = DataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True)
    # data_loader = ImageDataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True,pre_load=True)

    # create heatmaps
    print('Creating heatmaps for Part_{} ...'.format(dataset))
    for i, (img, den) in enumerate(data_loader):
        data = img
        gt = den
        img_name = data_loader.filename_list[i]
        gt = np.squeeze(gt)  # shape(1, h, w, 1) -> shape(h, w)
        save_heatmap(gt, data, img_name, output_dir, gt=True)
    print('All Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create heatmaps for", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
