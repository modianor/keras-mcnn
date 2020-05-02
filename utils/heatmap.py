# -*- coding:utf-8 -*-
"""
   File Name:     heatmap.py
   Description:   generate and save heatmap
   Author:        steven.yi
   date:          2019/04/17
"""
import os

import cv2
import numpy as np
from PIL import Image
from pyheatmap.heatmap import HeatMap


def save_heatmap(density_map, img, img_name, output_dir, down_sample=True, gt=False):
    """
    生成热力图并保存
    :param density_map: 2d-array, 密度图
    :param img: numpy [B,H,W,1]
    :param img_name: "abc.jpg"
    :param output_dir: 结果保存目录
    :param down_sample: bool, 是否有下采样
    :param gt: bool, 是否生成gt的热力图
    :return:
    """
    counts = int(np.sum(density_map))  # 人数
    print('generating heatmap for', img_name)

    # 如果密度图进行下采样4倍, 则需要还原到原始大小

    if down_sample:
        h, w = density_map.shape
        den_resized = np.zeros((h * 4, w * 4))
        for i in range(h*4):
            for j in range(w*4):
                den_resized[i][j] = density_map[int(i / 4)][int(j / 4)] / 16
        density_map = den_resized

    density_map = density_map * 1000
    data = []
    h, w = img.shape[1:3]
    for row in range(h):
        for col in range(w):
            try:
                for k in range(int(density_map[row][col])):
                    data.append([col + 1, row + 1])
            except IndexError:
                continue
    # 生成heatmap
    hm = HeatMap(data, width=w, height=h)
    # 保存heatmap
    hm_name = 'heatmap_' + img_name.split('.')[0] + '.png'
    hm.heatmap(save_as=os.path.join(output_dir, hm_name))

    # 使用蓝色填充heatmap背景, 并显示人群数量
    im = Image.open(os.path.join(output_dir, hm_name))
    x, y = im.size
    bg = Image.new('RGBA', im.size, (0, 0, 139))
    bg.paste(im, (0, 0, x, y), im)
    im_arr = np.array(bg)
    text = 'GT Count: {}'.format(counts) if gt else 'Est Count: {}'.format(counts)
    cv2.putText(im_arr, text, (10, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    im = Image.fromarray(im_arr)
    im.save(os.path.join(output_dir, hm_name))
