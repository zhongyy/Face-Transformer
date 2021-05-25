import torch
import torch.nn as nn
import sys
from vit_pytorch import ViT_face
from util.utils import get_val_data, perform_val, perform_val_deit, buffer_val, test_forward
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter_yy import FaceDataset
import torch.utils.data as data
import argparse
import os

def main(args):
    print(args)
    DEVICE = torch.device("cuda:0")
    DATA_ROOT = './Data/ms1m-retinaface-t1/'
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]

    model = ViT_face(
        image_size=112,
        patch_size=8,
        loss_type='CosFace',
        GPU_ID= DEVICE,
        num_class=NUM_CLASS,
        dim=512,
        depth=20,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    model_root = args.model
    model.load_state_dict(torch.load(model_root))

    TARGET = [i for i in args.target.split(',')]
    vers = get_val_data('./eval/', TARGET)
    for ver in vers:
        name, data_set, issame = ver
        time = test_forward(DEVICE, model, data_set)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='pretrained model')
    parser.add_argument('--target', default='lfw', help='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))