import torch
import torch.nn as nn
import sys
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from util.utils import get_val_data, perform_val
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os

def main(args):
    print(args)
    MULTI_GPU = False
    DEVICE = torch.device("cuda:0")
    DATA_ROOT = '/raid/Data/ms1m-retinaface-t1/'
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]

    if args.network == 'VIT' :
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
    elif args.network == 'VITs':
        model = ViTs_face(
            loss_type='CosFace',
            GPU_ID=DEVICE,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    model_root = args.model
    model.load_state_dict(torch.load(model_root))


    #debug
    w = torch.load(model_root)
    for x in w.keys():
        print(x, w[x].shape)
    #embed()
    TARGET = [i for i in args.target.split(',')]
    vers = get_val_data('./eval/', TARGET)
    acc = []

    for ver in vers:
        name, data_set, issame = ver
        accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, 512, args.batch_size,
                                                                      model, data_set, issame)
        print('[%s]XNorm: %1.5f' % (name, xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (name, accuracy, std))
        print('[%s]Best-Threshold: %1.5f' % (name, best_threshold))
        acc.append(accuracy)
    print('Average-Accuracy: %1.5f' % (np.mean(acc)))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='training set directory')
    parser.add_argument('--network', default='VITs',
                        help='training set directory')
    parser.add_argument('--target', default='lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30',
                        help='')
    parser.add_argument('--batch_size', type=int, help='', default=20)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))