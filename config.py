import torch, os
import yaml
from IPython import embed


def get_config(args):
    configuration = dict(
        SEED=1337,  # random seed for reproduce results
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        EMBEDDING_SIZE=512,  # feature dimension
    )

    if args.workers_id == 'cpu' or not torch.cuda.is_available():
        configuration['GPU_ID'] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration['GPU_ID'] = [int(i) for i in args.workers_id.split(',')]
    if len(configuration['GPU_ID']) == 0:
        configuration['DEVICE'] = torch.device('cpu')
        configuration['MULTI_GPU'] = False
    else:
        configuration['DEVICE'] = torch.device('cuda:%d' % configuration['GPU_ID'][0])
        if len(configuration['GPU_ID']) == 1:
            configuration['MULTI_GPU'] = False
        else:
            configuration['MULTI_GPU'] = True

    configuration['NUM_EPOCH'] = args.epochs
    configuration['BATCH_SIZE'] = args.batch_size

    if args.data_mode == 'retina':
        configuration['DATA_ROOT'] = './Data/ms1m-retinaface-t1/'
    else:
        raise Exception(args.data_mode)
    configuration['EVAL_PATH'] = './eval/'
    assert args.net in [ 'VIT','VITs']
    configuration['BACKBONE_NAME'] = args.net
    assert args.head in ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']
    configuration['HEAD_NAME'] = args.head
    configuration['TARGET'] = [i for i in args.target.split(',')]

    if args.resume:
        configuration['BACKBONE_RESUME_ROOT'] = args.resume
    else:
        configuration['BACKBONE_RESUME_ROOT'] = ''  # the root to resume training from a saved checkpoint
    configuration['WORK_PATH'] = args.outdir  # the root to buffer your checkpoints
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration
