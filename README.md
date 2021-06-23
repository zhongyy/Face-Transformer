# Face-Transformer

This is the code of Face Transformer for Recognition (https://arxiv.org/abs/2103.14803v2). 

Recently there has been great interests of Transformer not only in NLP but also in computer vision. We wonder if transformer can be used in face recognition and whether it is better than CNNs. Therefore, we investigate the performance of Transformer models in face recognition. The models are trained on a large scale face recognition database MS-Celeb-1M and evaluated on several mainstream benchmarks, including LFW, SLLFW, CALFW, CPLFW, TALFW, CFP-FP, AGEDB and IJB-C databases. We demonstrate that Transformer models achieve comparable performance as CNN with similar number of parameters and MACs. 

![arch](https://github.com/zhongyy/Face-Transformer/blob/main/arch.jpg)

## Usage Instructions

### 1. Preparation
The code is mainly adopted from [Vision Transformer](https://github.com/lucidrains/vit-pytorch), and [DeiT](https://github.com/facebookresearch/deit). In addition to PyTorch and torchvision, install [vit_pytorch](https://github.com/lucidrains/vit-pytorch) by [Phil Wang](https://github.com/lucidrains), and package [timm==0.3.2](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). Sincerely appreciate for their contributions. 
All needed Packages are found in requirements.txt -> Simply install all packages by:
```
pip install -r requirements.txt
```

Copy the files of folder "copy-to-vit_pytorch-path" to vit_pytorch path.
```
.
├── __init__.py
├── vit_face.py
└── vits_face.py
```
### 2. Databases
You can download the training databases, MS-Celeb-1M (version [ms1m-retinaface](https://github.com/deepinsight/insightface/tree/master/challenges/iccv19-lfr)), and put it in folder 'Data'. 

You can download the testing databases as follows and put them in folder 'eval'. 

- LFW: [Baidu Netdisk](https://pan.baidu.com/s/1WwFA1lS1_6elleu6kxMGDQ)(password: dfj0), [Google Drive](https://drive.google.com/file/d/17ICjkR3EB8IE-PeoPZRYqYOcFhuaWqar/view?usp=sharing)
- SLLFW: [Baidu Netdisk](https://pan.baidu.com/s/19lb0f9ZkAunKDpTzhJQUag)(password: l1z6), [Google Drive](https://drive.google.com/file/d/1oJZb-8jcJqAfXpg62bzGWpkHeabiqO0Q/view?usp=sharing)
- CALFW: [Baidu Netdisk](https://pan.baidu.com/s/1QyjRZNE0chm9BmobE2iOHQ)(password: vvqe), [Google Drive](https://drive.google.com/file/d/1KRPCobKoVA3MLGqvW6zbOqTZ3lVK3ysD/view?usp=sharing)
- CPLFW: [Baidu Netdisk](https://pan.baidu.com/s/1ZmnIBu1IwBq6pPBGByxeyw)(password: jyp9), [Google Drive](https://drive.google.com/file/d/1IhIChTARWvZwoV0H4khHhNGFs7BdioBG/view?usp=sharing)
- TALFW: [Baidu Netdisk](https://pan.baidu.com/s/1p-qhd2IdV9Gx6F6WaPhe5Q)(password: izrg), [Google Drive](https://drive.google.com/file/d/1hNNi3iz_w0MtYD1vvLDz4Ieq7tzkSQ82/view?usp=sharing) 
- CFP_FP: [Baidu Netdisk](https://pan.baidu.com/s/1lID0Oe9zE6RvlAdhtBlP1w)(password: 4fem), [Google Drive](https://drive.google.com/file/d/13MPwlCqjiO6OqZWQkyHl0CjcJa4UEnEy/view?usp=sharing)--refer to [Insightface](https://github.com/deepinsight/insightface/)
- AGEDB: [Baidu Netdisk](https://pan.baidu.com/s/1vf08K1C5CSF4w0YpF5KEww)(password: rlqf), [Google Drive](https://drive.google.com/file/d/15el0xh5E6tSYJQ1KurAGgfggNjqg_t6d/view?usp=sharing)--refer to [Insightface](https://github.com/deepinsight/insightface/)



### 3. Train Models

- ViT-P8S8
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s1 --warmup-epochs 1 --lr 3e-4 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s2 --warmup-epochs 0 --lr 1e-4 -r path_to_model 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s3 --warmup-epochs 0 --lr 5e-5 -r path_to_model 
```

- ViT-P12S8
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s1 --warmup-epochs 1 --lr 3e-4 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s2 --warmup-epochs 0 --lr 1e-4 -r path_to_model 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s3 --warmup-epochs 0 --lr 5e-5 -r path_to_model 
```

### 4. Pretrained Models and Test Models (on LFW, SLLFW, CALFW, CPLFW, TALFW, CFP_FP, AGEDB)
You can download the following models
- ViT-P8S8: [Baidu Netdisk](https://pan.baidu.com/s/1ppgQe1GG3oa2-uz2zzL6EQ)(password: spkf), [Google Drive](https://drive.google.com/drive/folders/1U7MDZSS38cMIvtEWohaLAH4j7JgBNy0T?usp=sharing)
- ViT-P12S8: [Baidu Netdisk](https://pan.baidu.com/s/1VrDfvz4SvYVnPcTlHVKAkg)(password: 7caa), [Google Drive](https://drive.google.com/drive/folders/1tKjPdDz9WiD-dCjHnkdnyLSs9HS9XUGW?usp=sharing)

You can test Models

The content of “property” file for “ms1m_retinaface” dataset is as follows:
"93431,112,112"

```
python test.py --model ./results/ViT-P12S8_ms1m_cosface/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth --network VIT 

python test.py --model ./results/ViT-P12S8_ms1m_cosface/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth --network VITs 
```


