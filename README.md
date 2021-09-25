# Mining Latent Classes for Few-shot Segmentation

[Lihe Yang](https://github.com/LiheYoung), [Wei Zhuo](https://scholar.google.com.au/citations?user=Q-UjnzEAAAAJ&hl=zh-CN), [Lei Qi](http://palm.seu.edu.cn/qilei/), [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/), [Yang Gao](https://cs.nju.edu.cn/gaoyang/)

![](./docs/pipeline.png)



The codebase contains baseline of our paper [Mining Latent Classes for Few-shot Segmentation](https://arxiv.org/abs/2103.15402), ICCV 2021 Oral.

Some key modifications to the simple yet effective metric learning framework:
- Remove the final residual stage in ResNet for stronger generalization
- Remove the final ReLU for feature matching
- Freeze all the BatchNorms from ImageNet pretrained model


## Data preparation

### Download

**Pretrained model:** [ResNet-50](https://drive.google.com/file/d/11yONyypvBEYZEh9NIOJBGMdiLLAgsMgj/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1mX1yYvkcyOkAVjZZSIf6uMBPlooZCmpk/view?usp=sharing)

**Dataset:** [Pascal images and ids](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
| [Semantic segmentation annotations](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)

### File organization

```
├── ./pretrained
    ├── resnet50.pth
    └── resnet101.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    ├── SegmentationClass
    └── ImageSets
```


## Run the code

```
CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset pascal --data-root [Your Pascal Path] \
  --backbone resnet50 --fold 0 --shot 1
```

You may change the ``backbone`` from ``resnet50`` to ``resnet101``, change the ``fold`` from ``0`` to ``1/2/3``, or change the ``shot`` from ``1`` to ``5`` for other settings.

## Acknowledgement

We thank [PANet](https://arxiv.org/abs/1908.06391), [PPNet](https://arxiv.org/abs/2007.06309), [PFENet](https://arxiv.org/abs/2008.01449) and other FSS works for their great contributions.


## Citation

```bibtex
@inproceedings{yang2021mining,
  title={Mining Latent Classes for Few-shot Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  journal={ICCV},
  year={2021}
}
```
