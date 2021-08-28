
<h3 align="center">Change is Everywhere <br>Single-Temporal Supervised Object Change Detection <br>in Remote Sensing Imagery</h3>

<h5 align="right">by <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a> and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper`](https://arxiv.org/abs/2108.07002)] [[`BibTeX`](#Citation)]

<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/changestar.png"><br><br>
</div>

This is an official implementation of STAR and ChangeStar in our ICCV 2021 paper [Change is Everywhere: Single-Temporal Supervised Object Change Detection for High Spatial Resolution Remote Sensing Imagery](#).

We hope that STAR will serve as a solid baseline and help ease future research in weakly-supervised object change detection.


---------------------
## News

- 2021/08/28, The code is available.
- 2021/07/23, The code will be released soon.
- 2021/07/23, This paper is accepted by ICCV 2021.

## Features

- Learning a good change detector from single-temporal supervision.
- Strong baselines for bitemporal and single-temporal supervised change detection.
- A clean codebase for weakly-supervised change detection.
- Support both bitemporal and single-temporal supervised settings

## <a name="Citation"></a>Citation
If you use STAR or ChangeStar (FarSeg) in your research, please cite the following paper:
```text
@inproceedings{zheng2021change,
  title={Change is Everywhere: Single-Temporal Supervised Object Change Detection for High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Ma, Ailong and Liangpei Zhang and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={},
  year={2021}
}

@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}
```



## Getting Started
### Install [EVer](https://github.com/Z-Zheng/ever)

```bash
pip install --upgrade git+https://github.com/Z-Zheng/ever.git
```

#### Requirements:
- pytorch >= 1.6.0
- python >=3.6

### Prepare Dataset
1. Download [xView2](https://xview2.org/) dataset (training set and tier3 set) and [LEVIR-CD](https://drive.google.com/open?id=1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim) dataset. 

2. Create soft link
```bash
ln -s </path/to/xView2> ./xView2
ln -s </path/to/LEVIR-CD> ./LEVIR-CD
```

### Training and Evaluation under Single-Temporal Supervision
```bash
bash ./scripts/trainxView2/r50_farseg_changemixin_symmetry.sh
```

### Training and Evaluation under Bitemporal Supervision
```bash
bash ./scripts/bisup_levircd/r50_farseg_changemixin.sh
```

## License
ChangeStar is released under the [Apache License 2.0](https://github.com/Z-Zheng/ChangeStar/blob/master/LICENSE).

Copyright (c) Zhuo Zheng. All rights reserved.

