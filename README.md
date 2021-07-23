
<h3 align="center">Change is Everywhere <br>Single-Temporal Supervised Object Change Detection <br>for High Spatial Resolution Remote Sensing Imagery</h3>


<h5 align="right">by <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Ailong Ma, <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a> and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper(soon)`](#)] [[`Project(soon)`](#)] [[`BibTeX`](#Citation)]

<p style="text-align: justify">
For high spatial resolution (HSR) remote sensing images, bitemporal supervised learning always dominates change detection using many pairwise labeled bitemporal images. However, it is very expensive and time-consuming to pairwise label large-scale bitemporal HSR remote sensing images. In this paper, we propose single-temporal supervised learning (STAR) for change detection from a new perspective of exploiting object changes in arbitrary image pairs as the supervisory signals. STAR enables us to train a high-accuracy change detector only using <b>unpaired</b> labeled images and generalize to paired bitemporal images. To evaluate the effectiveness of STAR, we design a simple yet effective change detector called ChangeStar, which can reuse any deep semantic segmentation architecture by the ChangeMixin module.
The comprehensive experimental results show that ChangeStar outperforms the baseline with a large margin under single-temporal supervision and achieves superior performance under bitemporal supervision.
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/changestar.png"><br><br>
</div>

This is an official implementation of STAR and ChangeStar in our ICCV 2021 paper [Change is Everywhere: Single-Temporal Supervised Object Change Detection for High Spatial Resolution Remote Sensing Imagery](#).

We hope that STAR will serve as a solid baseline and help ease future research in weakly-supervised object change detection.


---------------------
## News

- 2020/07/23, The code will be released soon.
- 2020/07/23, This paper is accepted by ICCV 2021.

## Features

- Learning a good change detector from single-temporal supervision.
- Strong baselines for bitemporal and single-temporal supervised change detection.
- A clean codebase for weakly-supervised change detection.
- Support both bitemporal and single-temporal supervised settings

## <a name="Citation"></a>Citation
If you use STAR or ChangeStar in your research, please cite the following paper:
```text
@inproceedings{zheng2021change,
  title={Change is Everywhere: Single-Temporal Supervised Object Change Detection for High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Ma, Ailong and Liangpei Zhang and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={},
  year={2021}
}
```



<!-- ## Getting Started
### Install EVER

```bash
pip install --upgrade git+https://github.com/Z-Zheng/ever.git
```

#### Requirements:
- pytorch >= 1.4.0
- python >=3.6 -->




