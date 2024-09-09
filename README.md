# OcularSeg  🔥

This repository contains the training/testing codes of our paper entitled "OcularSeg: Accurate and Efficient Multi-Modal Ocular Segmentation in Non-Constrained Scenarios".

## Requirements
* `Python>=3.8`
* `Pytorch>=1.13.0` 
* `timm>=0.5`

## Datasets
 - The UBIRIS.v2 dataset is publicly available at [http://iris.di.ubi.pt/ubiris2.html](http://iris.di.ubi.pt/ubiris2.html). 
 - The SBVPI and MOBIUS datasets are publicly available at [https://sclera.fri.uni-lj.si/datasets.html](https://sclera.fri.uni-lj.si/datasets.html). 
 - The ESLD dataset is publicly available at [https://www.cjig.cn/zh/article/doi/10.11834/jig.210177/](https://www.cjig.cn/zh/article/doi/10.11834/jig.210177/). 
 - The MMU dataset is publicly available at [https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset). 
 - Our segmentation annotations for UBIRIS.v2, SBVPI, and MMU can be found in [Quark Drive](https://pan.quark.cn/s/20231b46cb72).

If you are interested in utilizing our segmentation annotations for your research experiments, please email us for the access code to Quark Drive.

## Experiments
1. **How to test the model**
   +  Downloading our trained weight from [Google Drive](https://drive.google.com/file/d/1DbJQ26T50KPznXuL6WNNJM3711elIB7P/view?usp=sharing), and move it into `./checkpoints`
	+ modifying some settings in `test.py`
	+ running `test.py`
2.  **How to train the model**
	+ running `train.py`

## Reference
Some of the codes in this repo are borrowed from:
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [M<sup>2</sup>SNet](https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet)
* [Expectation-Maximization Attention Networks](https://github.com/XiaLiPKU/EMANet)

### Citation
If you find our work useful in your research, please consider citing:

```
@article{zhang2024ocularseg,
  title={OcularSeg: Accurate and Efficient Multi-Modal Ocular Segmentation in Non-Constrained Scenarios},
  author={Zhang, Yixin and Wang, Caiyong and Li, Haiqing and Sun, Xianyun and Tian, Qichuan and Zhao, Guangzhe},
  journal={Electronics},
  volume={13},
  number={10},
  pages={1967},
  year={2024}
}
```
	
### Questions

Please contact zhangyixin@stu.bucea.edu.cn (Miss Zhang) or wangcaiyong@bucea.edu.cn (Dr. Wang).