# Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval
We provide the code for reproducing result of our paper “Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval”.


## Requirements
Python==3.8.19    
torch==2.3.0+cu121    
numpy==1.24.4    
scikit-learn==1.2.0    
pillow==10.3.0    
 
## Division of each dataset
Before running, it is necessary to modify the corresponding path
```
DatasetSplit.py
```

## Training and Testing
```
python train.py
python test.py
```

# Results on five public datasets
1.[RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset) /
[best model](https://pan.baidu.com/s/1i1kj6CaDaA-UMeg2WHFLhA?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
|:----:| ----:|----:|----:|----:|----:|
| t2i  | 58.650 | 80.200 | 87.700 | 45.558 | 24.153 |
| i2t  | 63.900 | 85.100 | 91.800 | 43.019 | 15.141 |

2.[CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) /
[best model](https://pan.baidu.com/s/1vVYIGgIaMvtrZTCR2Grolg?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
|:----:| ----:|----:|----:|----:|----:|
| t2i  | 69.136 | 86.875 | 92.138 | 60.279 | 42.263 |
| i2t  | 79.440 | 95.381 | 97.690 | 55.707 | 21.161 | 

3.[ICFG-PEDES](https://github.com/zifyloo/SSAN) /
[best model](https://pan.baidu.com/s/1WgmcoeT1IKh7NtmbsQGuFA?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
|:----:| ----:|----:|----:|----:|----:|
| t2i  | 58.152 | 76.391 | 82.598 | 32.487 | 5.183 |
| i2t  | 62.757 | 83.303 | 89.153 | 31.357 | 3.161 |

4.[Flickr 30K](https://ieeexplore.ieee.org/document/7298932/?arnumber=7298932) /
[best model](https://pan.baidu.com/s/1gqI3ILimj8FFTrVESKPLVw?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
|:----:| ----:|----:|----:|----:|----:|
| t2i  | 61.969 | 79.522 | 85.102 | 70.355 | 72.283 |
| i2t  | 73.356 | 88.172 | 92.450 | 61.627 | 39.186 | 

5.[MSCOCO](http://link.springer.com/10.1007/978-3-319-10602-1_48) /
[best model](https://pan.baidu.com/s/1uxTHlVl-Ham2iOboCU9JPg?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
|:----:| ----:|----:|----:|----:|----:|
| t2i  | 49.004 | 76.831 | 85.581 | 61.451 | 61.451 |
| i2t  | 59.640 | 85.680 | 92.340 | 49.949 | 28.558 | 

## Acknowledgments
Some components implemented in this code use [CLIP](https://github.com/openai/CLIP) and [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate their contributions.


## Citation
If this work is helpful for your research, please cite our work:
```

```
