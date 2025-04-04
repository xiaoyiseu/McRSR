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
| task |   R1   |   R5   |  R10   |  mAP   |  
|:----:| ----:|----:|----:|----:|
| t2i  | 58.650 | 80.200 | 87.700 | 45.558 | 
| i2t  | 63.900 | 85.100 | 91.800 | 43.019 | 

2.[CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) /
[best model](https://pan.baidu.com/s/1vVYIGgIaMvtrZTCR2Grolg?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   | 
|:----:| ----:|----:|----:|----:|
| t2i  | 69.136 | 86.875 | 92.138 | 60.279 | 
| i2t  | 79.440 | 95.381 | 97.690 | 55.707 | 

3.[ICFG-PEDES](https://github.com/zifyloo/SSAN) /
[best model](https://pan.baidu.com/s/1WgmcoeT1IKh7NtmbsQGuFA?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   | 
|:----:| ----:|----:|----:|----:|
| t2i  | 59.769 | 77.131 | 82.981 | 33.289 |
| i2t  | 62.288 | 82.855 | 88.387 | 31.453 | 

4.[Flickr 30K](https://ieeexplore.ieee.org/document/7298932/?arnumber=7298932) /
[best model](https://pan.baidu.com/s/1gqI3ILimj8FFTrVESKPLVw?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |
|:----:| ----:|----:|----:|----:|
| t2i  | 61.969 | 79.522 | 85.102 | 70.355 | 
| i2t  | 73.356 | 88.172 | 92.450 | 61.627 | 

5.[MSCOCO (5K)](http://link.springer.com/10.1007/978-3-319-10602-1_48) /
[best model](https://pan.baidu.com/s/1uxTHlVl-Ham2iOboCU9JPg?pwd=1234 )
| task |   R1   |   R5   |  R10   |  mAP   |  
|:----:| ----:|----:|----:|----:|
| t2i  | 49.004 | 76.831 | 85.581 | 61.451 | 
| i2t  | 59.640 | 85.680 | 92.340 | 49.949 | 

