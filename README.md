# Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval
We provide the code for reproducing result of our paper “Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval”.


## Requirements
Python==3.8.19    
torch==2.3.0+cu121    
numpy==1.24.4    
scikit-learn==1.2.0    
pillow==10.3.0    

## Datasets
[RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset)\
[CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) \
[ICFG-PEDES](https://github.com/zifyloo/SSAN) \
[Flickr30K](https://ieeexplore.ieee.org/document/7298932/?arnumber=7298932) \
[MSCOCO](http://link.springer.com/10.1007/978-3-319-10602-1_48)    

### Division of each dataset
Before running, it is necessary to modify the corresponding path
```
DatasetSplit.py
```

## Training and Testing
```
python train.py
python test.py
```
### pre-trained model
RSTPReid: [best](https://pan.baidu.com/s/1i1kj6CaDaA-UMeg2WHFLhA?pwd=1234 )\
CUHK-PEDES: [best](https://pan.baidu.com/s/1vVYIGgIaMvtrZTCR2Grolg?pwd=1234 )\
ICFG-PEDES: [best](https://pan.baidu.com/s/1WgmcoeT1IKh7NtmbsQGuFA?pwd=1234 )\
Flickr30K: [best](https://pan.baidu.com/s/1gqI3ILimj8FFTrVESKPLVw?pwd=1234 )\
MSCOCO: [best](https://pan.baidu.com/s/1uxTHlVl-Ham2iOboCU9JPg?pwd=1234 )

# Results on five public datasets






## Citation
If this work is helpful for your research, please cite our work:
```

```
