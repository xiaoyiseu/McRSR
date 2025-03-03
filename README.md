PyTorch implementation for “Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval”

# Multi-Constraint Relational Semantic Alignment Towards Image-Text Retrieval

## 0. Requirements
Python==3.8.19    
torch==2.3.0+cu121    
numpy==1.24.4    
scikit-learn==1.2.0    
pillow==10.3.0    

## 1. Datasets
[RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset)\
[CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) \
[ICFG-PEDES](https://github.com/zifyloo/SSAN) \
[Flickr30K](https://ieeexplore.ieee.org/document/7298932/?arnumber=7298932) \
[MSCOCO](http://link.springer.com/10.1007/978-3-319-10602-1_48)    

### 1.1 Division of each dataset
```
import os
import json
import errno
import argparse

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', 
                        default='MSCOCO', 
                        type=str, 
                        help = "CUHK-PEDES, ICFG-PEDES, MSCOCO, Flickr_30k, RSTPReid")
    parser.add_argument('--dataset_root_dir', 
                        default=r"./MSCOCO", type=str)
    args = parser.parse_args()
    raw_annotation_file_name = ""
    if args.dataset_name == "CUHK-PEDES":
        raw_annotation_file_name = "reid_raw.json"
    elif args.dataset_name == "ICFG-PEDES":
        raw_annotation_file_name = "ICFG-PEDES.json"
    elif args.dataset_name == "RSTPReid":
        raw_annotation_file_name = "data_captions.json"
    elif args.dataset_name == "MSCOCO":
        raw_annotation_file_name = "caption_all.json"
    elif args.dataset_name == "Flickr_30k":
        raw_annotation_file_name = "caption_all.json"
    raw_annotation_file_path = os.path.join(args.dataset_root_dir, raw_annotation_file_name)
    # split raw annotations into training, validation and test dataset

    with open(raw_annotation_file_path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    train = []
    val = []
    test = []
    for ann in anns:
        if args.dataset_name in ["RSTPReid", "Flickr_30k", "MSCOCO"]:
            ann['file_path'] = ann.pop('img_paths')
        eval(ann['split']).append(ann)
    output_dir = os.path.join(args.dataset_root_dir, "processed_data")
    mkdir_if_missing(output_dir)
    json.dump(train, open(os.path.join(output_dir, "train.json"), 'w'))
    json.dump(val, open(os.path.join(output_dir, "val.json"), 'w'))
    json.dump(test, open(os.path.join(output_dir, "test.json"), 'w'))

```

## 2. training and testing
```
python train.py
python test.py
```

