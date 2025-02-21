import os
import json
import os.path as op
from typing import List
import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
class RSTPReid(BaseDataset):
    def __init__(self, root='', verbose=True):
        self.dataset_dir = op.join(root, 'RSTPReid')  # 数据集根目录
        self.img_dir = op.join(self.dataset_dir, 'imgs/')  # 图像文件夹

        # 直接读取train.json, val.json 和 test.json
        self.train_anno_path = op.join(self.dataset_dir, 'processed_data', 'train.json')
        self.val_anno_path = op.join(self.dataset_dir, 'processed_data', 'val.json')
        self.test_anno_path = op.join(self.dataset_dir, 'processed_data', 'test.json')

        self._check_before_run()

        # 加载并处理数据集
        self.train_annos = self._load_json(self.train_anno_path)
        self.val_annos = self._load_json(self.val_anno_path)
        self.test_annos = self._load_json(self.test_anno_path)

        # 处理数据并获取对应的 ID 容器
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.val, self.val_id_container = self._process_anno(self.val_annos)
        self.test, self.test_id_container = self._process_anno(self.test_annos)

        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
            self.show_dataset_info()

    def _load_json(self, anno_path: str):
        """
        加载 JSON 文件
        """
        with open(anno_path, 'r') as f:
            return json.load(f)

    def _load_json(self, anno_path: str):
        """
        加载 JSON 文件
        """
        with open(anno_path, 'r') as f:
            return json.load(f)

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0

            pid_container = {anno['id'] for anno in annos}  # 先获取所有 pid
            pid_mapping = {pid: idx for idx, pid in enumerate(sorted(pid_container))}  # 创建 pid 到新的连续编号的映射

            for anno in annos:
                pid = int(anno['id'])
                new_pid = pid_mapping[pid]  # 获取映射后的 pid
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions']  # caption list
                for caption in captions:
                    dataset.append((new_pid, image_id, img_path, caption))
                image_id += 1
            return dataset, pid_mapping
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []

            # 创建 pid_mapping
            pid_container = {anno['id'] for anno in annos}  # 先获取所有 pid
            pid_mapping = {pid: idx for idx, pid in enumerate(sorted(pid_container))}  # 创建 pid 到新的连续编号的映射

            for anno in annos:
                pid = int(anno['id'])
                new_pid = pid_mapping[pid]  # 获取映射后的 pid
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(new_pid)  # 使用新的 pid
                caption_list = anno['captions']  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(new_pid)  # 使用新的 pid
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_mapping

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' 不存在")
        if not op.exists(self.img_dir):
            raise RuntimeError(f"'{self.img_dir}' 不存在")
        if not op.exists(self.train_anno_path):
            raise RuntimeError(f"'{self.train_anno_path}' 不存在")
        if not op.exists(self.val_anno_path):
            raise RuntimeError(f"'{self.val_anno_path}' 不存在")
        if not op.exists(self.test_anno_path):
            raise RuntimeError(f"'{self.test_anno_path}' 不存在")
