from model import constraints
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch, os
import torch.nn as nn
from collections import OrderedDict

task_loss_map = {
    'ppe': constraints.PPE_loss,
    'mcu': constraints.MCU_loss,
    'fsa': constraints.FSA_loss
}

class McRSA(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids'] 
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        for task, loss_fn in task_loss_map.items():
            if task in self.current_task:
                ret.update({f'{task}_loss': loss_fn(i_feats, t_feats, batch['pids'])})
        return ret

def build_model(args, num_classes):
    model = McRSA(args, num_classes)
    convert_weights(model)
    return model