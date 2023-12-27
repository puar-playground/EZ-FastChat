import h5py
import json
from .visualize import show_layout
import PIL
from PIL import Image
import io
import os
import numpy as np
import requests
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class VisionGenome(Dataset):
    def __init__(self, split='train', data_path='./vg'):
        assert split in {'train', 'test', 'val'}
        self.split = split
        self.data_path = data_path
        vocab = json.load(open(os.path.join(data_path, 'vocab.json'), 'rb'))
        self.entities = vocab['object_idx_to_name']
        self.relations = vocab['pred_idx_to_name']
        self.data = h5py.File(os.path.join(data_path, f'{split}.h5'), "r")
        print(self.data.keys())
        self.image_paths = [x.decode("utf-8") for x in self.data['image_paths']]
        self.image_ids = [i - 1 for i in self.data['image_ids']]
        print(f'loaded, {len(self.entities)} entities, {len(self.relations)} relations, {self.__len__()} graphs')

        self.meta = json.load(open(os.path.join(data_path, 'meta.json'), 'rb'))

    def __getitem__(self, index):
        entity_names = [self.entities[i] for i in self.data['object_names'][index] if i >= 0]
        relationships = [self.relations[i] for i in self.data['relationship_predicates'][index] if i >= 0]
        obj = [entity_names[i] for i in self.data['relationship_objects'][index] if i >= 0]
        sbj = [entity_names[i] for i in self.data['relationship_subjects'][index] if i >= 0]
        bbox = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in self.data['object_boxes'][index] if b[0] != -1]
        triplet = [(s, r, o) for o, r, s in zip(obj, relationships, sbj)]
        img_path = self.image_paths[index]
        size = self.meta[img_path]

        return {'triplet': triplet, 'bbox': bbox, 'size': size}

    def __len__(self):
        return len(self.image_ids)

    def show(self, index, save_dir='./plot/'):
        self.mkdir(save_dir)

        entity_names = [self.entities[i] for i in self.data['object_names'][index] if i >= 0]
        bbox = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in self.data['object_boxes'][index] if b[0] != -1]
        img_path = self.image_paths[index]

        img = Image.open(os.path.join(self.data_path, 'raw', img_path))
        show_layout(bbox, entity_names, save_dir=save_dir, background=img, out_name=f'vg_{self.split}_{index}')

    def mkdir(self, target_dir):
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)


if __name__ == "__main__":

    vg_dataset = VisionGenome(split='test', data_path='./datasets/vg')
    print(vg_dataset[10])
    vg_dataset.show(10, save_dir='./plot')