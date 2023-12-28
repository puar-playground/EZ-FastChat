import numpy as np
from utils.visualize import show_layout
from utils.VisionGenome import VisionGenome
from utils.coco import CocoSceneGraphDataset
import random
import json
from tqdm import tqdm
import math
import torch
from utils.LLM_utils import *


if __name__ == "__main__":

    split = 'train'
    coco_dataset = CocoSceneGraphDataset(instances_json=f'./data/coco_stuff/instances_{split}2017.json',
                                         caption_json=f'./data/coco_stuff/captions_{split}2017.json')

    cap2box = []
    cap2sg = []

    for i in tqdm(range(len(coco_dataset))):

        objs, objs_names, boxes, triples, triple_names, size, image_id, cap = coco_dataset[i]
        # print('triple_names', triple_names)
        boxes[:, [0, 2]] *= size[0]
        boxes[:, [1, 3]] *= size[1]
        boxes = boxes.to(torch.int64).numpy().tolist()
        # print('boxes', boxes)
        # print('size', size)
        show_layout(boxes, objs_names, size=size, out_name='coco')
        # print('caption:', cap)


        prompt_sg = sg_instance(id=i, caption=cap, triples=triple_names)
        cap2sg.append(prompt_sg)

        prompt_bbox = bbox_instance(id=i, caption=cap, bbox=boxes, entity=objs_names, size=size)
        cap2box.append(prompt_bbox)

    print(len(cap2sg))
    json.dump(cap2sg, open(f'./data/finetune/sg_{split}.json', 'w'), indent=4)
    json.dump(cap2box, open(f'./data/finetune/box_{split}.json', 'w'), indent=4)
