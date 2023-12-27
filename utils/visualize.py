import os
import random
import gdown
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def download_arial(save_dir='./assets/'):
    os.makedirs(save_dir, exist_ok=True)
    file_dir = os.path.join(save_dir, 'arial.ttf')
    if not os.path.exists(file_dir):
        url = 'https://drive.google.com/uc?id=1GFuRwakFIPxItZes4isN3csQEOQOcWUE'
        gdown.download(url, save_dir, quiet=False)
    # else:
    #     print(f'arial.ttf already exists : {save_dir}')


def show_layout(bbox_list, tag_list, size=None, save_dir='./plot', background=None, out_name='layout'):

    os.makedirs(save_dir, exist_ok=True)
    download_arial('./assets/')

    if background is None:
        if size is None:
            bbox = np.array(bbox_list)
            size = (np.max(bbox[:, 2]), np.max(bbox[:, 3]))
        background = Image.new('RGB', size, (200, 200, 200))

    draw = ImageDraw.ImageDraw(background)

    font = ImageFont.truetype('./assets/arial.ttf', 16)

    for bbox, tag in zip(bbox_list, tag_list):
        l, t, r, b = bbox
        draw.rectangle(bbox, outline="yellow")
        draw.text((l, t), tag, font=font, fill="blue")
    background.save(os.path.join(save_dir, f'{out_name}.jpg'))


if __name__ == "__main__":

    bbox_list = [[19, 16, 97, 44], [28, 47, 93, 65], [21, 68, 102, 95]]
    tag_list = ['"monkey"', '"holding"', '"banana"']
    show_layout(bbox_list, tag_list)

