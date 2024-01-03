import json
import os

def sg_string(triples):
    s = ''
    for (h, r, t) in triples:
        s += f'({h}, {r}, {t})\n'
    return 'scene graph: ' + s

def system_prompt_sg(caption):
    system_prompt_sg = (f'Given a caption of an image, imagine a scene graph for the image. '
                        f'Imagine a scene graph for the image that describing the visual relationship between objects. '
                        f'Write the scene graph as a list of triples that consists of subject, relation, object. '
                        f'So let us begin. '
                        f'Caption: {caption}')
    return system_prompt_sg


def sg_instance(id, caption, triples):

    instance = {'id': f'identity_{id}', 'conversations': []}

    instance['conversations'] = [{
                "from": "human",
                "value": system_prompt_sg(caption)
            }, {
                "from": "gpt",
                "value": sg_string(triples)
            }]

    return instance



def bbox_string(entity, bbox):
    s = ''
    for e, b in zip(entity, bbox):
        s += f'{e} ({b[0]}, {b[1]}, {b[2]}, {b[3]})\n'
    return 'Layout: ' + s


def system_prompt_bbox(caption, size):

    w, h = size
    system_prompt_box = (f'Given a caption of an image, plan a layout for the image. '
                         f'Print one bounding box in each line, the format should be keyword (left, top, right, bottom). '
                         f'The size of the image is {w}x{h}. '
                         f'Therefore, the left and right coordinates of the bounding box should not exceed {w}, '
                         f'the top and bottom coordinates of the positions should not exceed {h}, '
                         f'including the coordinates of top, left, right, and bottom. '
                         f'So let us begin. '
                         f'Caption: {caption}')
    return system_prompt_box

def bbox_instance(id, caption, entity, bbox, size):

    instance = {'id': f'identity_{id}', 'conversations': []}

    instance['conversations'] = [{
                "from": "human",
                "value": system_prompt_bbox(caption, size)
            }, {
                "from": "gpt",
                "value": bbox_string(entity, bbox)
            }]

    return instance

def system_prompt_sg2bbox():

    system_prompt_sg2box = (f'A scene graph consists of a series of triples, such as (object-1, relation, object-2), '
                            f'that describe the visual relationships between objects in an image. '
                            f'For a given scene graph, plan the layout of the visual objects accordingly. '
                            f'For each object, determine its bounding box coordinates in the format (left, top, right, bottom). '
                            f'Considering the image size is 128 x 128, '
                            f'ensure that all coordinates of a bounding box do not exceed 128. '
                            f'Let us begin. ')

    return system_prompt_sg2box

def entity_string(entity):

    s_e = ', '.join(entity)

    return 'Objects: ' + s_e

def sg2bbox_instance(id, triples, entity, bbox):

    instance = {'id': f'identity_{id}', 'conversations': []}

    query = '\n' + entity_string(entity) + '\n' + sg_string(triples)

    instance['conversations'] = [{
                "from": "human",
                "value": system_prompt_sg2bbox() + query
            }, {
                "from": "gpt",
                "value": bbox_string(entity, bbox)
            }]

    return instance, query




