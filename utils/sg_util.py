import itertools
import random, math
import torch

def number_entity(entities):

    e_cnt = {e:0 for e in entities}
    numbered_entities = []
    for e in entities:
        e_cnt[e] += 1
        numbered_entities.append(f'{e}_{e_cnt[e]}')

    numbered_entities = [e if e_cnt[e]==1 else n_e for e, n_e in zip(entities, numbered_entities)]

    return numbered_entities


pred_idx_to_name = [
      '__in_image__',
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding',
    ]
vocab_pred_name_to_idx = {}
for idx, name in enumerate(pred_idx_to_name):
    vocab_pred_name_to_idx[name] = idx

def synth_relation(entities, boxes, vocab_pred_name_to_idx, numbered=True):
    # Add triples
    triples = []
    triple_names = []
    num_objs = len(entities)
    if numbered:
        entities = number_entity(entities)

    g = list(itertools.combinations(range(num_objs), 2))
    reverse = [1 if random.random() > 0.5 else 0 for _ in range(num_objs)]
    pairs = [(a, b) if r == 0 else (b, a) for (r, (a, b)) in zip(reverse, random.sample(g, k=num_objs))]

    for (s, o) in pairs:
        # Check for inside / surrounding
        sx0, sy0, sx1, sy1 = boxes[s]
        ox0, oy0, ox1, oy1 = boxes[o]
        d = torch.FloatTensor([0.5 * ((sx0 + sx1) - (ox0 + ox1)), 0.5 * ((sy0 + sy1) - (oy0 + oy1))])
        theta = math.atan2(d[1], d[0])

        if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            p_name = 'surrounding'
        elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            p_name = 'inside'
        elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            p_name = 'left of'
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            p_name = 'above'
        elif -math.pi / 4 <= theta < math.pi / 4:
            p_name = 'right of'
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            p_name = 'below'
        else:
            p_name = None

        p = vocab_pred_name_to_idx[p_name]
        triples.append([s, p, o])
        triple_names.append([entities[s], p_name, entities[o]])

    return triples, triple_names




if __name__ == '__main__':

    entities = ['A', 'A', 'B', 'C']
    numbered_entities = number_entity(entities)
    print(numbered_entities)

    boxes = torch.tensor([[10, 10, 50, 50], [60, 10, 110, 50], [10, 60, 50, 110], [60, 60, 110, 110]])

    triples, triple_names = synth_relation(entities, boxes, vocab_pred_name_to_idx)
    print(triple_names)
