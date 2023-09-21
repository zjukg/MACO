# import h5py
import torch
import numpy as np
import random


def read_entity_list(dataset):
    path = "./data/{}/entity2id.txt".format(dataset)
    file = open(path, 'r')
    ent_map = {}
    id_list = []
    for line in file.readlines()[1:]:
        ent, id = line[:-1].split('\t')
        ent_map[ent] = id
        id_list.append(id)
    return ent_map, id_list


def read_image_index(ent_map=None):
    path = "./data/image.txt"
    file = open(path, 'r')
    image_index = {}
    entity_has_index = []
    for line in file.readlines():
        ent, id = line[:-1].split('\t')
        if ent in ent_map.keys():
            image_index[ent] = id
            entity_has_index.append(ent)
    return image_index, entity_has_index


def valid_entity_map(entity_map, entity_has_index, miss_rate):
    valid_entity = {}
    select_entity_has_index = random.sample(entity_has_index, int(miss_rate * len(entity_map))) if miss_rate < 1.0 else entity_has_index
    for ent in select_entity_has_index:
        valid_entity[int(entity_map[ent])] = 1
    return valid_entity

def generate_valid(batch_entity, valid_entity):
    batch_valid_entity = []
    for ent in batch_entity:
        if ent.item() in valid_entity.keys():
            batch_valid_entity.append(ent)
    return batch_valid_entity


def read_h5_pretrain(entity_map, image_index):
    img_embeddings = torch.zeros((len(entity_map), 4096))
    file = "./data/FB15K_ImageData.h5"
    f = dict(h5py.File(file, 'r'))
    for ent in image_index.keys():
        img = image_index[ent]
        ent_id = entity_map[ent]
        vgg_feature = torch.from_numpy(np.array(f[img]))
        img_embeddings[int(ent_id)] = vgg_feature
        print(int(ent_id), vgg_feature)
    torch.save(img_embeddings, 'visual.pt')
    return img_embeddings


if __name__ == "__main__":
    pass
    