import os

import torch
import numpy as np
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import read_entity_list


class ImageDataset(Dataset):
    def __init__(self, dataset, ent_map) -> None:
        super(ImageDataset, self).__init__()
        self.image_root_path = "./images"
        self.dataset = dataset
        self.entity_map = ent_map
        self.ent_image_pairs = []
        self.img_size = (16, 16)
        self.read_images()
    
    def read_images(self):
        dirs = os.listdir(self.image_root_path)
        ent_count = 0
        for dir in dirs:
            ent = "/" + dir.replace(".", "/")
            if ent not in self.entity_map:
                continue
            ent_id = int(self.entity_map[ent])
            ent_count += 1
            print(ent_count)
            ent_path = os.path.join(self.image_root_path, dir)
            images = os.listdir(ent_path)
            image_count = 0
            for image in images:
                if image_count >= 10:
                    break
                image_count += 1
                print(ent_id, image_count)
                image_path = os.path.join(ent_path, image)
                img = Image.open(image_path)
                pixels = np.array(img.resize(self.img_size)).reshape(-1,)
                self.ent_image_pairs.append((ent_id, torch.from_numpy(pixels)))
            print(ent_id, image_count)


    def __len__(self):
        return len(self.ent_image_pairs)
    

    def __getitem__(self, index):
        return self.ent_image_pairs[index]


if __name__ == "__main__":
    data = "FB15K237"
    entity_map, _ = read_entity_list(data)
    dataset = ImageDataset(dataset=data, ent_map=entity_map)
    print(dataset.__len__())
    pkl.dump(dataset, open("./data/imagedata_small.pkl", 'wb'))
    """
    dataset = pkl.load(open("./data/imagedata_big.pkl", 'rb'))
    print(dataset.__len__())
    data_loader = DataLoader(dataset=dataset, batch_size=3)
    for (idx, batch_data) in enumerate(data_loader):
        ent_id, img = batch_data
        print(ent_id)
        print(img.shape)
        break
    """