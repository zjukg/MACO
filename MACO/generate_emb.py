import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import GraphEncoder, Discriminator, Generator
from build_graph import build_kg
from utils import read_entity_list, read_image_index, valid_entity_map, generate_valid
from datetime import datetime


logging.basicConfig(filename=str(datetime.now()) + ".log", level=logging.INFO)


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K237')
    arg.add_argument('-batch_size', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=500)
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-in_dim', type=int, default=128)
    arg.add_argument('-hidden_dim', type=int, default=256)
    arg.add_argument('-out_dim', type=int, default=512)
    arg.add_argument('-num_rel', type=int, default=238)
    arg.add_argument('-img_dim', type=int, default=768)
    arg.add_argument('-noise_dim', type=int, default=128)
    arg.add_argument('-lr', type=float, default=1e-4)
    arg.add_argument('-num_ent', type=int, default=14541)
    arg.add_argument('-missing_rate', type=float, default=0.5)
    arg.add_argument('-num_gen', type=int, default=1024)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    entity_list, entity_id_list = read_entity_list(args.dataset)
    image_index, valid_entity_list = read_image_index(entity_list)
    valid_entity = valid_entity_map(entity_list, valid_entity_list, args.missing_rate)
    entity_loader = DataLoader([int(ent) for ent in entity_id_list], batch_size=args.batch_size, shuffle=True)
    graph, rels = build_kg(args.dataset)
    GNN = GraphEncoder(
        graph, 
        rels, 
        args.in_dim, 
        args.hidden_dim, 
        args.out_dim,
        args.num_ent,
        args.num_rel,
        'cuda'
    )
    GNN.load_state_dict(torch.load("./save/gnn-FB15K237-0.5-vit.pth"), strict=True)
    G = Generator(
        args.noise_dim,
        args.out_dim,
        args.img_dim
    )
    G.load_state_dict(torch.load("./save/G-FB15K237-0.5-vit.pth"), strict=True)
    D = Discriminator(
        args.img_dim,
        args.out_dim
    )
    D.load_state_dict(torch.load("./save/D-FB15K237-0.5-vit.pth"), strict=True)
    img_emb = torch.load(open("./data/{}-vit.pth".format(args.dataset), 'rb'))
    pretrained_img_embs = nn.Embedding.from_pretrained(
        torch.load(open("./data/{}-vit.pth".format(args.dataset), 'rb'))
    ).requires_grad_(False)
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Move to cuda
    GNN.to(device)
    G.to(device)
    D.to(device)
    pretrained_img_embs.to(device)
    # Model Test Part
    GNN.eval()
    D.eval()
    G.eval()
    generated_visual_emb = torch.zeros_like(pretrained_img_embs.weight.data)
    count = 0
    with torch.no_grad():
        node_emb = GNN().to(device)
        for i in range(0, args.num_ent):
            if i in valid_entity.keys():
                generated_visual_emb[i] = pretrained_img_embs.weight.data[i]
            else:
                count += 1
                random_noise = torch.randn((args.num_gen, args.noise_dim)).to(device)
                generate_embs = G(random_noise, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
                # select real embeddings from the generated embeddings
                discriminate_results = D(generate_embs, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
                real_index = torch.nonzero(discriminate_results[:, 0] < discriminate_results[:, 1])
                # logging.info(real_index)
                real_embs = torch.index_select(generate_embs, dim=0, index=real_index.reshape(-1))
                    # Generate valid embeddings
                if not torch.any(torch.isnan(real_embs)).item() and real_embs.shape[0] != 0:
                    generated_visual_emb[i] = torch.mean(real_embs, dim=0)
                    logging.info("Entity: {}, Real Embeddings: {}".format(i, real_embs.shape[0]))
                else:
                    logging.info("Generate failed for entity: {}, Set Zero".format(i))
    torch.save(img_emb, open("{}-{}-source-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    torch.save(generated_visual_emb, open("{}-{}-gen-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    logging.info("Generated visual embeddings: {}".format(count))
    random_emb = torch.randn((args.num_ent, args.img_dim))
    for i in valid_entity.keys():
        random_emb[int(i)] = img_emb[int(i)].detach()
    torch.save(random_emb, open("{}-{}-random-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    logging.info("Complete missing visual embeddings with random setting: {}".format(count))
    count = 0
    # Generate embedding for all entities.
    with torch.no_grad():
        node_emb = GNN().to(device)
        for i in range(0, args.num_ent):
            count += 1
            random_noise = torch.randn((args.num_gen, args.noise_dim)).to(device)
            generate_embs = G(random_noise, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
            # select real embeddings from the generated embeddings
            discriminate_results = D(generate_embs, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
            real_index = torch.nonzero(discriminate_results[:, 0] < discriminate_results[:, 1])
            real_embs = torch.index_select(generate_embs, dim=0, index=real_index.reshape(-1))
            generated_visual_emb[i] = torch.mean(real_embs, dim=0)
            logging.info("Entity: {}, Real Embeddings: {}".format(i, real_embs.shape[0]))
    torch.save(generated_visual_emb, open("{}-{}-allgen-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
