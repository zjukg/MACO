import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from build_graph import build_kg
from model import Discriminator, Generator, GraphEncoder
from utils import (generate_valid, read_entity_list, read_image_index,
                   valid_entity_map)
from loss import ContrastiveLoss

logging.basicConfig(filename=str(datetime.now())+".log", level=logging.INFO)

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K237')
    arg.add_argument('-batch_size', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=500)
    arg.add_argument('-seed', type=int, default=42) # 42
    arg.add_argument('-in_dim', type=int, default=128)
    arg.add_argument('-hidden_dim', type=int, default=256)
    arg.add_argument('-out_dim', type=int, default=768)
    arg.add_argument('-num_rel', type=int, default=238)
    arg.add_argument('-img_dim', type=int, default=768)
    arg.add_argument('-noise_dim', type=int, default=128)
    arg.add_argument('-lr', type=float, default=1e-4)
    arg.add_argument('-num_ent', type=int, default=14541)
    arg.add_argument('-missing_rate', type=float, default=0.2)
    arg.add_argument('-num_gen', type=int, default=512)
    arg.add_argument('-temp', type=float, default=0.5)
    arg.add_argument('-lamda', type=float, default=0.01)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    logging.info(args)
    entity_list, entity_id_list = read_entity_list(args.dataset)
    image_index, valid_entity_list = read_image_index(entity_list)
    valid_entity = valid_entity_map(entity_list, valid_entity_list, args.missing_rate)
    print(valid_entity_list)
    logging.info("The number of entities which have visual feature: {}".format(len(valid_entity.keys())))
    logging.info(valid_entity.keys())
    entity_loader = DataLoader([int(ent) for ent in entity_id_list], batch_size=args.batch_size, shuffle=True)
    graph, rels = build_kg(args.dataset)
    print(graph)
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
    G = Generator(
        args.noise_dim,
        args.out_dim,
        args.img_dim
    )
    D = Discriminator(
        args.img_dim,
        args.out_dim
    )
    GNN_solver = optim.Adam(GNN.parameters(), lr=args.lr)
    G_solver = optim.Adam(G.parameters(), lr=args.lr)
    D_solver = optim.Adam(D.parameters(), lr=args.lr)
    G_loss = nn.CrossEntropyLoss()
    D_loss = nn.CrossEntropyLoss()
    contrast_loss = ContrastiveLoss(temp=args.temp)
    img_emb = torch.load(open("./data/{}-vit.pth".format(args.dataset), 'rb'))
    unseen_entitiy = 0
    for i in range(args.num_ent):
        if i not in valid_entity:
            print("set unseen embedding zero: {}".format(i))
            unseen_entitiy += 1
            img_emb[i] = torch.zeros((1, args.img_dim))
    print("Unseen entitiy number: {}".format(unseen_entitiy))
    pretrained_img_embs = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Move to cuda
    GNN.to(device)
    G.to(device)
    D.to(device)
    pretrained_img_embs.to(device)
    # Training 
    GNN.train()
    G.train()
    D.train()
    training_range = tqdm(range(1, args.epoch + 1))
    training_loss_d = []
    training_loss_g = []
    for epoch in training_range:
        epoch_loss_d = 0
        epoch_loss_g = 0
        for (idx, batch_data) in enumerate(entity_loader):
            # training D
            D_solver.zero_grad()
            GNN_solver.zero_grad()
            # GNN message passing
            node_emb = GNN().to(device)
            # Real
            batch_valid_entity = torch.LongTensor(generate_valid(batch_data, valid_entity)).to(device)
            batch_real_node_emb = torch.index_select(node_emb, dim=0, index=batch_valid_entity)
            batch_real_img_emb = pretrained_img_embs(batch_valid_entity)
            real_out = D(batch_real_img_emb, batch_real_node_emb)
            real_loss = D_loss(real_out, torch.ones(real_out.shape[0]).long().to(device))
            # Fake
            random_noise = torch.randn((batch_data.shape[0], args.noise_dim)).to(device).requires_grad_(False)
            batch_node_emb = torch.index_select(node_emb, dim=0, index=batch_data.to(device))
            fake_image_emb = G(random_noise, batch_node_emb)
            fake_out = D(fake_image_emb, batch_node_emb)
            fake_loss = D_loss(fake_out, torch.zeros(fake_out.shape[0]).long().to(device))
            batch_loss_d = real_loss + fake_loss
            # update D and GNN
            batch_loss_d.backward()
            GNN_solver.step()
            D_solver.step()
            epoch_loss_d += batch_loss_d.item()
            # training G
            G_solver.zero_grad()
            GNN_solver.zero_grad()
            node_emb = GNN().to(device)
            random_noise = torch.randn((batch_data.shape[0], args.noise_dim)).to(device).requires_grad_(False)
            batch_node_emb = torch.index_select(node_emb, dim=0, index=batch_data.to(device))
            fake_image_emb = G(random_noise, batch_node_emb)
            fake_out = D(fake_image_emb, batch_node_emb)
            # push G to generate true features
            batch_loss_g = G_loss(fake_out, torch.ones(fake_out.shape[0]).long().to(device)) + args.lamda * contrast_loss(batch_node_emb, fake_image_emb)
            # update G and GNN
            batch_loss_g.backward()
            GNN_solver.step()
            G_solver.step()
            epoch_loss_g += batch_loss_g.item()
            # Remove from GPU
            # batch_real_node_emb.cpu()
            # batch_node_emb.cpu()
            # fake_out.cpu()
            # fake_image_emb.cpu()
            training_range.set_description("Epoch: {}, Batch: {}, D:{}, G:{}".format(epoch, idx, batch_loss_d, batch_loss_g))
        logging.info("Epoch: {}, D loss: {}, G loss:{}".format(epoch, epoch_loss_d, epoch_loss_g))
        training_loss_d.append(epoch_loss_d)
        training_loss_g.append(epoch_loss_g)
    torch.save(GNN.state_dict(), open("./save/gnn-{}-{}-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    torch.save(D.state_dict(), open("./save/D-{}-{}-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    torch.save(G.state_dict(), open("./save/G-{}-{}-vit.pth".format(args.dataset, args.missing_rate), 'wb'))
    # 生成部分的代码
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
                while True:
                    count += 1
                    random_noise = torch.randn((args.num_gen, args.noise_dim)).to(device)
                    generate_embs = G(random_noise, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
                    # select real embeddings from the generated embeddings
                    discriminate_results = D(generate_embs, node_emb[i].expand(args.num_gen, node_emb.shape[1]))
                    real_index = torch.nonzero(discriminate_results[:, 0] < discriminate_results[:, 1])
                    logging.info(real_index)
                    real_embs = torch.index_select(generate_embs, dim=0, index=real_index.reshape(-1))
                    # Generate valid embeddings
                    if real_embs.shape[0] != 0:
                        generated_visual_emb[i] = torch.mean(real_embs, dim=0)
                        logging.info("Entity: {}, Real Embeddings: {}".format(i, real_embs.shape[0]))
                        break
                    else:
                        logging.info("Generate failed for entity: {}, Set Zero".format(i))
                        break
    torch.save(img_emb, open("./embs/{}-{}-source-vit-{}-{}.pth".format(args.dataset, args.missing_rate, args.temp, args.lamda), 'wb'))
    torch.save(generated_visual_emb, open("./embs/{}-{}-gen-vit-{}-{}.pth".format(args.dataset, args.missing_rate, args.temp, args.lamda), 'wb'))
    logging.info("Generated visual embeddings: {}".format(count))
    random_emb = torch.randn((args.num_ent, args.img_dim))
    for i in valid_entity.keys():
        random_emb[int(i)] = img_emb[int(i)].detach()
    torch.save(random_emb, open("./embs/{}-{}-random-vit-{}-{}.pth".format(args.dataset, args.missing_rate, args.temp, args.lamda), 'wb'))
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
            if real_embs.shape[0] != 0:
                generated_visual_emb[i] = torch.mean(real_embs, dim=0)
                logging.info("Entity: {}, Real Embeddings: {}".format(i, real_embs.shape[0]))
    torch.save(generated_visual_emb, open("./embs/{}-{}-allgen-vit-{}-{}.pth".format(args.dataset, args.missing_rate, args.temp, args.lamda), 'wb'))
