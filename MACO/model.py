import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

class GraphEncoder(nn.Module):
    def __init__(self, graph, rels, in_dim, hidden_dim, out_dim, num_node, num_rel, device='cuda'):
        super(GraphEncoder, self).__init__()
        self.graph = graph.to(device)
        self.relations = torch.tensor(rels).to(device)
        self.node_emb = torch.randn((num_node, in_dim)).to(device)
        self.conv1 = RelGraphConv(in_dim, hidden_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.Tanh())
        self.conv2 = RelGraphConv(hidden_dim, out_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.ReLU())
        nn.init.xavier_uniform_(self.node_emb)
    
    def forward(self):
        # print(self.graph.device, self.node_emb.device, self.relations.device)
        h = self.conv1(self.graph, self.node_emb, etypes=self.relations)
        out = self.conv2(self.graph, h, etypes=self.relations)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_dim, node_dim):
        super(Discriminator, self).__init__()
        self.img_proj_dim = 256
        self.num_classes = 2
        self.linear_proj1 = nn.Linear(img_dim, self.img_proj_dim)
        self.activate_func = nn.LeakyReLU()
        self.linear_proj2 = nn.Linear(self.img_proj_dim + node_dim, self.num_classes)

    def forward(self, batch_img_emb, batch_node_emb):
        batch_img_proj = self.linear_proj1(batch_img_emb)
        batch_img_proj = self.activate_func(batch_img_proj)
        batch_emb = torch.cat((batch_img_proj, batch_node_emb), dim=-1)
        out = self.linear_proj2(batch_emb)
        return out


class Generator(nn.Module):
    def __init__(self, noise_dim, node_dim, img_dim):
        super(Generator, self).__init__()
        self.proj_dim = 1024
        self.drop_p = 0.5
        assert node_dim == img_dim
        self.generator_model = nn.Sequential(
            # nn.Linear(noise_dim + node_dim, self.proj_dim),
            nn.Linear(noise_dim, self.proj_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(self.proj_dim, img_dim)
            # nn.LeakyReLU() # 仅在生成图片的时候使用
        )


    def forward(self, random_noise, batch_node_emb):
        out = self.generator_model(random_noise)
        return out
