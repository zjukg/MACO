import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model


class MMDisMult(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=None, epsilon=None, img_emb=None, img_dim=4096, test_mode='lp', beta=None):
        super(MMDisMult, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.img_dim = img_dim
        self.test_mode = test_mode
        self.img_proj = nn.Linear(self.img_dim, self.dim)
        self.img_embeddings = img_emb
        self.beta = beta

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

    def _calc(self, h, t, r, mode):
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h * (r * t)
        else:
            score = (h * r) * t
        score = torch.sum(score, -1).flatten()
        return score

    def forward(self, data, batch_size, neg_mode='normal', neg_num=1):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        if neg_mode == "twin":
            h_unimodal, t_unimodal = batch_h, batch_t
            # 对多模态分数部分的structure embedding进行复原，变回原样本
            h_multimodal_ent = torch.tensor(batch_h[:batch_size].clone()).repeat(neg_num + 1)
            t_multimodal_ent = torch.tensor(batch_t[:batch_size].clone()).repeat(neg_num + 1)
            mode = data['mode']
            # unimodal 的部分一切正常
            h_uni = self.ent_embeddings(h_unimodal)
            t_uni = self.ent_embeddings(t_unimodal)
            r = self.rel_embeddings(batch_r)
            h_img_uni_emb = self.img_proj(self.img_embeddings(h_unimodal))
            t_img_uni_emb = self.img_proj(self.img_embeddings(t_unimodal))
            # multimodal的部分只采用img的负样本，ent的负样本恢复成原样
            h_multi = self.ent_embeddings(h_multimodal_ent)
            t_multi = self.ent_embeddings(t_multimodal_ent)
            score = (
                    self._calc(h_uni, t_uni, r, mode)
                    + self._calc(h_img_uni_emb, t_img_uni_emb, r, mode)
                    + self._calc(h_img_uni_emb, t_multi, r, mode)
                    + self._calc(h_multi, t_img_uni_emb, r, mode)
            ) / 4
            return score
        if neg_mode == "adaptive":
            # adaptive negative sampling selector
            mode = data['mode']
            h_img_neg, t_img_neg = batch_h[batch_size:].detach(), batch_t[batch_size:].detach()
            r_neg = batch_r[batch_size:].detach()
            h_neg = self.ent_embeddings(h_img_neg)
            t_neg = self.ent_embeddings(t_img_neg)
            r_neg = self.rel_embeddings(r_neg)
            h_img_ent_emb = self.img_proj(self.img_embeddings(h_img_neg))
            t_img_ent_emb = self.img_proj(self.img_embeddings(t_img_neg))
            # 正常模态的分数
            neg_score1 = self._calc(h_neg, t_neg, r_neg, mode) + self._calc(h_img_ent_emb, t_img_ent_emb, r_neg, mode)
            # 跨模态的分数
            neg_score2 = (
                self._calc(h_img_ent_emb, t_neg, r_neg, mode)
                    + self._calc(h_neg, t_img_ent_emb, r_neg, mode)
            )
            selector = (neg_score2 < neg_score1).int()
            # 改了，注意
            img_idx = torch.nonzero(selector).reshape((-1, ))
            p = img_idx.shape[0] / (batch_size * neg_num)
            self.log_file.write('{}\n'.format(p))
            num = int(neg_num * p * batch_size)
            # 这里也改了，目前是2score2-new的形态，原先index可能要反过来
            h_ent, h_img, t_ent, t_img = batch_h.clone(), batch_h.clone(), batch_t.clone(), batch_t.clone()
            h_ent[batch_size: batch_size + num] = batch_h[0: num].clone()
            t_ent[batch_size: batch_size + num] = batch_t[0: num].clone()
        else:
            num = int(neg_num * self.beta * batch_size) if batch_size != None else 0
            h_ent, h_img, t_ent, t_img = None, None, None, None
            
            if neg_mode == "normal":
                h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
            else:
                # 只对图像进行负采样，原本的batch_h/batch_t中包含的就是负样本
                if neg_mode == "img":
                    h_img, t_img = batch_h, batch_t
                    h_ent = torch.tensor(batch_h[:batch_size]).repeat(neg_num + 1)
                    t_ent = torch.tensor(batch_t[:batch_size]).repeat(neg_num + 1)
                elif neg_mode == "hybrid":
                    h_ent, h_img, t_ent, t_img = batch_h.clone(), batch_h.clone(), batch_t.clone(), batch_t.clone()
                    h_ent[batch_size: batch_size + num] = batch_h[0: num].clone()
                    t_ent[batch_size: batch_size + num] = batch_t[0: num].clone()
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        t = self.ent_embeddings(t_ent)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(h_img))
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        score = (
            self._calc(h, t, r, mode)
            + self._calc(h_img_emb, t_img_emb, r, mode)
            + self._calc(h_img_emb, t, r, mode)
            + self._calc(h, t_img_emb, r, mode)
        ) / 4
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def l3_regularization(self):
        return (self.ent_embeddings.weight.norm(p=3)**3 + self.rel_embeddings.weight.norm(p=3)**3)

    def cross_modal_score_ent2img(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        r = self.rel_embeddings(batch_r)
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        # 跨模态链接预测的过程中，只考虑h+r和尾部图像的匹配度
        score = self._calc(h, t_img_emb, r, mode)
        return score

    def predict(self, data):
        if self.test_mode == 'cmlp':
            score = -self.cross_modal_score_ent2img(data)
        else:
            score = -self.forward(data, batch_size=None, neg_mode='normal')
        return score.cpu().data.numpy()

    def set_test_mode(self, new_mode):
        self.test_mode = new_mode
