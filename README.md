# MACO: A Modality Adversarial and Contrastive Framework for Modality-missing Multi-modal Knowledge Graph Completion
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/MACO/main/LICENSE)
[![AAAI](https://img.shields.io/badge/NLPCC'23-brightgreen)](http://tcci.ccf.org.cn/conference/2023/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
 - [*MACO: A Modality Adversarial and Contrastive Framework for Modality-missing Multi-modal Knowledge Graph Completion*](https://arxiv.org/abs/2308.06696)

> Recent years have seen significant advancements in multi-modal knowledge graph completion (MMKGC). MMKGC enhances knowledge graph completion (KGC) by integrating multi-modal entity information, thereby facilitating the discovery of unobserved triples in the large-scale knowledge graphs (KGs). Nevertheless, existing methods emphasize the design of elegant KGC models to facilitate modality interaction, neglecting the real-life problem of missing modalities in KGs. The missing modality information impedes modal interaction, consequently undermining the model's performance. In this paper, we propose a modality adversarial and contrastive framework (MACO) to solve the modality-missing problem in MMKGC. MACO trains a generator and discriminator adversarially to generate missing modality features that can be incorporated into the MMKGC model. Meanwhile, we design a cross-modal contrastive loss to improve the performance of the generator. Experiments on public benchmarks with further explorations demonstrate that MACO could achieve state-of-the-art results and serve as a versatile framework to bolster various MMKGC models.


## 🌈 Model Architecture
![Model_architecture](figure/model.png)

## 🔬 Dependencies
- python 3
- torch >= 1.8.0
- numpy
- dgl-cu111 == 0.9.1
- All experiments are performed with one A100-40G GPU.

## 📕 Code Path
- `MACO/` pretrain with MACO to complete the modality information. We have prepared the FB15K-237 dataset and the visual embeddings extracted with Vision Transformer (ViT). You should first download it from [this link](https://drive.google.com/file/d/1XN7e1_6ERZWPrg3f0gis5ZPb3tpKEWmv/view?usp=drive_link).
- `MMKGC/` run multi-modal KGC to evaluate the quality of generated visual features.

- run MACO
```shell
cd MACO/
# download the FB15K-237 visual embeddings and put it in data/
# run the training code
python main.py
```

- run MMKGC
```shell
cd MMKGC/
# Put the generated visual embedding in MACO to visual/ 
mv ../MACO/EMBEDDING_NAME visual/

# run the MMKGC model
DATA=FB15K237
NUM_BATCH=1024
KERNEL=transe
MARGIN=6
LR=2e-5
NEG_NUM=32
VISUAL=random-vit
MS=0.6
POSTFIX=2.0-0.01

CUDA_VISIBLE_DEVICES=0 nohup python run_ikrl.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=1000 \
  -dim=128 \
  -save=./checkpoint/ikrl/$DATA-New-$KERNEL-$NUM_BATCH-$MARGIN-$LR-$VISUAL-large-$MS \
  -img_grad=False \
  -img_dim=768 \
  -neg_num=$NEG_NUM \
  -kernel=$KERNEL \
  -visual=$VISUAL \
  -learning_rate=$LR \
  -postfix=$POSTFIX \
  -missing_rate=$MS > ./log/IKRL$MS-$DATA-$KERNEL-4score-$MARGIN-$VISUAL-$MS-$POSTFIX.txt &
```
This is a simple demo to run IKRL model. The scripts to train other models (TBKGC, RSME) can be found in `MMKGC/scripts/`.

## 💡 Related Works
There are also some other works about multi-modal knowledge graphs from ZJUKG team. If you are interest in multi-modal knowledge graphs, you could have a look at them:

### Multi-modal Entity Alignment
- (ACM MM 2023) [MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid](https://github.com/zjukg/MEAformer)
- (ISWC 2023) [Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment](https://github.com/zjukg/UMAEA)

### Multi-modal Knowledge Graph Completion
- (SIGIR 2022) [Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion](https://github.com/zjukg/MKGformer)
- (IJCNN 2023) [Modality-Aware Negative Sampling for Multi-modal Knowledge Graph Embedding](https://github.com/zjukg/MANS)

### Knowledge Graph with Large Language Models
- [KG-LLM-Papers](https://github.com/zjukg/KG-LLM-Papers)

### Open-source Tools
- [NeuralKG](https://github.com/zjukg/NeuralKG)

## 🤝 Cite:
Please condiser citing this paper if you use the code from our work.
Thanks a lot :)

```bigquery
@article{zhang2023maco,
  title={MACO: A Modality Adversarial and Contrastive Framework for Modality-missing Multi-modal Knowledge Graph Completion},
  author={Zhang, Yichi and Chen, Zhuo and Zhang, Wen},
  journal={arXiv preprint arXiv:2308.06696},
  year={2023}
}
```
