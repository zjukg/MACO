import torch
import mmns
from mmns.config import Trainer, Tester
from mmns.module.model import TBKGC
from mmns.module.loss import MarginLoss, SoftplusLoss, SigmoidLoss
from mmns.module.strategy import NegativeSampling
from mmns.data import TrainDataLoader, TestDataLoader
from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    # set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./visual/' + args.dataset + '-{}-{}-{}.pth'.format(args.missing_rate, args.visual, args.postfix))
    transe = TBKGC(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        p_norm=1,
        norm_flag=True,
        img_dim=args.img_dim,
        img_emb=img_emb,
        test_mode='lp'
    )
    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=args.margin, adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        neg_mode='normal'
    )
    trainer_model1 = Trainer(
        model=model, 
        data_loader=train_dataloader,
        train_times=args.epoch, 
        alpha=args.learning_rate, 
        use_gpu=True,
        opt_method='Adam', 
        train_mode='normal'
    )
        
    trainer_model1.run()
    transe.save_checkpoint(args.save) 
    # test the model
    transe.load_checkpoint(args.save)
    tester_model1 = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    tester_model1.run_link_prediction(type_constrain=False)


   

