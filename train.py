import sys
import os
from loader import get_loader
from micro import TRAIN, VAL, TEST
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss
from utils.tools import continue_train, get_logger, calculate_params_flops,set_seed
import torch
from train_val_epoch import train_epoch,val_epoch
import argparse

torch.cuda.set_device(0)
print("Current device ID:", torch.cuda.current_device())
set_seed(42)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    default="PH2",
    help="input datasets name including ISIC2017, ISIC2018, PH2, Kvasir, or BUSI",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default="8",
    help="input batch_size",
)
parser.add_argument(
    "--imagesize",
    type=int, 
    default=256,
    help="input image resolution.",
)
parser.add_argument(
    "--log",
    type=str,
    default="log",
    help="input log folder: ./log",
)
parser.add_argument(
    "--continues",
    type=int,
    default=0,
    help="1: continue to run; 0: don't continue to run",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints',
    help="the checkpoint path of last model: ./checkpoints",
)

def get_model():
    model=Model(input_channels=[8,16,24,32],scale_factor=[1,2,4,8])
    model = model.cuda()
    return model

def train(args):
    #init_checkpoint folder
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #logger
    logger = get_logger('train', os.path.join(os.getcwd(),args.log))
    #initialization cuda
    #get model
    model=get_model()
    #calculate parameters and flops
    calculate_params_flops(model,size=args.imagesize,logger=logger)
    #set loss function
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = 0.001,
            betas = (0.9,0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            amsgrad = False
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.00001,
            last_epoch = -1
        )
    #Do continue to run?
    if args.continues:
        model,start_epoch,min_loss,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')
    #get loader
    train_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TRAIN)
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=VAL)
    # 
    #running settings
    min_loss=0
    start_epoch=0
    end_epoch=300
    steps=0
    #start to run the model
    for epoch in range(start_epoch, end_epoch):
        torch.cuda.empty_cache()
        #train model
        steps=train_epoch(train_loader,model,criterion,optimizer,scheduler,epoch, steps,logger,save_cycles=20)
        # #validate model
        loss,miou=val_epoch(val_loader,model,criterion,logger)
        if miou>min_loss:
            print('save best.pth')
            min_loss=miou
            min_epoch=epoch
            torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, 'best.pth'))


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)