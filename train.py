import os
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset  import ModelNet
from model import PointNet,DGCNN_Classification


def train(args:argparse.Namespace):
    seed=args.manual_seed
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        args.use_gpus=True
    
    device = torch.device("cuda" if args.use_gpus else "cpu")
    
    if device.type=="cuda":
        torch.cuda.manual_seed(seed)
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

    transform=None
    train_set=ModelNet(train=True,num_points=args.num_points,transform=transform)
    test_set=ModelNet(train=False,num_points=args.num_points,transform=transform)
    train_dataloader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    test_dataloader=DataLoader(test_set,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)

    num_classes=len(train_set.classes)
    args.num_classes=num_classes

    if args.model=="DGCNN":
        model=DGCNN_Classification(args,args.num_classes)
    else:
        raise NotImplementedError
    model.to(device)
    pytorch_total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable params: {pytorch_total_params}")

    optimizer=optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999))
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,args.num_epochs,eta_min=args.lr)




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=10,help="Batch Size")
    parser.add_argument("--manual_seed",type=int,default=9,help="Manual Seed")
    parser.add_argument("--model",type=str,default="DGCNN",help="Model architecture")
    parser.add_argument("--num_epochs",type=int,default=10,help="Number of Epochs")
    parser.add_argument("--num_workers",type=int,default=2,help="Number of Workers")
    parser.add_argument("--num_points",type=int,default=1024,help="Number of Sampling Points")
    parser.add_argument("--lr",type=float,default=0.002,help="Learning Rate")
    parser.add_argument("--k",type=int,default=20,help="Number of nearest neighbors")

    args=parser.parse_args()
    train(args)

if __name__=="__main__":
    main()