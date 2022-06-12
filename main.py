import argparse
from distutils.command.build import build
import datetime
import torch
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from datasets import build_dataset
from model import build_model
import time 
from engine import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('AI_Final_Project_Script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # Model parameters
    parser.add_argument('--model', default='vit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-type', default='transformer', type=str, help = "resnet or transformer")
    # optimizer option

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Dataset parameters
    parser.add_argument('--data-path', default='/home/shadowpa0327/AFAD-Full', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)


    return parser




def main(args):
    print(args)
    seed = args.seed
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    ##### dataset part #####
    dataset_train = build_dataset(args, is_train=True)
    dataset_val   = build_dataset(args, is_train=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # models part
    model = build_model(model_name=args.model, model_type=args.model_type)
    model = model.to(device)

    # optimizer part 
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy, min_loss = 0.0, 1e9

    # load from check point 
    if args.resume:
        print(f"load from checkpoint:{args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        acc, loss = evaluate(data_loader_val, model, device)
        print(f"Validation result on {len(dataset_val)} images | Accuracy : {max_accuracy:.2f}%, MAE loss: {loss}")
        return

    # main training loop
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch)

        if args.output_dir:
            checkpoint_paths = output_dir / 'checkpoint.pth'  
            torch.save({
                    'model': model.state_dict(),
                    'args' : args,
                    'epoch' : epoch
                }, checkpoint_paths
            )


        acc, loss = evaluate(data_loader_val, model, device)

        if max_accuracy < acc : 
            if args.output_dir:
                checkpoint_paths = output_dir / 'best_gender.pth'  
                torch.save({
                        'model': model.state_dict(),
                        'args' : args,
                        'epoch' : epoch
                    }, checkpoint_paths
                )
            
            max_accuracy = acc

        if min_loss > loss : 
            if args.output_dir:
                checkpoint_paths = output_dir / 'best_age.pth'  
                torch.save({
                        'model': model.state_dict(),
                        'args' : args,
                        'epoch' : epoch
                    }, checkpoint_paths
                )
            
            min_loss = loss

        print(f"Max accuracy: {max_accuracy:.2f}%, Min MAE loss: {loss:.2f}")
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(f"Epoch:{epoch}/{args.epochs} | Training : gender_accuracy {train_acc:.2f} | age_mae_loss {train_loss:.2f}  | Testing: gender_accuracy {acc:.2f} | age_mae_loss {loss:.2f}\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    

if __name__ == '__main__':
    config=[
        '--batch-size' , '64',
        '--data-path', '/home/shadowpa0327/AFAD-Full',
        '--num_workers' , '8' ,
        '--epochs' , '200',
        '--lr', '1e-5',
        '--weight-decay', '0.05',
        '--output_dir', 'Vit_small',
        '--model', 'vit_small_patch16_224',
        '--model-type', 'transformer'
        #'--eval'
    ]
    parser = argparse.ArgumentParser('AI_Final_Proj Hao Chun and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(args=config)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)