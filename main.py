# Copyright (c) EEEM071, University of Surrey

import datetime
import os
import os.path as osp
import sys
import time
import warnings

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src import models
from src.data_manager import ImageDataManager
from src.eval_metrics import evaluate
from src.losses import CrossEntropyLoss, TripletLoss, DeepSupervision, CenterLoss # CircleLoss, convert_label_to_similarity

from src.custom_losses import ArcLoss, ArcFaceLoss
# from src.custom_models import ResNet
from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer
from src.utils.avgmeter import AverageMeter
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from src.utils.loggers import Logger, RankLogger
from src.utils.torchtools import (
    count_num_param,
    accuracy,
    load_pretrained_weights,
    save_checkpoint,
    resume_from_checkpoint,
)
from src.utils.visualtools import visualize_ranked_results

# global variables
parser = argument_parser()
args = parser.parse_args()

# save training script (train.sh) for future reruns
import shutil
import os
def save_training_script(save_dir):
    # Source file path
    src_path = './train.sh'
    filename = os.path.basename(src_path)
    # Destination file path
    dst_path = os.path.join(save_dir, filename)
    # Copy the file
    shutil.copy(src_path, dst_path)




def main():
    global args
    if args.random_erase:
        aug = "REA+CJ+FLIP"
    else:
        aug = "None"
    experiment_name = f"Experiment_ARCH{args.arch}_OPT{args.optim}_SHEDULER{args.lr_scheduler}_BS_{args.train_batch_size}_AUG{aug}_SAMPLER{args.train_sampler}_HW{args.height}{args.width}"
    args.save_dir = osp.join(args.save_dir, experiment_name)

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False


    now = datetime.datetime.now()

    log_name = f"log_test_{now.strftime('%A, %B %d, %Y %I:%M %p')}.txt" if args.evaluate else f"log_train_{now.strftime('%A, %B %d, %Y %I:%M %p')}.txt"

    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print(f"==========\nArgs:{args}\n==========")

    save_training_script(args.save_dir)
    print(f"==========training file saved to  {args.save_dir}==========")
    
    writer = SummaryWriter(log_dir=args.save_dir)

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print(f"Initializing model: {args.arch}")
    model = models.init_model(
        name=args.arch,
        num_classes=dm.num_train_pids,
        loss={"xent", "htri"},
        pretrained=not args.no_pretrained,
        use_gpu=use_gpu,
    )
    print("Model size: {:.3f} M".format(count_num_param(model)))
    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model
    # model
    criterion_xent = CrossEntropyLoss(
        num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth
    )
    criterion_htri = TripletLoss(margin=args.margin)
    # criterion_htri = CrossEntropyLoss(num_classes=576)
    # criterion_htri = CenterLoss(num_classes=dm.num_train_pids, feat_dim=2048)
    # criterion_htri = ArcLoss() #this worked
    # criterion_htri = ArcFaceLoss(num_classes=576, embedding_size=512, margin=0.5, scale=32) 
    # CenterLoss(num_classes=576, feat_dim=128) #CircleLoss(m=0.25, gamma=256) #CenterLoss(576, 512) #CenterLoss(feat_dim=512) #nn.CrossEntropyLoss() #ArcLoss()
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )
        

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print(f"Evaluating {name} ...")
            queryloader = testloader_dict[name]["query"]
            galleryloader = testloader_dict[name]["gallery"]
            distmat = test(
                model, queryloader, galleryloader, use_gpu, return_distmat=True
            )

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat,
                    dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, "ranked_results", name),
                    topk=20,
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print("=> Start training")
    """
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    """
    for epoch in range(args.start_epoch, args.max_epoch):
        lr = scheduler.get_last_lr()[0]
        # writer.add_hparams({'lr': lr, 'bsize': args.train_batch_size}, {})
        train(
            epoch,
            model,
            criterion_xent,
            criterion_htri,
            optimizer,
            trainloader,
            use_gpu,
            writer
            
        )

        scheduler.step()
        
        if (
            (epoch + 1) > args.start_eval
            and args.eval_freq > 0
            and (epoch + 1) % args.eval_freq == 0
            or (epoch + 1) == args.max_epoch
        ):
            print("=> Test")

            for name in args.target_names:
                print(f"Evaluating {name} ...")
                queryloader = testloader_dict[name]["query"]
                galleryloader = testloader_dict[name]["gallery"]
                rank1, mAP = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)
                writer.add_scalar('Test/rank1', rank1, epoch+1)
                writer.add_scalar('Test/mAP', mAP, epoch+1)

            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "rank1": rank1,
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "optimizer": optimizer.state_dict(),
                },
                args.save_dir,
            )

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")
    ranklogger.show_summary()


def train(
    epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, writer
):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    for p in model.parameters():
        p.requires_grad = True  # open all layers

    end = time.time()

    NUM_ACCUMULATION_STEPS = 3
    total_loss = 0
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if ( (batch_idx+1) % NUM_ACCUMULATION_STEPS!=0) or ((batch_idx+1) == len(trainloader)):
        #     total_loss += loss
        # if ( (batch_idx+1) % NUM_ACCUMULATION_STEPS==0) or ((batch_idx+1) == len(trainloader)):
            
        #     # gradient accumalation
        #     total_loss += loss
        #     total_loss = total_loss / NUM_ACCUMULATION_STEPS

        #     # Backward pass
        #     loss.backward()
        #     # parameters updated
        #     optimizer.step()
        #     # Reset gradient tensors
        #     optimizer.zero_grad()
        #     total_loss = 0

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                "Xent {xent.val:.4f} ({xent.avg:.4f})\t"
                "Htri {htri.val:.4f} ({htri.avg:.4f})\t"
                "Acc {acc.val:.2f} ({acc.avg:.2f})\t".format(
                    epoch + 1,
                    batch_idx + 1,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    xent=xent_losses,
                    htri=htri_losses,
                    acc=accs,
                )
            )

        # Log to TensorBoard
        iteration = epoch * len(trainloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), iteration)
        writer.add_scalar('Train/Xent_Loss', xent_loss.item(), iteration)
        writer.add_scalar('Train/Htri_Loss', htri_loss.item(), iteration)
        writer.add_scalar('Train/Accuracy', accs.avg, iteration)

        end = time.time()

    
def test(
    model,
    queryloader,
    galleryloader,
    use_gpu,
    ranks=[1, 5, 10, 20],
    return_distmat=False,
):
    batch_time = AverageMeter()

    model.eval()
    # print(model)
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time() 
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            # print("features--------->", features.shape)

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print(
            "Extracted features for query set, obtained {}-by-{} matrix".format(
                qf.size(0), qf.size(1)
            )
        )

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print(
            "Extracted features for gallery set, obtained {}-by-{} matrix".format(
                gf.size(0), gf.size(1)
            )
        )

    print(
        f"=> BatchTime(s)/BatchSize(img): {batch_time.avg:.3f}/{args.test_batch_size}"
    )

    m, n = qf.size(0), gf.size(0)
    # Euclidean distance
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    # # Cosine
    # # qf_norm = qf / qf.norm(dim=1, keepdim=True)
    # # gf_norm = gf / gf.norm(dim=1, keepdim=True)
    # # distmat = torch.mm(qf_norm, gf_norm.t()).numpy()

    # import torch.nn.functional as F
    # qf_norm = F.normalize(qf, p=1, dim=1)
    # gf_norm = F.normalize(gf, p=1, dim=1)
    # distmat = 1 - torch.mm(qf_norm, gf_norm.t()).numpy()


    print("Computing CMC and mAP")
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0], mAP


if __name__ == "__main__":
    main()
