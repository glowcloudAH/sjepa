# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import wandb
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.ukbb import make_ukbb

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms


# --
log_timings = True
log_freq = 20
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    rescale_sigma = args['data']['rescale_sigma']
    ftsurrogate = args['data']['ftsurrogate']
    jitter = args['data']['jitter']
    spec_augment = args['data']['spec_augment']
    time_flip = args['data']['time_flip']
    sign_flip = args['data']['sign_flip']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['data_path']
    val_folder = args['data']['val_path']
    downstream_train_path = args['data']['downstream_train_path']
    downstream_val_path = args['data']['downstream_val_path']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    #folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    
    folder = wandb.run.dir

    dump = os.path.join(folder, 'params-ijepa.yaml')
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_resizing=crop_size,
        ftsurrogate=ftsurrogate,
        jitter=jitter,
        rescale_sigma=rescale_sigma,
        time_flip=time_flip,
        sign_flip=sign_flip,
        spec_augment = spec_augment
        )

    ipe = 1
    # -- init data-loaders/samplers
    if image_folder != "None":
        _, unsupervised_loader, unsupervised_sampler  = make_ukbb(#, unsupervised_sampler = make_mimic(
                transform=None,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                data_file=image_folder,
                copy_data=copy_data,
                drop_last=True)
        ipe = len(unsupervised_loader)

    if val_folder != "None":
        _, val_loader,_ = make_ukbb(
                transform=None,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                data_file=val_folder,
                copy_data=copy_data,
                drop_last=True
        )

    if downstream_train_path != "None":
        _, downstream_train_loader,_ = make_ukbb(
                transform=None,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                data_file=downstream_train_path,
                copy_data=copy_data,
                drop_last=True
        )
        _, downstream_val_loader,_ = make_ukbb(
                transform=None,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                data_file=downstream_val_path,
                copy_data=copy_data,
                drop_last=True
        )
    

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    if dist.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        ctx_enc_meter = AverageMeter()
        target_enc_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        val_predict_meter = AverageMeter()
        val_maskB_meter = AverageMeter()
        val_maskA_meter = AverageMeter()

        if image_folder != "None":
            for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
                
                def load_imgs():
                    # -- unsupervised imgs
                    imgs = udata[0].to(device, non_blocking=True)
                    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                    return (imgs, masks_1, masks_2)
                imgs, masks_enc, masks_pred = load_imgs()
                maskA_meter.update(len(masks_enc[0][0]))
                maskB_meter.update(len(masks_pred[0][0]))
                
                
                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                    # --

                    def forward_target():
                        with torch.no_grad():
                            h = target_encoder(imgs)
                            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                            B = len(h)
                            # -- create targets (masked regions of h)
                            h = apply_masks(h, masks_pred)
                            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                            target_enc_meter.update(h.mean())
                            return h

                    def forward_context():
                        z = encoder(imgs, masks_enc)
                        ctx_enc_meter.update(z.mean())
                        z = predictor(z, masks_enc, masks_pred)
                        return z

                    def loss_fn(z, h):
                        loss = F.smooth_l1_loss(z, h)
                        loss = AllReduce.apply(loss)
                        return loss

                    # Step 1. Forward
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)

                    #  Step 2. Backward & step
                    if use_bfloat16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    grad_stats = grad_logger(encoder.named_parameters())
                    optimizer.zero_grad()

                    # Step 3. momentum update of target encoder
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                    return (float(loss), _new_lr, _new_wd, grad_stats)
                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
                loss_meter.update(loss)
                time_meter.update(etime)

                # -- Logging
                def log_stats():
                    csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                    if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                        logger.info('[%d, %5d] loss: %.3f '
                                    'masks: %.1f %.1f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2e] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                    loss_meter.avg,
                                    maskA_meter.avg,
                                    maskB_meter.avg,
                                    _new_wd,
                                    _new_lr,
                                    torch.cuda.max_memory_allocated() / 1024.**2,
                                    time_meter.avg))

                        if grad_stats is not None:
                            logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                        % (epoch + 1, itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max))
                        
                        wandb.log({"loss":loss_meter.avg, 
                                    "masksA": maskA_meter.val, 
                                    "maskB": maskB_meter.val,
                                    "wd":_new_wd, 
                                    "lr": _new_lr,
                                    "context encoding avg": ctx_enc_meter.avg,
                                    "target encoding avg": target_enc_meter.avg,
                                    "context encoding min": ctx_enc_meter.min,
                                    "target encoding min": target_enc_meter.min,
                                    "context encoding max": ctx_enc_meter.max,
                                    "target encoding max": target_enc_meter.max,
                                    "grad_stats_first": grad_stats.first_layer,
                                    "grad_stats_last": grad_stats.last_layer,
                                    "grad_stats min": grad_stats.min,
                                    "grad_stats max": grad_stats.max})


                log_stats()

                assert not np.isnan(loss), 'loss is nan'

        
        
        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
        
        if val_folder != "None":
            encoder.eval()
            target_encoder.eval()
            predictor.eval()
            for itr, (udata, masks_enc, masks_pred) in enumerate(val_loader):
                

                def load_imgs():
                    # -- unsupervised imgs
                    imgs = udata[0].to(device, non_blocking=True)
                    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                    return (imgs, masks_1, masks_2)
                imgs, masks_enc, masks_pred = load_imgs()
                val_maskA_meter.update(len(masks_enc[0][0]))
                val_maskB_meter.update(len(masks_pred[0][0]))
                
                
                def val_step():
                    def forward_target():
                        with torch.no_grad():
                            h = target_encoder(imgs)
                            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                            B = len(h)
                            # -- create targets (masked regions of h)
                            h = apply_masks(h, masks_pred)
                            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                            return h

                    def forward_context():
                        with torch.no_grad():
                            z = encoder(imgs, masks_enc)
                            z = predictor(z, masks_enc, masks_pred)
                            return z

                    def loss_fn(z, h):
                        loss = F.smooth_l1_loss(z, h)
                        loss = AllReduce.apply(loss)
                        return loss

                    # Step 1. Forward
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)


                    return (float(loss), grad_stats)
                (loss, grad_stats), etime = gpu_timer(val_step)
                val_loss_meter.update(loss)

                assert not np.isnan(loss), 'loss is nan'

            wandb.log({"val_loss":val_loss_meter.avg, "val_masksA": val_maskA_meter.avg, 
                    "val_maskB": val_maskB_meter.avg})
            logger.info('avg. val loss %.3f' % val_loss_meter.avg)
            
            encoder.train()
            target_encoder.train()
            predictor.train()
            optimizer.zero_grad()

        if downstream_train_path != "None":
            encoder.eval()
            target_encoder.eval()
            predictor.eval()
            encodings_train = torch.tensor([])
            labels_train = torch.tensor([])
            encodings_val = torch.tensor([])
            labels_val = torch.tensor([])
            for itr, (udata, masks_enc, masks_pred) in enumerate(downstream_train_loader):
                def load_imgs():
                    # -- unsupervised imgs
                    imgs = udata[0].to(device, non_blocking=True)
                    labels = udata[1]
                    
                    return (imgs, labels)
                imgs, labels = load_imgs()
                labels_train=torch.cat((labels_train,labels.cpu()), 0)

                def forward():
                    with torch.no_grad():
                        h = encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        return h


                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward() # shape of h: (B,600,768) e.g. B=32
                    encodings_train = torch.cat((encodings_train,h.detach().cpu()), 0)


            for itr, (udata, masks_enc, masks_pred) in enumerate(downstream_val_loader):
                def load_imgs():
                    # -- unsupervised imgs
                    imgs = udata[0].to(device, non_blocking=True)
                    labels = udata[1]
                    return (imgs, labels)
                imgs, labels = load_imgs()
                labels_val=torch.cat((labels_val,labels.cpu()), 0)

                def forward():
                    with torch.no_grad():
                        h = encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        return h


                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward()
                    
                    encodings_val=torch.cat((encodings_val,h.detach().cpu()), 0)

            encodings_train = encodings_train.mean(dim=1)
            encodings_val = encodings_val.mean(dim=1)
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                                      random_state=0))

            pipe.fit(
                np.asarray(encodings_train), #.reshape(len(encodings_train),-1)
                np.asarray(labels_train).flatten())
            
            train_proba = pipe.predict_proba(
                np.asarray(encodings_train),  #.reshape(len(encodings_train),-1)
                )[:, 1]
            
            train_pred = pipe.predict(
                np.asarray(encodings_train),  #.reshape(len(encodings_train),-1)
                )
            
            train_acc = accuracy_score(np.asarray(labels_train).flatten(), train_pred)
            train_auc = roc_auc_score(np.asarray(labels_train).flatten(), train_proba)
            train_f1 = f1_score(np.asarray(labels_train).flatten(), train_pred)
            
            val_pred = pipe.predict(
                np.asarray(encodings_val), #.reshape(len(encodings_val),-1)
                )
            
            val_proba = pipe.predict_proba(
                np.asarray(encodings_val), #.reshape(len(encodings_val),-1)
                )[:, 1]
            
            val_acc = accuracy_score(np.asarray(labels_val).flatten(), val_pred)
            val_auc = roc_auc_score(np.asarray(labels_val).flatten(), val_proba)
            val_f1 = f1_score(np.asarray(labels_val).flatten(), val_pred)
            
            
            
            wandb.log({"downstream_train_acc": train_acc, 
                       "downstream_val_acc": val_acc,
                       "downstream_train_auc": train_auc, 
                       "downstream_val_auc": val_auc,
                       "downstream_train_f1": train_f1, 
                       "downstream_val_f1": val_f1})
                

            
            
            encoder.train()
            target_encoder.train()
            predictor.train()
            optimizer.zero_grad()
        


if __name__ == "__main__":
    main()