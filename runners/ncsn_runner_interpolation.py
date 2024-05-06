import datetime
import logging
import imageio
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import sys
import time
import yaml

import torch

from functools import partial

from torch.distributions.gamma import Gamma
from torchvision.utils import make_grid, save_image

from datasets import data_transform, inverse_data_transform
from datasets.andi.ANDI_dataset_interpolation import get_ANDI_dataloader
from losses import get_optimizer, warmup_lr, dynamic_lr
from losses.dsm import anneal_dsm_score_estimation
from models import (ddpm_sampler,
                    ddim_sampler,
                    FPNDM_sampler,
                    anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_consistent)
from models.ema import EMAHelper

scaler = torch.cuda.amp.GradScaler()

__all__ = ['NCSNRunner']


def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)


def conditioning_fn(config, X, num_frames_pred=0, prob_mask_cond=0.0, prob_mask_future=0.0, conditional=True):
    imsize = config.data.image_size
    if not conditional:
        return X.reshape(len(X), -1, imsize, imsize), None, None

    cond = config.data.num_frames_cond
    train = config.data.num_frames
    pred = num_frames_pred
    future = getattr(config.data, "num_frames_future", 0)

    # Frames to train on / sample
    pred_frames = X[:, cond:cond+pred].reshape(len(X), -1, imsize, imsize)

    # Condition (Past)
    cond_frames = X[:, :cond].reshape(len(X), -1, imsize, imsize)

    if prob_mask_cond > 0.0:
        cond_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_cond)
        cond_frames = cond_mask.reshape(-1, 1, 1, 1) * cond_frames
        cond_mask = cond_mask.to(torch.int32) # make 0,1
    else:
        cond_mask = None

    # Future
    if future > 0:

        if prob_mask_future == 1.0:
            future_frames = torch.zeros(len(X), config.data.channels*future, imsize, imsize)
            # future_mask = torch.zeros(len(X), 1, 1, 1).to(torch.int32) # make 0,1
        else:
            future_frames = X[:, cond+train:cond+train+future].reshape(len(X), -1, imsize, imsize)
            if prob_mask_future > 0.0:
                if getattr(config.data, "prob_mask_sync", False):
                    future_mask = cond_mask
                else:
                    future_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_future)
                future_frames = future_mask.reshape(-1, 1, 1, 1) * future_frames
            #     future_mask = future_mask.to(torch.int32) # make 0,1
            # else:
            #     future_mask = None

        cond_frames = torch.cat([cond_frames, future_frames], dim=1)

    return pred_frames, cond_frames, cond_mask   # , future_mask


def stretch_image(X, ch, imsize):
    return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)


# Make and load model
def load_model(ckpt_path, device):
    # Parse config file
    with open(os.path.join(os.path.dirname(ckpt_path), 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Load config file
    config = dict2namespace(config)
    config.device = device
    # Load model
    scorenet = get_model(config)
    if config.device != torch.device('cpu'):
        scorenet = torch.nn.DataParallel(scorenet)
        states = torch.load(ckpt_path, map_location=config.device)
    else:
        states = torch.load(ckpt_path, map_location='cpu')
        states[0] = OrderedDict([(k.replace('module.', ''), v) for k, v in states[0].items()])
    scorenet.load_state_dict(states[0], strict=False)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(scorenet)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(scorenet)
    scorenet.eval()
    return scorenet, config


def get_model(config):

    version = getattr(config.model, 'version', 'SMLD').upper()
    arch = getattr(config.model, 'arch', 'ncsn')
    depth = getattr(config.model, 'depth', 'deep')

    if arch == 'unetmore':
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        return UNetMore_DDPM(config).to(config.device)#.to(memory_format=torch.channels_last).to(config.device)
    elif arch in ['unetmore3d', 'unetmorepseudo3d']:
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        return UNetMore_DDPM(config).to(config.device)#.to(memory_format=torch.channels_last).to(config.device)

    else:
        Exception("arch is not valid [ncsn, unet, unetmore, unetmore3d]")

class NCSNRunner():
    def __init__(self, args, config, config_uncond):
        self.args = args
        self.config = config
        self.config_uncond = config_uncond
        self.version = getattr(self.config.model, 'version', "SMLD")
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        self.get_mode()
        self.epochs = RunningAverageMeter()
        self.losses_train, self.losses_test = RunningAverageMeter(), RunningAverageMeter()
        self.lr_meter, self.grad_norm = RunningAverageMeter(), RunningAverageMeter()
        self.time_train, self.time_elapsed = RunningAverageMeter(), RunningAverageMeter()
        self.time_train_prev = self.time_elapsed_prev = 0

    def get_mode(self):
        self.condf, self.condp = self.config.data.num_frames_cond, getattr(self.config.data, "prob_mask_cond", 0.0)
        self.futrf, self.futrp = getattr(self.config.data, "num_frames_future", 0), getattr(self.config.data, "prob_mask_future", 0.0)
        self.prob_mask_sync = getattr(self.config.data, "prob_mask_sync", False)
        if not getattr(self.config.sampling, "ssim", False):
            if getattr(self.config.sampling, "fvd", False):
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, "three"
            else:
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, None
        elif self.condp == 0.0 and self.futrf == 0:                                                   # (1) Prediction
            self.mode_pred, self.mode_interp, self.mode_gen = "one", None, None
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:                            # (1) Interpolation
            self.mode_pred, self.mode_interp, self.mode_gen = None, "one", None
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:                             # (1) Interp + (2) Pred
            self.mode_pred, self.mode_interp, self.mode_gen = "two", "one", None
        elif self.condp > 0.0 and self.futrf == 0:                                                  # (1) Pred + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = "one", None, "three"
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:  # (1) Interp + (2) Pred + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = "two", "one", "three"
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:      # (1) Interp + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = None, "one", "three"

    def get_time(self):
        curr_time = time.time()
        curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
        return curr_time_str, elapsed

    def convert_time_stamp_to_hrs(self, time_day_hr):
        time_day_hr = time_day_hr.split(",")
        if len(time_day_hr) > 1:
            days = time_day_hr[0].split(" ")[0]
            time_hr = time_day_hr[1]
        else:
            days = 0
            time_hr = time_day_hr[0]
        # Hr
        hrs = time_hr.split(":")
        return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600

    def train(self):
        if self.config.data.dataset.upper() == 'ANDI':
            dataloader = get_ANDI_dataloader(self.config.data.data_root, self.config.data.train_list_path,
                                             batch_size=self.config.training.batch_size, shuffle=True,
                                             num_workers=self.config.data.num_workers,
                                             # uniform sample along temporal axis
                                             num_segments=self.config.data.channels // self.config.data.cont_local,
                                             train=True, cont_local=self.config.data.cont_local,
                                             interpolate=True)
            val_loader = get_ANDI_dataloader(self.config.data.data_root, self.config.data.val_list_path,
                                              batch_size=self.config.training.sampling_batch_size, shuffle=False,
                                              num_workers=self.config.data.num_workers,
                                              num_segments=self.config.data.channels // self.config.data.cont_local,
                                              train=False, cont_local=self.config.data.cont_local,
                                              interpolate=True)
            val_iter = iter(val_loader)
            train_iter = iter(dataloader)
        else:
            raise NotImplemented

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        scorenet = get_model(self.config)

        logging.info(f"Number of parameters: {count_parameters(scorenet)}")
        logging.info(f"Number of trainable parameters: {count_training_parameters(scorenet)}")

        optimizer = get_optimizer(self.config, scorenet.parameters())

        scorenet = torch.nn.DataParallel(scorenet)

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            logging.info(f"Number of GPUs : {num_devices}")
            for i in range(num_devices):
                logging.info(torch.cuda.get_device_properties(i))
        else:
            logging.info(f"Running on CPU!")

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'))
            scorenet.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resuming training from checkpoint.pt in {self.args.log_path} at epoch {start_epoch}, step {step}.")

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(getattr(self.config.model, 'num_classes'))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(getattr(self.config.model, 'num_classes')):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(getattr(self.config.model, 'num_classes')):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        print(scorenet)
        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        cond, test_cond = None, None

        # Future
        future = getattr(self.config.data, "num_frames_future", 0)

        # Initial samples
        n_init_samples = min(36, self.config.training.sampling_batch_size)
        init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)
        if self.version == "SMLD":
            init_samples = torch.rand(init_samples_shape, device=self.config.device)
            init_samples = data_transform(self.config, init_samples)
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            if getattr(self.config.model, 'gamma', False):
                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                init_samples = z - used_k*used_theta # we don't scale here
            else:
                init_samples = torch.randn(init_samples_shape, device=self.config.device)

        # Sampler
        sampler = self.get_sampler()

        self.total_train_time = 0
        self.start_time = time.time()

        early_end = False

        for _ in range(3):
            visualization_sample = next(val_iter)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for batch, data in enumerate(dataloader):

                optimizer.zero_grad()
                lr = warmup_lr(optimizer, step, getattr(self.config.optim, 'warmup', 0), self.config.optim.lr)
                scorenet.train()
                step += 1

                # Data
                prev_visit = data_transform(self.config, data['image_prev'].to(self.config.device))
                current_visit = data_transform(self.config, data['image_current'].to(self.config.device))
                future_visit = data_transform(self.config, data['image_future'].to(self.config.device))
                cond = torch.concat([prev_visit, future_visit], dim=1)
                cond_mask = None


                # Loss
                itr_start = time.time()
                loss = anneal_dsm_score_estimation(scorenet, current_visit, labels=None, cond=cond, cond_mask=cond_mask,
                                                   loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                   gamma=getattr(self.config.model, 'gamma', False),
                                                   L1=getattr(self.config.training, 'L1', False), hook=hook,
                                                   all_frames=getattr(self.config.model, 'output_all_frames', False))

                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(scorenet.parameters(),
                                                       getattr(self.config.optim, 'grad_clip', np.inf))
                scaler.step(optimizer)
                scaler.update()

                # Training time
                itr_time = time.time() - itr_start
                self.total_train_time += itr_time
                self.time_train.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.total_train_time))) + self.time_train_prev)

                # Record
                self.losses_train.update(loss.item(), step)
                self.epochs.update(epoch + (batch + 1)/len(dataloader))
                self.lr_meter.update(lr)
                self.grad_norm.update(grad_norm.item())

                if step == 1 or step % getattr(self.config.training, "log_freq", 1) == 0:
                    logging.info("elapsed: {}, train time: {:.04f}, mem: {:.03f}GB, GPUmem: {:.03f}GB, step: {}, lr: {:.06f}, grad: {:.04f}, loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        self.time_train.val, get_proc_mem(), get_GPU_mem(), step, lr, grad_norm, loss.item()))

                if self.config.model.ema:
                    ema_helper.update(scorenet)

                if step >= self.config.training.n_iters:
                    early_end = True
                    break

                # Save model
                if (step % 1000 == 0 and step != 0) or step % self.config.training.snapshot_freq == 0:
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    logging.info(f"Saving checkpoint.pt in {self.args.log_path}")
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))
                    if step % self.config.training.snapshot_freq == 0:
                        ckpt_path = os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step))
                        logging.info(f"Saving {ckpt_path}")
                        torch.save(states, ckpt_path)

                test_scorenet = None
                # Get test_scorenet
                if step == 1 or step % self.config.training.val_freq == 0 or (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    if self.config.model.ema:
                        test_scorenet = ema_helper.ema_copy(scorenet)
                    else:
                        test_scorenet = scorenet

                    test_scorenet.eval()

                # Validation
                if step == 1 or step % self.config.training.val_freq == 0:
                    test_data = visualization_sample
                    test_current_visit = data_transform(self.config, test_data['image_current'].to(self.config.device))
                    test_prev_visit = data_transform(self.config, test_data['image_prev'].to(self.config.device))
                    test_future_visit = data_transform(self.config, test_data['image_future'].to(self.config.device))
                    test_cond = torch.concat([test_prev_visit, test_future_visit], dim=1)
                    test_cond_mask = None

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_scorenet, test_current_visit, labels=None, cond=test_cond, cond_mask=test_cond_mask,
                                                                    loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                                    gamma=getattr(self.config.model, 'gamma', False),
                                                                    L1=getattr(self.config.training, 'L1', False), hook=test_hook,
                                                                    all_frames=getattr(self.config.model, 'output_all_frames', False))
                    # tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                    # test_tb_hook()
                    self.losses_test.update(test_dsm_loss.item(), step)
                    logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, test_loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        step, get_proc_mem(), get_GPU_mem(), test_dsm_loss.item()))

                # Sample from model
                if step == 1 or (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    logging.info(f"Saving images in {self.args.log_sample_path}")

                    # Samples
                    if conditional:
                        try:
                            data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            data = next(val_iter)

                        test_prev_visit = data_transform(self.config, data['image_prev'].to(self.config.device)[:2])
                        test_current_visit = data_transform(self.config, data['image_current'].to(self.config.device)[:2])
                        test_future_visit = data_transform(self.config, data['image_future'].to(self.config.device)[:2])
                        test_cond = torch.concat([test_prev_visit, test_future_visit], dim=1)
                        test_cond_mask = None
                    with torch.cuda.amp.autocast():
                        all_samples = sampler(init_samples, test_scorenet, cond=test_cond, cond_mask=test_cond_mask,
                                          n_steps_each=self.config.sampling.n_steps_each,
                                          step_lr=self.config.sampling.step_lr, just_beta=False,
                                          final_only=True, denoise=self.config.sampling.denoise,
                                          subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                          clip_before=getattr(self.config.sampling, 'clip_before', True),
                                          verbose=False, log=False, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')

                    pred = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                   self.config.data.image_size, self.config.data.image_size)
                    # pred = torch.clip(difference_pred, -1.0, 1.0)
                    pred = inverse_data_transform(self.config, pred)


                    def calculate_l2(x1, x2):
                        # x [B, C, H, W]
                        return torch.sqrt(((x1 - x2) ** 2).flatten(1).sum(dim=-1)).mean()

                    if conditional:
                        reali = inverse_data_transform(self.config, test_current_visit.to('cpu'))
                        gt_prev = inverse_data_transform(self.config, test_prev_visit.to('cpu'))
                        gt_future = inverse_data_transform(self.config, test_future_visit.to('cpu'))

                        # apply mask
                        masked_pred = pred.detach().clone()
                        masked_pred[gt_prev == 0] = 0

                        # calculate L2 distance
                        l2_pred_real = calculate_l2(pred, reali)
                        l2_pred_cond = calculate_l2(pred, gt_prev)
                        l2_real_cond = calculate_l2(reali, gt_prev)

                        # masked_pred = pred.detach().clone()
                        # masked_pred[condi == 0] = 0
                        masked_l2_pred_real = calculate_l2(masked_pred, reali)
                        masked_l2_pred_cond = calculate_l2(masked_pred, gt_prev)

                        logging.info(f'Training: L2 pred-real: {l2_pred_real.item()} pred-cond: {l2_pred_cond.item()} '
                                     f'real-cond: {l2_real_cond.item()} Masked L2 pred-real: {masked_l2_pred_real.item()} '
                                     f'pred-cond: {masked_l2_pred_cond.item()}')

                        logging.info(f'Training masked L2 pred-real: {masked_l2_pred_real.item()} '
                                     f'masked L2 pred-cond: {masked_l2_pred_cond.item()} ')

                        # if future > 0:
                        #     condi, futri = condi[:, :self.config.data.num_frames_cond*self.config.data.channels], \
                        #                    condi[:, self.config.data.num_frames_cond*self.config.data.channels:]

                        # Stretch out multiple frames horizontally
                        pred = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                        masked_pred = stretch_image(masked_pred, self.config.data.channels, self.config.data.image_size)

                        reali = stretch_image(reali, self.config.data.channels, self.config.data.image_size)
                        gt_prev = stretch_image(gt_prev, self.config.data.channels, self.config.data.image_size)
                        gt_future = stretch_image(gt_future, self.config.data.channels, self.config.data.image_size)

                        # if future > 0:
                        #     futri = stretch_image(futri, self.config.data.channels, self.config.data.image_size)

                        padding = 0.5 * torch.ones(len(reali), self.config.data.channels, self.config.data.image_size, 4)

                        image_list = []
                        diff_list = []
                        for b in range(reali.size(0)):
                            prev_b = gt_prev[b, ::self.config.data.cont_local].unsqueeze(1)
                            future_b = gt_future[b, ::self.config.data.cont_local].unsqueeze(1)
                            reali_b = reali[b, ::self.config.data.cont_local].unsqueeze(1)
                            pred_b = pred[b, ::self.config.data.cont_local].unsqueeze(1)
                            masked_pred_b = masked_pred[b, ::self.config.data.cont_local].unsqueeze(1)

                            image_list.append(prev_b)
                            image_list.append(reali_b)
                            image_list.append(future_b)
                            image_list.append(pred_b)
                            image_list.append(masked_pred_b)

                            break

                        image_grid_list = [make_grid(img, nrow=self.config.data.channels // self.config.data.cont_local) for img in image_list]
                        diff_grid_list = [make_grid(diff, nrow=self.config.data.channels // self.config.data.cont_local) for diff in diff_list]

                        for i in range(len(image_grid_list) + len(diff_grid_list)):
                            ax = plt.subplot(len(image_grid_list) + len(diff_grid_list), 1, i + 1)
                            ax.axis('off')
                            if i < len(image_grid_list):
                                plt.imshow(image_grid_list[i].permute(1, 2, 0), cmap=matplotlib.colormaps['gray'])

                            else:
                                grid = diff_grid_list[i - len(image_grid_list)]
                                M = max(-grid.min(), grid.max())
                                print(grid.min(), grid.max())
                                grid[grid == grid.max()] = M
                                grid[grid == grid.min()] = -M
                                plt.imshow(grid.permute(1, 2, 0)[:, :, :1], cmap=matplotlib.colormaps['bwr'])
                            plt.colorbar()

                        plt.savefig(os.path.join(self.args.log_sample_path, 'train_image_grid_{}.png'.format(step)),
                                    bbox_inches='tight')

                    # torch.save(pred, os.path.join(self.args.log_sample_path, 'train_samples_{}.pt'.format(step)))

                    del all_samples

                del test_scorenet

                self.time_elapsed.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=(time.time() - self.start_time)))) + self.time_elapsed_prev)

            if early_end:
                break

        # Save model at the very end
        states = [
            scorenet.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        logging.info(f"Saving checkpoints in {self.args.log_path}")
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step)))
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))

    def get_sampler(self):
        # Sampler
        if self.version == "SMLD":
            consistent = getattr(self.config.sampling, 'consistent', False)
            sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
        elif self.version == "DDPM":
            print('Using DDPM sampler')
            sampler = partial(ddpm_sampler, config=self.config)
        elif self.version == "DDIM":
            print('Using DDIM sampler')
            sampler = partial(ddim_sampler, config=self.config)
        elif self.version == "FPNDM":
            sampler = partial(FPNDM_sampler, config=self.config)

        return sampler


    def write_to_pickle(self, pickle_file, my_dict):
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as handle:
                old_dict = pickle.load(handle)
            for key in my_dict.keys():
                old_dict[key] = my_dict[key]
            my_dict = {}
            for key in sorted(old_dict.keys()):
                my_dict[key] = old_dict[key]
        with open(pickle_file, 'wb') as handle:
            pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def write_to_yaml(self, yaml_file, my_dict):
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                old_dict = yaml.load(f, Loader=yaml.FullLoader)
            for key in my_dict.keys():
                old_dict[key] = my_dict[key]
            my_dict = {}
            for key in sorted(old_dict.keys()):
                my_dict[key] = old_dict[key]
        with open(yaml_file, 'w') as f:
            yaml.dump(my_dict, f, default_flow_style=False)
