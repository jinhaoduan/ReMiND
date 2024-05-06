
import argparse
import numpy as np
import os
import torch
import tqdm
import yaml
import random
import ants
from collections import OrderedDict

from datasets.andi.ANDI_dataset_interpolation import get_ANDI_dataloader

try:
    from torchvision.transforms.functional import resize, InterpolationMode

    interp = InterpolationMode.NEAREST
except:
    from torchvision.transforms.functional import resize

    interp = 0

from datasets import data_transform, inverse_data_transform
from main import dict2namespace
from models.ema import EMAHelper
from runners.ncsn_runner_interpolation import get_model, conditioning_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from models import ddim_sampler

from torchvision.transforms import Resize
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')


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


def stretch_image(X, ch, imsize):
    return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2)

def reverse_dense_idxs(x, config):
    x = x.reshape(-1, config.data.image_size, config.data.image_size)
    temp_seg_idx_base = torch.arange(0, x.size(0), config.data.channels)
    dense_idxs = torch.concat([temp_seg_idx_base + i for i in range(config.data.channels)]).reshape(-1)
    return x[dense_idxs].reshape(-1, config.data.image_size, config.data.image_size).unsqueeze(0)



def convert_to_nii_type(sample, mask_visit, channel_min, channel_max, apply_mask, n_segments, cont_local=1):
    # apply mask
    if apply_mask:
        sample[mask_visit == 0] = 0
    if cont_local == 1:
        # sample: [21, 8, 128, 128] -> [1, 168, 128, 128]
        nii_sample = sample.transpose(0, 1).reshape(sample.size(0) * sample.size(1), sample.size(2), sample.size(3))
    else:
        # sample [14, cont_local * n_segment, 128, 128] -> [1, 168, 128, 128]
        nii_sample = sample.reshape(-1, n_segments, cont_local, sample.size(2), sample.size(3)).transpose(0, 1).reshape(1, -1, sample.size(2), sample.size(3))
    nii_sample = F.pad(Resize(170)(nii_sample), (40, 46, 25, 61), value=0)
    nii_sample *= channel_max.to(nii_sample.device)
    nii_sample += channel_min.to(nii_sample.device)
    new_nii_sample = torch.zeros(170, 256, 256)
    new_nii_sample[1:-1, ...] = nii_sample
    return new_nii_sample.permute(1, 2, 0)


def complete_sample_interpolation(ckpt_path, test_list_path, save_as_nii=False):
    device = 'cuda'
    scorenet, config = load_model(ckpt_path, device)

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    # load global template
    global_template = ants.image_read(os.path.join('./datasets/andi/resource', 'T_template0.nii.gz'), dimension=3)
    # Initial samples
    print(config.data.num_frames_future)
    test_loader = get_ANDI_dataloader(config.data.data_root, test_list_path,
                                      batch_size=1, shuffle=False,
                                      num_workers=config.data.num_workers,
                                      num_segments=config.data.channels // config.data.cont_local,
                                      cont_local=None if config.data.cont_local == 1 else config.data.cont_local,
                                      dense_sample=True,
                                      train=False,
                                      interpolate=True if config.data.num_frames_future >= 1 else False)

    for i, data in enumerate(tqdm.tqdm(test_loader)):
        image_current_name = data['image_current_path'][0].split('/')[-1]
        image_prev = data_transform(config, data['image_prev'].to(config.device))
        if config.data.num_frames_future >= 1:
            image_future = data_transform(config, data['image_future'].to(config.device))
        else:
            image_future = None


        # from [1, C, H, W] -> [C//c, c, H, W]
        image_prev = image_prev.unsqueeze(0).reshape(-1, config.data.channels, config.data.image_size,
                                                      config.data.image_size)

        image_future = image_future.unsqueeze(0).reshape(-1, config.data.channels, config.data.image_size,
                                                         config.data.image_size)
        cond = torch.concat([image_prev, image_future], dim=1)


        init_samples = torch.randn(len(image_prev), config.data.channels * config.data.num_frames,
                                   config.data.image_size, config.data.image_size,
                                   device=config.device)

        # sample for DDPM
        with torch.cuda.amp.autocast():
            all_samples = ddim_sampler(init_samples, scorenet, cond=cond[:len(init_samples)],
                                       n_steps_each=config.sampling.n_steps_each,
                                       step_lr=config.sampling.step_lr, just_beta=False,
                                       final_only=True, denoise=config.sampling.denoise,
                                       subsample_steps=getattr(config.sampling, 'subsample', None),
                                       verbose=False)


        sample = all_samples[-1].reshape(all_samples[-1].shape[0], config.data.channels,
                                         config.data.image_size, config.data.image_size)

        sample = inverse_data_transform(config, sample)
        image_prev = inverse_data_transform(config, image_prev)

        if save_as_nii:

            nii_sample = convert_to_nii_type(sample, image_prev, data['image_prev_min'], data['image_prev_max'],
                                             apply_mask=True, n_segments=config.data.channels // config.data.cont_local,
                                             cont_local=config.data.cont_local).cpu().numpy()

            ants_img = ants.from_numpy(nii_sample,
                                       origin=global_template.origin,
                                       spacing=global_template.spacing,
                                       direction=global_template.direction)
            ants.plot(ants_img)

            ants.image_write(ants_img, f'./interpolated-{image_current_name}')

def fix_seed():
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--test-list-path', type=str, required=True, help='Path to the test list')
    parser.add_argument('--save-as-nii', action='store_true')
    args = parser.parse_args()

    result = complete_sample_interpolation(args.ckpt_path, args.test_list_path, save_as_nii=args.save_as_nii)


