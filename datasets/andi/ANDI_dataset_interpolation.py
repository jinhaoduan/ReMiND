import os.path
import sys

import ants
import torch
import torchvision
from torchvision.transforms import Resize

def uniform_temporal_segment(num_ch, N, fixed=False):
    # segment images into N uniform parts and randomly select 1 channel from each part
    if fixed:
        idxs = torch.arange(0, num_ch, num_ch // N)[:N]
    else:
        len_segment = num_ch // N
        len_segment_last = num_ch % N + len_segment
        rand_idxs = torch.randint(0, len_segment, (N,))
        rand_idxs_last = torch.randint(0, len_segment_last, (1,))
        rand_idxs[-1] = rand_idxs_last[0]
        offset = torch.arange(num_ch)[::len_segment]
        idxs = offset[:rand_idxs.size(0)] + rand_idxs
    return idxs



def uniform_temporal_segment_continuous_local(num_ch, N, cont_local=3):
    # segment images into N uniform parts and randomly select 1 channel from each part
    len_segment = num_ch // (N)
    len_segment_last = num_ch % (N) + len_segment
    rand_idxs = torch.randint(0 + (cont_local // 2), len_segment - (cont_local // 2), (N,))
    rand_idxs_last = torch.randint(0 + (cont_local // 2), len_segment_last - (cont_local // 2), (1,))
    rand_idxs[-1] = rand_idxs_last[0]
    # sample neighbors
    rand_idxs = rand_idxs.reshape(-1, 1)
    rand_neighbor_idxs = []
    rand_neighbor_idxs.append(rand_idxs)
    for n in range(cont_local // 2):
        rand_neighbor_idxs.insert(0, rand_idxs - (n + 1))
        rand_neighbor_idxs.append(rand_idxs + (n + 1))

    rand_idxs = torch.concat(rand_neighbor_idxs, dim=1)
    offset = torch.arange(num_ch)[::len_segment]
    idxs = offset[:rand_idxs.size(0)].reshape(-1, 1) + rand_idxs
    idxs = idxs.reshape(-1)

    return idxs


class ANDIInterpolationDatasetFullSkull(torch.utils.data.Dataset):

    def __init__(self, data_list_f, dataset_root, transforms=None, num_segments=None, dense_sample=False, train=False, cont_local=None):
        super(ANDIInterpolationDatasetFullSkull, self).__init__()

        self.data = self._load_data(data_list_f)
        self.root = dataset_root
        self.transforms = transforms
        self.num_segments = num_segments
        self.resize = Resize(128)
        self.dense_sample = dense_sample
        self.train = train
        self.cont_local = cont_local

    def _load_data(self, data_list_f):
        data = []
        with open(data_list_f, 'r') as f:
            for l in f.readlines():
                data.append(l.strip())
        return data

    def _load_ADNI_data(self, path):
        img = torch.from_numpy(ants.image_read(path, dimension=3).numpy())
        return img

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(x):
        x_channel_min = x.flatten(1).min()
        x -= x_channel_min
        x_channel_max = x.flatten(1).max()
        x /= x_channel_max

        return x, x_channel_min, x_channel_max

    def __getitem__(self, idx):
        image_prev_path, image_current_path, image_future_path, \
        image_prev_id, image_current_id, image_future_id, \
        image_prev_label, image_current_label, image_future_label = self.data[idx].split(' ')

        subj_name = image_prev_path.split('/')[0]

        image_prev = self._load_ADNI_data(os.path.join(self.root, image_prev_path)).permute(2, 0, 1)
        image_current = self._load_ADNI_data(os.path.join(self.root, image_current_path)).permute(2, 0, 1)
        image_future = self._load_ADNI_data(os.path.join(self.root, image_future_path)).permute(2, 0, 1)

        # apply uniform temporal sampling
        if self.num_segments is not None and not self.dense_sample:
            image_prev = image_prev
            image_current = image_current
            image_future = image_future
            if self.cont_local is None or self.cont_local == 1:
                temp_seg_idxs = uniform_temporal_segment(image_prev.size(0), N=self.num_segments, fixed=not self.train)
            else:
                temp_seg_idxs = uniform_temporal_segment_continuous_local(image_prev.size(0), N=self.num_segments, cont_local=self.cont_local)
            image_prev = image_prev[temp_seg_idxs]
            image_current = image_current[temp_seg_idxs]
            image_future = image_future[temp_seg_idxs]
        elif self.dense_sample:
            # return [21, 8, 128, 128] from [1:-1] channels
            image_prev = image_prev[1:-1]
            image_current = image_current[1:-1]
            image_future = image_future[1:-1]
            if self.cont_local is None or self.cont_local == 1:
                # rearrange from [0, 2, ..., 167] to [0, 21, ..., XX, 1, 22, ..., XX+1]
                temp_seg_idx_base = torch.arange(0, image_prev.size(0), image_prev.size(0) // self.num_segments)
                dense_idxs = torch.concat(
                    [temp_seg_idx_base + i for i in range(image_prev.size(0) // self.num_segments)]).reshape(-1)
            else:
                # rearrange to [168, 128, 128] -> [14, 12, 128, 128]
                temp_seg_idx_base = torch.arange(0, image_prev.size(0), image_prev.size(0) // self.num_segments)
                temp_seg_idx_base = temp_seg_idx_base.repeat_interleave(3) + torch.tensor(
                    list(range(self.cont_local)) * self.num_segments)
                dense_idxs = torch.concat([temp_seg_idx_base + self.cont_local * i for i in
                                           range(image_prev.size(0) // self.num_segments // self.cont_local)]).reshape(
                    -1)

            image_prev = image_prev[dense_idxs]
            image_current = image_current[dense_idxs]
            image_future = image_future[dense_idxs]
        else:
            raise NotImplemented

        image_prev = self.resize(image_prev)
        image_current = self.resize(image_current)
        image_future = self.resize(image_future)


        # apply transform
        if self.transforms is not None:
            image_prev = self.transforms(image_prev)
            image_current = self.transforms(image_current)
            image_future = self.transforms(image_future)

        # 1, C, H, W -> C, H, W
        image_prev, image_prev_min, image_prev_max = self.normalize(image_prev.squeeze(0))
        image_current, _, _ = self.normalize(image_current.squeeze(0))
        image_future, _, _ = self.normalize(image_future.squeeze(0))

        return {
            'image_prev': image_prev,
            'image_prev_name': image_prev_path.split('/')[-1],
            'image_current': image_current,
            'image_current_name': image_current_path.split('/')[-1],
            'image_future': image_future,
            'image_future_name': image_future_path.split('/')[-1],
            'image_prev_min': image_prev_min,
            'image_prev_max': image_prev_max,
            'image_prev_id': image_prev_id,
            'image_current_id': image_current_id,
            'image_future_id': image_future_id,
            'image_prev_path': image_prev_path,
            'image_current_path': image_current_path,
            'image_future_path': image_future_path,
            'stage': image_prev_label,
            'subj_name': subj_name

        }


def get_ANDI_dataloader(data_root, list_path, batch_size=16, shuffle=False, num_workers=4,
                        num_segments=None, dense_sample=False, train=False, interpolate=False, cont_local=None):

    if interpolate:
        print('Using AC->B dataset')
        dataset = ANDIInterpolationDatasetFullSkull(list_path, data_root, num_segments=num_segments, dense_sample=dense_sample,
                                           train=train, cont_local=cont_local)
    else:
        raise NotImplemented

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             drop_last=True)
    return dataloader
