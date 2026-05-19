import random
import os
from os.path import expanduser

import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import imwrite
from XImage import CXImage
import blobfile as bf


def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out


def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)


def eval_imswrite( srs=None, img_names=None, dset=None, name=None, ext='tif', lrs=None, gts=None, gt_keep_masks=None,conf=None, verify_same=True):
    img_names = to_file_ext(img_names, ext)

    if srs is not None:
        sr_dir_path = expanduser(conf.data.eval.paths.srs)
        write_images(srs, img_names, sr_dir_path)

    if gt_keep_masks is not None:
        mask_dir_path = expanduser(
            conf.data.eval.paths.gt_keep_masks)
        write_images(gt_keep_masks, img_names, mask_dir_path)

    gts_path = conf.data.eval.paths.gts
    if gts is not None and gts_path:
        gt_dir_path = expanduser(gts_path)
        write_images(gts, img_names, gt_dir_path)

    if lrs is not None:
        lrs_dir_path = expanduser(
            conf.data.eval.paths.lrs)
        write_images(lrs, img_names, lrs_dir_path)


def load_data_inpa(
    conf,
    model_flag='train',
    ** kwargs
):

    if model_flag=='eval':
        gt_dir = os.path.expanduser(conf.data.eval.gt_path)
        gt_paths = _list_image_files_recursively(gt_dir)

        mask_paths=''
        refer_paths=''
        mask_dir = os.path.expanduser(conf.data.eval.mask_path)
        mask_paths = _list_image_files_recursively(mask_dir)
        if conf.condition.condition_flag:
            refer_dir=os.path.expanduser(conf.data.eval.refer_path)
            refer_paths=_list_image_files_recursively(refer_dir)

        assert len(gt_paths) == len(mask_paths)
        random_crop = conf.data.eval.random_crop
        random_flip = conf.data.eval.random_flip
        return_dict = conf.data.eval.return_dict
        max_len = conf.data.eval.max_len
        offset = conf.data.eval.offset
        mask_loader_flag = True
    else:
        gt_dir = os.path.expanduser(conf.data.train.gt_path)
        gt_paths = _list_image_files_recursively(gt_dir)

        mask_paths = ''
        refer_paths = ''
        if conf.condition.condition_flag:
            refer_dir = os.path.expanduser(conf.data.train.refer_path)
            refer_paths = _list_image_files_recursively(refer_dir)
        assert len(gt_paths) == len(refer_paths)
        random_crop = conf.data.train.random_crop
        random_flip = conf.data.train.random_flip
        return_dict = conf.data.train.return_dict
        max_len = conf.data.train.max_len
        offset = conf.data.train.offset
        mask_loader_flag = False


    dataset = ImageDatasetInpa(
        conf.model.image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        refer_paths=refer_paths,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        return_dict=return_dict,
        max_len=max_len,
        offset= offset,
        mask_loader_flag= mask_loader_flag,
        condition_flag=conf.condition.condition_flag,
        augment=conf.data.train.augment
    )

    if model_flag=='eval':
        loader = DataLoader(
            dataset, batch_size=conf.data.eval.batch_size, shuffle=False, num_workers=0, drop_last=conf.data.eval.drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=conf.data.train.batch_size, shuffle=True, num_workers=0, drop_last=conf.data.train.drop_last
        )

    if model_flag == 'train':
        def generator_func():
            while True:
                for data in loader:
                    yield data

        return generator_func()
    else:
        return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif","tif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        refer_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        return_dict=False,
        max_len=None,
        offset=0,
        mask_loader_flag=True,
        condition_flag="sr",
        augment=False
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        self.local_gts = gt_paths[shard:][::num_shards]
        if mask_loader_flag:
            mask_paths = sorted(mask_paths)[offset:]
            self.local_masks = mask_paths[shard:][::num_shards]
        if condition_flag!='uncondition':
            refer_paths=sorted(refer_paths)[offset:]
            self.refer_gts=refer_paths[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len
        self.mask_loader_flag=mask_loader_flag
        self.condition_flag=condition_flag
        self.augment=augment

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        gt_data = self.imread(gt_path)
        if self.condition_flag!='uncondition':
            refer_path=self.refer_gts[idx]
            refer_data=self.imread(refer_path)
        if self.mask_loader_flag:
            mask_path = self.local_masks[idx]
            mask_data = self.imread(mask_path)

        if self.random_crop:
            raise NotImplementedError()
        else:
            if self.augment==True:
                augment_flag=np.random.randint(0,3)
            else:
                augment_flag=0
            arr_gt = center_crop_arr(gt_data, self.resolution,num=augment_flag)
            if self.condition_flag!='uncondition':
                arr_ref=center_crop_arr(refer_data,self.resolution,num=augment_flag)
            if self.mask_loader_flag:
                arr_mask = center_crop_arr(mask_data, self.resolution,num=augment_flag)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_ref=arr_ref[:,::-1]
            if self.mask_loader_flag:
                arr_mask = arr_mask[:, ::-1]

        arr_gt,gt_min,gt_max=dem_zip(arr_gt)
        if self.mask_loader_flag:
            if np.max(arr_mask)>1:
                arr_mask = arr_mask.astype(np.float32) / 255.0
            arr_mask=np.expand_dims(arr_mask, axis=2)
            arr_mask=np.ascontiguousarray(arr_mask)

        if self.condition_flag=='sr':
            arr_ref,ref_min,ref_max = dem_zip(arr_ref)
            arr_ref = np.ascontiguousarray(arr_ref)
        if self.condition_flag=='tf':
            arr_ref=np.where(arr_ref == 255, -1, arr_ref)
            arr_ref=np.expand_dims(arr_ref, axis=2)
            arr_ref = np.ascontiguousarray(arr_ref)

        if self.return_dict:
            name = os.path.basename(gt_path)
            outdict = {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_min': gt_min,
                'gt_max': gt_max
            }
            if self.mask_loader_flag:
                outdict.update({
                    'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1])
                })
            if self.condition_flag!="uncondition":
                outdict.update({
                    'refer': np.transpose(arr_ref, [2, 0, 1])
                })
            return outdict
        else:
            raise NotImplementedError()

    def imread(self, path):
        img=CXImage()
        img.Open(path)
        data=img.GetData(np.float32)

        return data


def center_crop_arr(arr, image_size,num=0):

    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    res=arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if num == 1:
        res = np.fliplr(res)
    elif num == 2:
        res = np.flipud(res)
    elif num == 3:
        res = np.rot90(res, 2)
    return res


def dem_zip(image_array):
    if len(image_array.shape) == 2:
        min_value = np.min(image_array)
        max_value = np.max(image_array)
        data_range = max_value - min_value
        low_value = min_value - data_range * 0.1
        high_value = max_value + data_range * 0.1

        image_array = ((image_array - low_value) / (high_value - low_value)).astype(np.float32)
        image_array = image_array * 2 - 1
        image = np.expand_dims(image_array, axis=2)
        return image, min_value, max_value
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        min_values = np.min(image_array, axis=(0, 1))
        max_values = np.max(image_array, axis=(0, 1))
        ranges = max_values - min_values
        low_values = min_values - ranges * 0.1
        high_values = max_values + ranges * 0.1

        image_size, _, num_channels = image_array.shape

        for i in range(num_channels):
            image_array[:, :, i] = (
                        (image_array[:, :, i] - low_values[i]) / (high_values[i] - low_values[i])).astype(
                np.float32)
            image_array[:, :, i] = image_array[:, :, i] * 2 - 1
        image_array[np.isnan(image_array)] = 0
        return image_array, min_values, max_values
    else:
        print("invalid image shape!")