import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from utils import imwrite
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data_inpa, eval_imswrite, write_images
from guided_diffusion.recordImages import writeRecords
import yaml

# Workaround for system library dependency
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    select_args,
)


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample[:, :, :, 0]


def to255(sample):
    if sample is None:
        return sample
    sample = ((sample + 1) / 2 * 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample[0, :, :, 0]


def toF32(sample, min_e, max_e):
    if sample is None:
        return sample
    sample = (sample + 1) / 2
    value_range = max_e - min_e
    low_value = min_e - value_range * 0.1
    high_value = max_e + value_range * 0.1
    sample = (high_value - low_value) * sample + low_value

    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    img = sample[:, :, :, 0]
    return img


def writeRecordX0(records, times, min_e, max_e, imageNames, path):
    index = 0
    imgs = []
    pre = times[0]
    for record, time in zip(records, times):
        if index % 200 == 0 or index < 10 or time == 0:
            if time < pre:
                img8 = to255(record)
                imgs.append(img8)
                pre = time
        index += 1
    writeRecords(imgs, path, imageNames[0])


def main(conf):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = dist_util.dev()
    conf.device = device
    dict_model_and_diffusion = {**vars(conf.model), **(vars(conf.diffusion))}
    
    if conf.condition.condition_flag == "uncondition":
        model, diffusion = create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, model_and_diffusion_defaults().keys()), conf=conf
        )
    else:
        dict_model_and_diffusion.update(**vars(conf.condition))
        model, diffusion = sr_create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, sr_model_and_diffusion_defaults()), conf=conf
        )
    
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    
    if conf.model.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress
    cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, y, gt=gt)

    print("sampling...")
    dl = load_data_inpa(conf, model_flag='eval')

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']
        
        if conf.condition.condition_flag != "uncondition":
            model_kwargs["y"] = batch['refer']
        
        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result, ref, records, times = sample_fn(
            model_fn,
            (batch_size, conf.model.in_channels, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )

        min_e = batch['gt_min'].item()
        max_e = batch['gt_max'].item()
        writeRecordX0(records, times, min_e, max_e, batch['GT_name'], path=conf.data.eval.paths.records)

        srs = toF32(result['sample'], min_e=min_e, max_e=max_e)
        gts = toF32(result['gt'], min_e=min_e, max_e=max_e)
        
        tmp = result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) * \
              th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))
        lrs = toF32(tmp, min_e=min_e, max_e=max_e)
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], conf=conf, verify_same=False
        )

    print("sampling complete")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = parser.parse_args()
    
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    main(new_config)