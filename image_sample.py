"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.log_dir)
    
    assert args.guidance_type in ['dps', 'dds']
    my_kwargs = {
        'sim_guided': args.sim_guided,
        'eta': args.eta,
        'exp_name': os.path.basename(args.log_dir),
        'prop_dir': args.prop_dir,
        'save_inter': args.save_inter,
        'interval': args.interval,
        'inter_rate': args.inter_rate,
        'guidance_type': args.guidance_type,
        'use_normed_grad': args.use_normed_grad,
        'stoptime': args.stoptime,
        'use_adjgrad_norm': args.use_adjgrad_norm,
        'sim_type': args.sim_type,
        'tsr': args.tsr,
        'manual_class_id': args.manual_class_id,
    }
    print("my_kwargs: ", my_kwargs)
    
    if not args.gpu_id == '':
        logger.log("using device %s" % args.gpu_id)
        th.cuda.set_device(th.device(f"cuda:{int(args.gpu_id)}"))

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if args.manual_class_id != "":
                assert 0 <= int(args.manual_class_id) < args.num_classes
                classes = th.ones(args.batch_size, dtype=th.long, device=dist_util.dev()) * int(args.manual_class_id)
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
            print("classes: ", classes)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3 if not args.gray_imgs else 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            my_kwargs=my_kwargs
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    if args.save_img:
        # import imageio
        from PIL import Image
        
        os.makedirs(os.path.join(logger.get_dir(), "imgs"), exist_ok=True)
        for i, img in enumerate(arr):
            
            pil_img = Image.fromarray(img.squeeze())
            pil_img.save(os.path.join(logger.get_dir(), "imgs", f"{i}.png"))
            # imageio.imwrite(os.path.join(logger.get_dir(), "imgs", f"{i}.png"), img)
            
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        log_dir="./log_dir",
        gray_imgs=True, ##### modified by sum: for admitting grayscale images #####
        num_classes=3,
        manual_class_id="",
        gpu_id="",
        save_img=False,
        sim_guided=False, # parameters for simulation-guided sampling
        eta=1.0,
        prop_dir='top',
        save_inter=False,
        interval=10,
        inter_rate=1,
        guidance_type='dps',
        use_normed_grad=False,
        stoptime=0.0,
        use_adjgrad_norm = False,
        sim_type = 'waveguide',
        tsr=100
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
