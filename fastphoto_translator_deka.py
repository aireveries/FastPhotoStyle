#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch

import process_stylization
from photo_wct import PhotoWCT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_glob", required=True)
    parser.add_argument("--real-glob", required=True)
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--blocksize", default=8, type=int)
    parser.add_argument("--blockidx", default=0, type=int)
    parser.add_argument("--nvar", default=5, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--minsize", default=256, type=int)
    parser.add_argument("--maxsize", default=960, type=int)
    return parser.parse_args()


def load_model(model='./PhotoWCTModels/photo_wct.pth', fast=True, cuda=1):
    """load model, fast=lighter version"""
    # Load model
    p_wct = PhotoWCT()
    p_wct.load_state_dict(torch.load(model))

    if fast:
        from photo_gif import GIFSmoothing
        p_pro = GIFSmoothing(r=35, eps=0.001)
    else:
        from photo_smooth import Propagator
        p_pro = Propagator()
    if cuda:
        p_wct.cuda(0)

    return p_wct, p_pro


def process_image(p_wct, p_pro, content_image_path, content_seg_path=[], style_image_path='./images/style1.png',
                  style_seg_path=[], output_image_path='./results/example1.png', save_intermediate=False, no_post=False,
                  cuda=1, minsize=256, maxsize=960):
    """wrapper function of stylization"""
    process_stylization.stylization(stylization_module=p_wct, smoothing_module=p_pro,
                                    content_image_path=content_image_path, style_image_path=style_image_path,
                                    content_seg_path=content_seg_path, style_seg_path=style_seg_path,
                                    output_image_path=output_image_path, cuda=cuda, save_intermediate=save_intermediate,
                                    no_post=no_post, minsize=minsize, maxsize=maxsize)


def runner(args):
    synthetic_images_list = sorted(list(glob(args.synthetic_glob)))
    print('Synthetic images: {}'.format(len(synthetic_images_list)))
    # real images list
    real_images_list = sorted(list(glob(args.real_glob)))
    print('Real images: {}'.format(len(real_images_list)))

    # load model
    p_wct, p_pro = load_model(fast=args.fast)
    print('Model loaded')

    for sip_ix, synthetic_image_path in enumerate(synthetic_images_list[args.blockidx::args.blocksize]):
        print("Progress: {}/{}".format(sip_ix, len(synthetic_images_list[args.blockidx::args.blocksize])))
        # pick nvar random real images
        real_images = [real_images_list[i] for i in list(np.random.permutation(len(real_images_list))[:args.nvar])]

        for style_ix, real_image_path in enumerate(real_images):
            # image name
            bname = os.path.basename(synthetic_image_path)
            bname = os.path.splitext(bname)
            bname = bname[0] + "-" + str(style_ix) + bname[1]
            bname = os.path.abspath(os.path.join(args.outdir, bname))

            process_image(p_wct, p_pro, content_image_path=synthetic_image_path, style_image_path=real_image_path,
                          output_image_path=bname , no_post=True, minsize=args.minsize, maxsize=args.maxsize)


def main():
    args = parse_args()
    runner(args)


if __name__ == "__main__":
    main()


