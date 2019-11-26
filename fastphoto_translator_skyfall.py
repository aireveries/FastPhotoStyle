#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import process_stylization
from photo_wct import PhotoWCT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-folder", required=True)
    parser.add_argument("--real-glob", required=True)
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--split_names", default=["train", "test", "val"], nargs='+')
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

def runner(args, partition):
    synth_path = Path(args.synthetic_folder)

    # open synthetic annotation file and list all images
    ann_file = f"{synth_path / 'annotations/instances_{}.json'}".format(partition)
    with open(ann_file, "r") as f:
        ann = json.load(f)
    print(os.listdir('/mnt/'))

    synthetic_images_list = list(sorted(img['path'] for img in ann['images']))
    print('Synthetic images: {}'.format(len(synthetic_images_list)))
    # real images list
    real_images_list = sorted(list(glob(args.real_glob)))
    print('Real images: {}'.format(len(real_images_list)))

    # load model
    p_wct, p_pro = load_model(fast=args.fast)
    print('Model loaded')

    # create directories
    labels_loc = os.path.join(args.outdir, "labels", partition)
    images_loc = os.path.join(args.outdir, "images", partition)
    annotations_loc = os.path.join(args.outdir, "annotations")
    metadata_loc = os.path.join(args.outdir, "munit_metadata")

    # Updated annotations json
    os.makedirs(labels_loc, exist_ok=True)
    os.makedirs(images_loc, exist_ok=True)
    os.makedirs(annotations_loc, exist_ok=True)

    # for each synthetic image, we create nvar translation. need to recreate the json file
    if args.blockidx == 0:
        # Replicate images, annotations
        n_images, n_annotations = len(ann["images"]), len(ann["annotations"])

        all_images = [None] * n_images * args.nvar
        all_anns = [None] * n_annotations * args.nvar

        # image_id is not necessarily equal to its array index
        image_id_to_arr_idx_map = {}
        for ix, ann_img in enumerate(ann['images']):
            image_id_to_arr_idx_map[ann_img['id']] = ix + 1

        for i in range(args.nvar):
            for j in range(n_images):
                all_images[n_images * i + j] = copy.deepcopy(ann["images"][j])
                all_images[n_images * i + j]["id"] = n_images * i + j + 1

                dirname, bname = ann["images"][j]["path"].split('/')[-2:]
                bname = os.path.splitext(bname)
                bname = dirname + '-' + bname[0] + "-{}".format(i) + bname[1]
                bname = os.path.abspath(os.path.join(images_loc, bname))

                all_images[n_images * i + j]["file_name"] = bname
                all_images[n_images * i + j]["coco_url"] = bname
                all_images[n_images * i + j]["path"] = bname

        for i in range(args.nvar):
            for j in range(n_annotations):
                all_anns[n_annotations * i + j] = copy.deepcopy(ann["annotations"][j])
                all_anns[n_annotations * i + j]["id"] = n_annotations * i + j + len(all_images) + 1
                all_anns[n_annotations * i + j]["image_id"] = n_images * i + image_id_to_arr_idx_map[
                    ann["annotations"][j]["image_id"]]

        ann["images"] = all_images
        ann["annotations"] = all_anns

        annotations_path = os.path.join(annotations_loc, "instances_{}.json".format(partition))
        with open(annotations_path, "w") as f:
            json.dump(ann, f)

    for sip_ix, synthetic_image_path in enumerate(synthetic_images_list[args.blockidx::args.blocksize]):
        print("Progress: {}/{}".format(sip_ix, len(synthetic_images_list[args.blockidx::args.blocksize])))
        # pick two random real image
        real_images = [real_images_list[i] for i in list(np.random.permutation(len(real_images_list))[:args.nvar])]

        for style_ix, real_image_path in enumerate(real_images):
            # image name
            dirname, bname = synthetic_image_path.split('/')[-2:]
            bname = os.path.splitext(bname)
            bname = dirname + '-' + bname[0] + "-" + str(style_ix) + bname[1]
            bname = os.path.abspath(os.path.join(images_loc, bname))

            process_image(p_wct, p_pro, content_image_path=synthetic_image_path, style_image_path=real_image_path,
                          output_image_path=bname , no_post=True, minsize=args.minsize, maxsize=args.maxsize)

def main():
    args = parse_args()
    for split in args.split_names:
        runner(args, split)


if __name__ == "__main__":
    main()


