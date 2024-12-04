# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines
import os

@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )
    print('scene_dir')
    print(cfg.SCENE_DIR)
    sequence_list = test_dataset.sequence_list

    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    output_dir = batch[
        "scene_dir"
    ].split('/')  # which is also cfg.SCENE_DIR for DemoLoader

    output_dir = os.path.join(output_dir[0],output_dir[1])
    print(output_dir)
    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = (
        batch["crop_params"] if batch["crop_params"] is not None else None
    )

    # Cache the original images for visualization, so that we don't need to re-load many times
    original_images = batch["original_images"]

    # Run VGGSfM
    # Both visualization and output writing are performed inside VGGSfMRunner
    vggsfm_runner.run(
        images,
        masks=masks,
        original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir,
        scene_dir = cfg.SCENE_DIR,
        scene_name = cfg.scene_name
    )

    print("Demo Finished Successfully")

    return True


if __name__ == "__main__":
    demo_fn()

