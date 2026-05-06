import argparse
import logging
import os
import random
import time

import imageio
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images, save_videos
from tools.eval import do_evaluation
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from utils.misc import import_str

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)

    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)

    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    for w in ["raw"] + args.train_weathers:
        os.makedirs(os.path.join(log_dir, "images", w), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "buffer_maps", w), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "videos", w), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "metrics", w), exist_ok=True)

    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
                wandb.init(
                    project=args.project,
                    # entity=args.entity,
                    sync_tensorboard=True,
                    settings=wandb.Settings(start_method="fork"),
                    # resume=args.resume_from is not None
                )
                is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")

    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./",
        ["configs", "datasets", "models", "utils", "tools"],
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_weathers = ["raw"] + args.train_weathers

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        weathers=train_weathers,
        dataset=dataset,
        model_config=cfg.model,
        vgg_config=cfg.vgg,
        device=device
    )

    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    # setup optimizer
    trainer.initialize_optimizer()

    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)

    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        # ----------------------------------------------------------------------------
        # ----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            for w in train_weathers:
                with torch.no_grad():
                    render_results = render_images(
                        trainer=trainer,
                        weather=w,
                        dataset=dataset.full_image_set,
                        compute_metrics=True,
                        compute_error_map=cfg.render.vis_error,
                        vis_indices=[
                            vis_timestep * dataset.pixel_source.num_cams + i
                            for i in range(dataset.pixel_source.num_cams)
                        ],
                    )
                if args.enable_wandb:
                    wandb.log(
                        {
                            f"image_metrics/{w}/psnr": render_results["psnr"],
                            f"image_metrics/{w}/ssim": render_results["ssim"],
                            f"image_metrics/{w}/occupied_psnr": render_results["occupied_psnr"],
                            f"image_metrics/{w}/occupied_ssim": render_results["occupied_ssim"],
                        }
                    )
                vis_frame_dict = save_videos(
                    render_results,
                    save_pth=os.path.join(
                        cfg.log_dir, "images", w, f"step_{step}.png"
                    ),  # don't save the video
                    layout=dataset.layout,
                    num_timestamps=1,
                    keys=render_keys,
                    save_seperate_video=cfg.logging.save_seperate_video,
                    num_cams=dataset.pixel_source.num_cams,
                    fps=cfg.render.fps,
                    verbose=False,
                )
                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({f"image_rendering/{w}/" + k: wandb.Image(v)})

            del render_results
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------------------
        # ----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()  # zero grad

        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()

        outputs_dict, image_infos_dict, cam_infos_dict, loss_dict = {}, {}, {}, {}
        image_infos_dict, cam_infos_dict = dataset.train_image_set.next(train_step_camera_downscale, train_weathers)
        for w in train_weathers:
            image_infos, cam_infos = image_infos_dict[w], cam_infos_dict[w]
            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)

            # forward & backward
            outputs = trainer(image_infos, cam_infos, w)
            trainer.update_visibility_filter()

            loss = trainer.compute_losses(
                outputs=outputs,
                image_infos=image_infos,
                cam_infos=cam_infos,
                weather=w
            )

            raw_image_infos, raw_cam_infos = image_infos_dict["raw"], cam_infos_dict["raw"]
            content_loss = trainer.compute_content_loss(
                outputs=outputs,
                image_infos=image_infos,
                cam_infos=cam_infos,
                content_image_infos=raw_image_infos,
                content_cam_infos=raw_cam_infos,
            )
            loss.update(content_loss)

            # check nan or inf
            for k, v in loss.items():
                if torch.isnan(v).any():
                    raise ValueError(f"NaN detected in loss {k} at step {step}")
                if torch.isinf(v).any():
                    raise ValueError(f"Inf detected in loss {k} at step {step}")
            trainer.backward(loss)

            outputs_dict[w] = outputs
            image_infos_dict[w] = image_infos
            cam_infos_dict[w] = cam_infos
            loss_dict[w] = loss

        # after training step
        trainer.optimizer_step()
        trainer.postprocess_per_train_step(step=step)


        # ----------------------------------------------------------------------------
        # -------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            for w in train_weathers:
                metric_dict = trainer.compute_metrics(
                    outputs=outputs_dict[w],
                    image_infos=image_infos_dict[w],
                )

                metric_logger.update(**{f"train_metrics/{w}/" + k: v.item() for k, v in metric_dict.items()})
                metric_logger.update(**{f"losses/{w}/" + k: v.item() for k, v in loss_dict[w].items()})

            metric_logger.update(
                **{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
            metric_logger.update(
                **{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})

        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        # ----------------------------------------------------------------------------
        # ----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
                (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )

        # ----------------------------------------------------------------------------
        # ------------------------    Cache Image Error    ---------------------------
        if (
                step > 0 and trainer.optim_general.cache_buffer_freq > 0
                and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            trainer.set_eval()
            with torch.no_grad():
                for w in train_weathers:
                    logger.info("Caching image error for weather: %s", w)
                    dataset.pixel_source.update_downscale_factor(
                        1 / dataset.pixel_source.buffer_downscale
                    )
                    render_results = render_images(
                        trainer=trainer,
                        dataset=dataset.full_image_set,
                        weather=w
                    )
                    dataset.pixel_source.reset_downscale_factor()
                    dataset.pixel_source.update_image_error_maps(render_results)

                    # save error maps
                    merged_error_video = dataset.pixel_source.get_image_error_video(
                        dataset.layout
                    )
                    imageio.mimsave(
                        os.path.join(
                            cfg.log_dir, "buffer_maps", w, f"buffer_maps_{step}.mp4"
                        ),
                        merged_error_video,
                        fps=cfg.render.fps,
                    )
            logger.info("Done caching rgb error maps")

    logger.info("Training done!")

    for w in train_weathers:
        do_evaluation(
            step=step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            render_keys=render_keys,
            weather=w,
            args=args,
        )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)

    # weather
    parser.add_argument("--train_weathers", help="weather type", nargs='*', type=str)

    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")

    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str,
                        help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="weathercity", type=str,
                        help="wandb run name, also used to enhance log_dir")

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    final_step = main(args)
