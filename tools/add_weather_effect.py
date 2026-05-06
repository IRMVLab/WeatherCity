import argparse
import logging
import os
import random
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import kornia

import torch.nn.functional as F

from datasets.driving_dataset import DrivingDataset
from models.gaussians.basics import quat_to_rotmat, quat_mult, RGB2SH, SH2RGB
from models.trainers.scene_graph_weather import MultiTrainer_weather
from utils.logging import setup_logging
from utils.misc import import_str

from models.gaussians.basics import *

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

weather_gaussians = None
velocities = None

total_time = 0
total_frames = 0


def set_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup(args):
    cfg = OmegaConf.load(args.config_file)

    weather_cfg = OmegaConf.load(args.effects_config_file)
    cfg.weather = weather_cfg

    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    cfg = OmegaConf.merge(cfg, args_from_cli)

    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for w in args.weathers:
        os.makedirs(os.path.join(log_dir, "images", w), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "videos", w), exist_ok=True)

    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("Command-line args:\n" + "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))

    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}")

    saved_effects_cfg_path = os.path.join(log_dir, "weather_effects_config.yaml")
    with open(saved_effects_cfg_path, "w") as f:
        OmegaConf.save(config=weather_cfg, f=f)
    logger.info(f"Weather effects config saved to {saved_effects_cfg_path}")

    set_seeds(cfg.seed)
    return cfg


def get_particle_spawn_bounds(cfg, device):
    scene_box = torch.tensor(cfg.scene_aabb, device=device, dtype=torch.float32)
    lower = scene_box[0].clone()
    upper = scene_box[1].clone()

    # Adjust spawn bounds to be slightly outside the scene for better particle effects
    lower[0] -= 10.0
    lower[2] -= 2.0
    upper[0] = lower[0] + 60.0
    return lower, upper


def sample_positions_uniform(num_particles, lower, upper, device):
    span = torch.clamp(upper - lower, min=1e-6)
    return torch.rand(num_particles, 3, device=device, dtype=torch.float32) * span + lower


def sample_positions_at_top(num_particles, lower, upper, device, top_thickness_ratio=0.05):
    positions = sample_positions_uniform(num_particles, lower, upper, device)
    z_span = torch.clamp(upper[2] - lower[2], min=1e-6)
    top_thickness = torch.clamp(z_span * top_thickness_ratio, min=1e-3)
    positions[:, 2] = upper[2] - torch.rand(num_particles, device=device, dtype=torch.float32) * top_thickness
    return positions


def sample_wind_velocities(num_particles, weather_params, device):
    vel = weather_params.velocity
    wind_angle = np.deg2rad(weather_params.wind_angle)
    wind_angle_std = np.deg2rad(weather_params.wind_angle_std)
    wind_azimuth = torch.tensor(np.deg2rad(weather_params.wind_azimuth), device=device)

    tilt_angles = torch.normal(
        mean=wind_angle,
        std=wind_angle_std,
        size=(num_particles,),
        device=device
    )

    vx = torch.sin(tilt_angles) * vel * torch.cos(wind_azimuth)
    vy = -torch.cos(tilt_angles) * vel * torch.sin(wind_azimuth)
    vz = torch.sin(tilt_angles) * vel
    return torch.stack([vx, vy, vz], dim=1)


def quats_from_velocity(velocities, device):
    num_particles = velocities.shape[0]
    default_dir = torch.tensor([0.0, 0.0, -1.0], device=device)
    vel_dirs = F.normalize(velocities, dim=1)

    rot_axes = torch.cross(default_dir.expand(num_particles, 3), vel_dirs, dim=1)
    small_axis_mask = torch.norm(rot_axes, dim=1) < 1e-6
    rot_axes = F.normalize(rot_axes, dim=1)
    rot_axes[small_axis_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

    dot = torch.sum(default_dir.expand(num_particles, 3) * vel_dirs, dim=1)
    rot_angles = torch.acos(torch.clamp(dot, -1.0, 1.0))

    return torch.cat([
        torch.cos(rot_angles * 0.5).unsqueeze(1),
        rot_axes * torch.sin(rot_angles * 0.5).unsqueeze(1)
    ], dim=1)


@torch.no_grad()
def respawn_outside_scene(positions, lower, upper):
    outside = ((positions < lower.unsqueeze(0)) | (positions > upper.unsqueeze(0))).any(dim=1)
    if outside.any():
        positions[outside] = sample_positions_at_top(
            outside.sum().item(),
            lower,
            upper,
            positions.device
        )
    return positions

def initialize_weather_particles(trainer, cfg):
    global weather_gaussians, velocities
    device = trainer.device
    weather_type = cfg.weather_type
    weather_params = cfg.weather[weather_type]
    num_particles = int(weather_params.num_particles)
    spawn_lower, spawn_upper = get_particle_spawn_bounds(cfg, device)
    positions = sample_positions_uniform(num_particles, spawn_lower, spawn_upper, device)

    opacities = torch.normal(
        mean=weather_params.opacity_mean,
        std=weather_params.opacity_std,
        size=(num_particles, 1),
        device=device
    )
    opacities = torch.clamp(opacities, min=0.0, max=1.0)

    colors = torch.tensor(weather_params.color, device=device).float().repeat(num_particles, 1)

    if weather_type == 'rain':
        scales = torch.tensor(
            [weather_params.scale_x, weather_params.scale_y, weather_params.scale_z],
            device=device
        ).repeat(num_particles, 1)

        scales[:, 2] *= torch.normal(
            mean=1.0,
            std=weather_params.scale_std,
            size=(num_particles,),
            device=device,
        )

        velocities = sample_wind_velocities(num_particles, weather_params, device)
        quats = quats_from_velocity(velocities, device)

        logger.info(f"Rain: velocity={weather_params.velocity:.2f} m/s, "
                    f"wind_angle={weather_params.wind_angle:.1f}°±{weather_params.wind_angle_std:.1f}°, "
                    f"azimuth={weather_params.wind_azimuth:.1f}°")

    elif weather_type == 'snow':
        num_flakes = num_particles
        positions = positions.repeat(3, 1)
        opacities = opacities.repeat(3, 1)
        colors = colors.repeat(3, 1)

        base_scales = torch.tensor(
            [weather_params.scale_x, weather_params.scale_y, weather_params.scale_z],
            device=device
        ).repeat(num_flakes, 1)

        base_scales *= torch.normal(
            mean=1.0,
            std=weather_params.scale_std,
            size=(num_flakes, 1),
            device=device,
        )

        angle_60 = torch.tensor(np.pi / 3, device=device)
        rot1 = torch.tensor([torch.cos(angle_60 / 2), 0, 0, torch.sin(angle_60 / 2)], device=device)
        rot2 = torch.tensor([torch.cos(-angle_60 / 2), 0, 0, torch.sin(-angle_60 / 2)], device=device)

        base_quats = torch.rand(num_flakes, 4, device=device)
        base_quats = base_quats / torch.norm(base_quats, dim=1, keepdim=True)

        quats1 = quat_mult(base_quats, rot1.expand_as(base_quats))
        quats2 = quat_mult(base_quats, rot2.expand_as(base_quats))

        scales = base_scales.repeat(3, 1)
        quats = torch.cat([base_quats, quats1, quats2], dim=0)

        velocities = sample_wind_velocities(num_flakes, weather_params, device)

        turbulence = weather_params.get('velocity_turbulence', 0.0)
        velocities += (torch.rand(num_flakes, 3, device=device) - 0.5) * turbulence
        velocities = velocities.repeat(3, 1)

        logger.info(f"Snow: velocity={weather_params.velocity:.2f} m/s, "
                    f"wind_angle={weather_params.wind_angle:.1f}°±{weather_params.wind_angle_std:.1f}°, "
                    f"azimuth={weather_params.wind_azimuth:.1f}°")

    weather_gaussians = {
        'means': positions,
        'opacities': opacities,
        'rgbs': colors,
        'scales': scales,
        'quats': quats,
    }
    logger.info(f"Initialized {weather_gaussians['means'].shape[0]} Gaussian particles for {weather_type}.")

def update_weather_particles(delta_time, cam_infos, cfg):
    global weather_gaussians, velocities
    if weather_gaussians is None or velocities is None:
        return

    weather_gaussians['means'] += velocities * delta_time

    spawn_lower, spawn_upper = get_particle_spawn_bounds(cfg, weather_gaussians['means'].device)
    weather_gaussians['means'] = respawn_outside_scene(
        weather_gaussians['means'],
        spawn_lower,
        spawn_upper
    )

def collect_gaussians(
        self,
        cam: dataclass_camera,
        weather: str
) -> dataclass_gs:
    gs_dict = {
        "_means": [],
        "_scales": [],
        "_quats": [],
        "_rgbs": [],
        "_opacities": [],
        "class_labels": [],
    }
    
    # if "Background" in self.gaussian_classes.keys():
    #     self.gaussian_classes.pop("Background")
    # print(f"gaussian_classes: {self.gaussian_classes.keys()}")
    # self.gaussian_classes = {k: v for k, v in self.gaussian_classes.items() if k == "Background"}
    for class_name in self.gaussian_classes.keys():
        gs = self.models[class_name].get_gaussians(cam, weather)
        if gs is None:
            continue

        # collect gaussians
        gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name],
                                        device=self.device)
        for k, _ in gs.items():
            gs_dict[k].append(gs[k])

    for k, v in gs_dict.items():
        gs_dict[k] = torch.cat(v, dim=0)

    # get the class labels
    self.pts_labels = gs_dict.pop("class_labels")
    if self.render_dynamic_mask:
        self.dynamic_pts_mask = (self.pts_labels != 0).float()

    gaussians = dataclass_gs(
        _means=gs_dict["_means"],
        _scales=gs_dict["_scales"],
        _quats=gs_dict["_quats"],
        _rgbs=gs_dict["_rgbs"],
        _opacities=gs_dict["_opacities"],
        detach_keys=[],  # if "means" in detach_keys, then the means will be detached
        extras=None  # to save some extra information (TODO) more flexible way
    )

    return gaussians

def render_gauss(
        self,
        image_infos,
        camera_infos,
        weather_gauss,
        weather,
        novel_view=False):
    normed_time = image_infos["normed_time"].flatten()[0]
    self.cur_frame = torch.argmin(
        torch.abs(self.normalized_timestamps - normed_time)
    )

    for model in self.models.values():
        if hasattr(model, 'in_test_set'):
            model.in_test_set = self.in_test_set

    for class_name in self.gaussian_classes.keys():
        model = self.models[class_name]
        if hasattr(model, 'set_cur_frame'):
            model.set_cur_frame(self.cur_frame)

    processed_cam = self.process_camera(
        camera_infos=camera_infos,
        image_ids=image_infos["img_idx"].flatten()[0],
        novel_view=novel_view
    )

    vanilla_gs = collect_gaussians(
        self,
        cam=processed_cam,
        weather=weather,
        image_ids=image_infos["img_idx"].flatten()[0]
    )

    if weather_gauss is not None:
        gs = dataclass_gs(
            _means=torch.cat([vanilla_gs.means, weather_gauss['means']], dim=0),
            _scales=torch.cat([vanilla_gs.scales, weather_gauss['scales']], dim=0),
            _quats=torch.cat([vanilla_gs.quats, weather_gauss['quats']], dim=0),
            _rgbs=torch.cat([vanilla_gs.rgbs, weather_gauss['rgbs']], dim=0),
            _opacities=torch.cat([vanilla_gs.opacities, weather_gauss['opacities']], dim=0),
            detach_keys=[],
            extras=None
        )
    else:
        gs = vanilla_gs

    t1 = time.time()

    outputs, render_fn = self.render_gaussians(
        gs=gs,
        cam=processed_cam,
        near_plane=self.render_cfg.near_plane,
        far_plane=self.render_cfg.far_plane,
        render_mode="RGB+ED",
        radius_clip=self.render_cfg.get('radius_clip', 0.)
    )

    t2 = time.time()
    global total_time, total_frames
    total_time += (t2 - t1)
    total_frames += 1

    sky_model = self.models[f'Sky.{weather}']
    outputs["rgb_sky"] = sky_model(image_infos)
    outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])

    outputs["rgb"] = self.affine_transformation(
        outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]),
        # outputs["rgb_gaussians"],
        image_infos
    )

    return outputs


def save_depth_maps(depth_tensor):
    """
    Save depth map (HxWx1) as both:
      - float32 TIFF (metric depth)
      - uint16 PNG (visualization)
    """
    depth = depth_tensor.squeeze(-1).detach().cpu().numpy().astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        dmin, dmax = np.percentile(depth[valid], [1, 99])
        if dmax <= dmin:
            dmin, dmax = depth[valid].min(), depth[valid].max()
        depth_vis = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    else:
        depth_vis = np.zeros_like(depth)

    depth_png = (depth_vis * 65535).astype(np.uint16)  # [H, W], uint16
    imageio.imwrite(f"depth_vis.png", depth_png)


def render_weather_layer(trainer, image_infos, cam_infos, cfg):
    weather_type = cfg.weather_type
    params = cfg.weather.weather_type

    with torch.no_grad():
        scene_outputs = render_gauss(
            trainer,
            image_infos,
            cam_infos,
            weather_gaussians,
            'raw' if weather_type == 'fog' else weather_type
        )
        C_render = scene_outputs['rgb']
        D_ref = scene_outputs['depth']
        sky_mask = image_infos['sky_masks'].bool()

    if weather_type == 'fog':
        depth = D_ref.clone()
        mask_expanded = sky_mask.bool().unsqueeze(-1)

        non_sky_depths = depth[~mask_expanded.expand_as(depth)]
        if non_sky_depths.numel() > 0:
            max_depth = non_sky_depths.max()
        else:
            max_depth = 1.0
        depth[mask_expanded.expand_as(depth)] = max_depth * 2.0

        if cfg.weather.fog.smooth:
            depth_bchw = depth.permute(2, 0, 1).unsqueeze(0).float()
            depth_smooth_bchw = kornia.filters.bilateral_blur(
                depth_bchw,
                kernel_size=(13, 13),
                sigma_color=0.7,
                sigma_space=(5.0, 5.0)
            )
            depth_smooth_bchw = kornia.filters.gaussian_blur2d(
                depth_smooth_bchw,
                kernel_size=(13, 13),
                sigma=(5, 5)
            )
            depth_smooth = depth_smooth_bchw.squeeze(0).permute(1, 2, 0)
        else:
            depth_smooth = depth

        I_style = params.intensity
        alpha_style = torch.clamp(
            1.0 - torch.exp(-I_style * depth_smooth),
            min=0.0,
            max=1.0
        )

        C_fog = torch.tensor(
            params.color,
            device=trainer.device,
            dtype=C_render.dtype
        )
        C_render_fog = C_fog * alpha_style + C_render * (1.0 - alpha_style)

        return torch.clamp(C_render_fog, 0, 1), C_render
    else:
        return torch.clamp(C_render, 0, 1), C_render


def cumulative_weather_simulation(trainer, cfg):
    params = cfg.weather.snow

    logger.info(f"Applying cumulative snow cover...")

    background_model = trainer.models.get("Background")
    if background_model:
        with torch.no_grad():
            quats = background_model._quats
            rot_matrices = quat_to_rotmat(quats)
            normals = rot_matrices[:, :, 2]

            up_vector = torch.tensor([0.0, 0.0, 1.0], device=trainer.device)
            dot_product = torch.sum(normals * up_vector, dim=1)

            upward_mask = dot_product > params.upward_normal_threshold
            snow_indices = torch.where(upward_mask)[0]

            if snow_indices.numel() > 0:
                white_sh = RGB2SH(torch.tensor([1.0, 1.0, 1.0], device=trainer.device))
                background_model._features_dc.data[snow_indices] = white_sh

                current_opacities = torch.sigmoid(background_model._opacities.data[snow_indices])
                new_opacities = torch.clamp(
                    current_opacities * params.snow_cover_opacity_increase,
                    0, 0.99
                )
                background_model._opacities.data[snow_indices] = torch.logit(new_opacities)

                background_model._means.data[snow_indices, 2] += params.snow_cover_elevation

                logger.info(f"Applied snow cover to {snow_indices.numel()} Gaussians.")
            else:
                logger.warning("No upward-facing surfaces found to apply snow cover.")
    else:
        logger.warning("Background model not found, cannot apply snow cover.")


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
    cfg.load_weathers = list({('raw' if w == 'fog' else w) for w in args.weathers} | {'raw'})

    trainer = import_str(cfg.trainer.type)(
        dataset=dataset,
        weathers=cfg.load_weathers,
        model_config=cfg.model,
        device=device,
        **cfg.trainer
    )
    trainer.set_eval()

    if args.load_from is None:
        logger.error("A checkpoint must be provided via --load_from.")
        return

    trainer.resume_from_checkpoint(ckpt_path=args.load_from, load_only_model=True)
    logger.info(f"Successfully loaded model from {args.load_from}")

    cfg.scene_aabb = dataset.get_aabb().reshape(2, 3).cpu().numpy().tolist()

    for weather_type in args.weathers:
        global weather_gaussians
        weather_gaussians = None

        cfg.weather_type = weather_type
        logger.info(f"Processing weather type: {weather_type}")

        if weather_type in ['rain', 'snow']:
            initialize_weather_particles(trainer, cfg)

        if weather_type == 'snow' and cfg.weather.snow.get('apply_cumulative', False):
            clean_trainer_state = trainer.state_dict()
            cumulative_trainer = import_str(cfg.trainer.type)(
                weathers=cfg.load_weathers,
                **cfg.trainer,
                num_timesteps=dataset.num_img_timesteps,
                model_config=cfg.model,
                num_train_images=len(dataset.train_image_set),
                num_full_images=len(dataset.full_image_set),
                test_set_indices=dataset.test_timesteps,
                scene_aabb=dataset.get_aabb().reshape(2, 3),
                device=device
            )
            cumulative_trainer.load_state_dict(clean_trainer_state)
            cumulative_weather_simulation(cumulative_trainer, cfg)
        else:
            cumulative_trainer = trainer

        output_video_dir = os.path.join(cfg.log_dir, "videos", weather_type)
        output_image_dir = os.path.join(cfg.log_dir, "images", weather_type)
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(os.path.join(output_image_dir, "effects"), exist_ok=True)
        os.makedirs(os.path.join(output_image_dir, "raw"), exist_ok=True)

        effects_writer = imageio.get_writer(
            os.path.join(output_video_dir, "effects.mp4"),
            fps=cfg.render.fps
        )
        raw_writer = imageio.get_writer(
            os.path.join(output_video_dir, "raw.mp4"),
            fps=cfg.render.fps
        )

        num_frames_to_render = dataset.num_img_timesteps
        for i in tqdm(range(num_frames_to_render), desc=f"Rendering {weather_type} frames"):
            weather = 'raw' if weather_type == 'fog' else weather_type
            image_infos, cam_infos = dataset.full_image_set.get_image(
                i * dataset.pixel_source.num_cams,
                1.0,
                weathers=[weather]
            )
            image_infos, cam_infos = image_infos[weather], cam_infos[weather]

            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v.cuda()
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v.cuda()

            if weather_type in ['rain', 'snow']:
                update_weather_particles(
                    delta_time=1.0 / cfg.render.fps,
                    cam_infos=cam_infos,
                    cfg=cfg
                )

            effects_frame, raw_frame = render_weather_layer(
                cumulative_trainer,
                image_infos,
                cam_infos,
                cfg
            )

            effects_frame_np = (effects_frame.cpu().numpy() * 255).astype(np.uint8)
            raw_frame_np = (raw_frame.cpu().numpy() * 255).astype(np.uint8)

            effects_writer.append_data(effects_frame_np)
            raw_writer.append_data(raw_frame_np)

            imageio.imwrite(
                os.path.join(output_image_dir, "effects", f"effects_{i:03d}.png"),
                effects_frame_np
            )
            imageio.imwrite(
                os.path.join(output_image_dir, "raw", f"raw_{i:03d}.png"),
                raw_frame_np
            )

        effects_writer.close()
        raw_writer.close()
        logger.info(f"Video for {weather_type} saved to {output_video_dir}")
        logger.info(f"Images for {weather_type} saved to {output_image_dir}")

    print(f"\nAverage rendering time per frame: {total_time / total_frames:.5f} seconds.")
    print(f"FPS: {total_frames / total_time:.5f} frames per second.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Add weather effects to a pre-trained 3D scene model and render a video."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the model's original configuration file."
    )
    parser.add_argument(
        "--effects_config_file",
        type=str,
        default="configs/weather_params.yaml",
        help="Path to the weather effects parameters yaml file."
    )
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        type=str,
        help="Root directory for output."
    )
    parser.add_argument(
        "--weathers",
        type=str,
        nargs='+',
        required=True,
        choices=['rain', 'snow', 'fog'],
        help="A list of weather effects to apply."
    )
    parser.add_argument(
        "--load_from",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint."
    )
    parser.add_argument(
        "--project",
        default="weather_modification",
        type=str,
        help="Project name for organizing outputs."
    )
    parser.add_argument(
        "--run_name",
        default=f"run_{current_time}",
        type=str,
        help="Run name for this specific execution."
    )
    parser.add_argument(
        "opts",
        help="Modify config options from the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    main(args)
