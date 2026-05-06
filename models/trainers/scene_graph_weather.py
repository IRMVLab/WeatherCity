import logging
import os
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torchvision import transforms

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.VGG import VGG, VGG_BY_BLOCKS
from utils.geometry import uniform_sample_sphere
from utils.misc import import_str

logger = logging.getLogger()


class MultiTrainer_weather(BasicTrainer):
    def __init__(
            self,
            vgg_config: OmegaConf = None,
            **kwargs
    ):
        self.vgg_config = vgg_config
        super().__init__(**kwargs)
        self.render_each_class = True
        self.misc_classes_keys = [
            'Affine', 'CamPose', 'CamPosePerturb'
        ]

        if self.vgg_config:
            # get VGG network (from https://github.com/leongatys/PytorchNeuralStyleTransfer)
            vgg = VGG()
            object.__setattr__(self, 'vgg', vgg)
            # Available here:https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download
            vgg_weights_path = Path(self.vgg_config.model_weight).absolute()
            logger.info(f"Loading VGG weights from {vgg_weights_path}")
            self.vgg.load_state_dict(torch.load(vgg_weights_path))
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg.to(self.device)

            wmeanstd = 1000
            weights_Gram_matrices = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
            self.content_weights = [self.vgg_config.content_weight]
            weights_layers_means = [w * n ** 2 * wmeanstd / 100 for n, w in
                                    zip([64, 128, 256, 512, 512], weights_Gram_matrices)]
            weights_layers_stds = weights_layers_means
            self.style_weights = (weights_Gram_matrices, weights_layers_means, weights_layers_stds)

            self.VGG_prep_tensor_img = transforms.Compose([
                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),  # subtract imagenet mean
                transforms.Lambda(lambda x: x.mul_(255)),
            ])

    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)

    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes

        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)

            if class_name in self.gaussian_classes.keys():
                logger.info(f"Initializing {class_name} model")
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    weathers=self.weathers,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )

            if class_name in self.misc_classes_keys:
                logger.info(f"Initializing {class_name} model")
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model

        # initialize Sky model
        if "Sky" in self.model_config:
            model_cfg = self.model_config["Sky"]
            for w in self.weathers:
                class_name = f"Sky.{w}"
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)
                self.models[class_name] = model

        logger.info(f"Initialized models: {self.models.keys()}")

        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)

    def safe_init_models(
            self,
            model: torch.nn.Module,
            instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
            self,
            dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}

        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = []

        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]

            empty = False
            if class_name == 'Background':
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None

                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0:  # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3  # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0:  # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))

                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0)
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]

                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)

                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )

                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )

            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )

            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )

            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )

            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")

        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")

        logger.info(f"Initialized gaussians from pcd")

    def forward(
            self,
            image_infos: Dict[str, torch.Tensor],
            camera_infos: Dict[str, torch.Tensor],
            weather: str,
            novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)

        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            weather=weather,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )

        # render sky
        sky_model = self.models[f'Sky.{weather}']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])

        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )

        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name + "_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name + "_opacity"] = sep_opacity
                    outputs[class_name + "_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth

        return outputs

    def compute_losses(
            self,
            outputs: Dict[str, torch.Tensor],
            image_infos: Dict[str, torch.Tensor],
            cam_infos: Dict[str, torch.Tensor],
            weather: str
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos, weather)

        return loss_dict

    def compute_content_loss(
            self,
            outputs: Dict[str, torch.Tensor],
            image_infos: Dict[str, torch.Tensor],
            cam_infos: Dict[str, torch.Tensor],
            content_image_infos: Dict[str, torch.Tensor],
            content_cam_infos: Dict[str, torch.Tensor],
            save_vgg_feature_map: bool = False,
            save_name: str = "vgg"
    ) -> Dict[str, torch.Tensor]:
        assert self.vgg_config is not None, "VGG config is not provided for content loss computation"

        output_img_h = cam_infos["height"].item()
        output_img_w = cam_infos["width"].item()
        content_img_h = content_cam_infos["height"].item()
        content_img_w = content_cam_infos["width"].item()

        assert output_img_h == content_img_h and output_img_w == content_img_w, \
            f"Output image size {output_img_w}x{output_img_h} does not match content image size {content_img_w}x{content_img_h}"

        image = outputs["rgb"]
        if self.training:
            image.retain_grad()
        rgb_pred = self.VGG_prep_tensor_img(image.permute(2, 0, 1)).unsqueeze(0)  # HWC -> BCHW
        if self.training:
            rgb_pred.retain_grad()

        with torch.no_grad():
            rgb_gt = self.VGG_prep_tensor_img(
                content_image_infos["pixels"].permute(2, 0, 1)
            ).unsqueeze(0)  # HWC -> BCHW

            # Compute content targets
            vgg_blocks_content_gt = VGG_BY_BLOCKS(
                self.vgg, rgb_gt, self.vgg_config.style_layers,
                content_layers=self.vgg_config.content_layers,
                verbose_mode=False
            )
            content_gt_by_blocks = vgg_blocks_content_gt.compute_content_layer_by_blocks()

        vgg_blocks_content_pred = VGG_BY_BLOCKS(self.vgg, rgb_pred, self.vgg_config.style_layers,
                                           content_layers=self.vgg_config.content_layers,
                                           verbose_mode=False)
        if self.training:
            loss = vgg_blocks_content_pred.global_content_loss_with_gradient(content_gt_by_blocks, self.content_weights)
        else:
            loss = vgg_blocks_content_pred.global_content_loss(content_gt_by_blocks, self.content_weights)
            
        if self.training:
            sky_mask = image_infos["sky_masks"].bool()
            print(f"rgb_pred.grad shape: {rgb_pred.grad.shape}, sky_mask shape: {sky_mask.shape}")
            rgb_pred.grad = rgb_pred.grad * (~sky_mask)
            torch.autograd.backward(rgb_pred, grad_tensors=rgb_pred.grad, retain_graph=True)

        if save_vgg_feature_map:
            os.mkdir('vgg_vis') if not os.path.exists('vgg_vis') else None
            vgg_blocks_content_gt.visualize_content_layer_stacked(save_path=f'vgg_vis/{save_name}_gt.png')
            vgg_blocks_content_pred.visualize_content_layer_stacked(save_path=f'vgg_vis/{save_name}_pred.png')

        del vgg_blocks_content_pred, content_gt_by_blocks, vgg_blocks_content_gt, rgb_gt, rgb_pred
        torch.cuda.empty_cache()

        return {"content_loss": loss.detach()}

    def compute_metrics(
            self,
            outputs: Dict[str, torch.Tensor],
            image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)

        return metric_dict
