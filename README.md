<p align="center">
  <img src="assets/WeatherCity-head.gif" width="99%" style="max-width: 100%; height: auto;" />
</p>

<p align="center">
  <b>WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-blue" height="26"/>
  <a href="https://arxiv.org/abs/2602.22096"><img src="https://img.shields.io/badge/arXiv-2602.22096-b31b1b" height="26"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" height="26"/></a>
</p>

This repository contains the official implementation of **WeatherCity**, a 4D urban scene reconstruction and controllable weather editing framework for autonomous driving simulation.

## Installation
WeatherCity is built on top of [DriveStudio](https://github.com/ziyc/drivestudio) and follows the same environment setup as DriveStudio. For DriveStudio installation, dataset downloading, masks, SMPL processing, and the original OmniRe pipeline, please refer to the [DriveStudio README](https://github.com/ziyc/drivestudio) and dataset docs such as [Waymo](docs/Waymo.md) and [NuScenes](docs/NuScenes.md).

WeatherCity additionally uses VGG features for content consistency loss. Make sure the VGG 19 weight file `vgg_conv.pth` configured in [configs/weathercity.yaml](configs/weathercity.yaml) exists.

## Data Preparation

First preprocess the dataset with DriveStudio. For example, Waymo scenes should be processed following [docs/Waymo.md](docs/Waymo.md), including images, cameras, LiDAR, masks, instances, and human pose files.

After DriveStudio preprocessing, WeatherCity expects images to be organized by weather under `images/`:

```text
data/waymo/processed/training/<scene_idx>/
├── images/
│   ├── raw/
│   │   ├── 000_0.jpg
│   │   ├── 001_0.jpg
│   │   └── ...
│   ├── rain/
│   │   ├── 000_0.jpg
│   │   ├── 001_0.jpg
│   │   └── ...
│   └── snow/
│       ├── 000_0.jpg
│       ├── 001_0.jpg
│       └── ...
├── dynamic_masks/
├── fine_dynamic_masks/
├── sky_masks/
├── lidar/
├── instances/
└── ...
```
where `raw/` contains the original images and `rain/`, `snow/`, etc. contain the edited weather images by [Qwen-Image](https://github.com/QwenLM/Qwen-Image).

Then make sure the dataset config loads the same edited weather folders:

```yaml
# configs/datasets/waymo/1cams.yaml
data:
  pixel_source:
    load_weather: ["rain", "snow"]
```

`raw` is added automatically by the dataloader, so `load_weather` should only contain edited weather names.

## Training

Set the project path and run WeatherCity training with the weather conditions you want to optimize:

```sh
export PYTHONPATH=$(pwd)
python tools/train_feature.py \
    --config_file configs/weathercity.yaml \
    --output_root output \
    --train_weathers rain snow \
    --project waymo_weathercity \
    --run_name 788_1_cam_000_029 \
    dataset=waymo/1cams \
    data.scene_idx=788 \
    data.start_timestep=0 \
    data.end_timestep=29
```

- `--train_weathers rain snow` must match the edited folders under `images/` and `load_weather` in the dataset config.

Training outputs are saved to:

```text
output/<project>/<run_name>/
```

## Evaluation

Render and evaluate a trained checkpoint:

```sh
export PYTHONPATH=$(pwd)
python tools/eval.py \
    --resume_from <path_to_checkpoint> \
    --eval_weathers rain snow
```
Evaluation renders `raw` plus the listed weather conditions. 

## Dynamic Weather Effects

After training the multi-weather scene, use the dynamic weather renderer to add controllable rain, snow, or fog:

```sh
export PYTHONPATH=$(pwd)
python tools/add_weather_effects.py \
    --config_file configs/weathercity.yaml \
    --effects_config_file configs/weather_effects.yaml \
    --output_root output_effects \
    --project waymo_weathercity_effects \
    --weathers rain snow fog \
    --run_name 788_1_cam_000_029_effects \
    --load_from <path_to_checkpoint> \
    dataset=waymo/1cams \
    data.scene_idx=788 \
    data.start_timestep=0 \
    data.end_timestep=29
```

Weather effect parameters are controlled in [configs/weather_effects.yaml](configs/weather_effects.yaml).

Rendered videos and frames are saved to:

```text
output_effects/<project>/<run_name>/
```

## Adding New Weather

To add a new weather condition, follow these steps:

1. Generate weather-edited images with the same filenames as the raw images.
2. Put them under the processed scene:

```text
/data/waymo/processed/training/<scene_id>/images/<new_weather>/
├── 000_0.jpg
├── 001_0.jpg
└── ...
```

3. Add the new weather name to the dataset config:

```yaml
# configs/datasets/waymo/1cams.yaml
data:
  pixel_source:
    load_weather: ["rain", "snow", "<new_weather>"]
```

4. Add an optimizer entry for the new weather decoder in [configs/weathercity.yaml](configs/weathercity.yaml):

```yaml
trainer:
  gaussian_optim_general_cfg:
    mlp.<new_weather>:
      lr: 0.0001
```

5. Include the new weather at training and evaluation time.

## Acknowledgments

WeatherCity is based on [DriveStudio](https://github.com/ziyc/drivestudio) and we also use [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian rasterization and text-guided image editing models ([Qwen-Image](https://github.com/QwenLM/Qwen-Image)) for weather background generation. The evaluation metrics are referenced from [WeatherEdit](https://github.com/Jumponthemoon/WeatherEdit).

Please also cite DriveStudio/OmniRe if you use this codebase.

## Citation

```bibtex
@misc{wu2026weathercityurbanscenereconstruction,
      title={WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation}, 
      author={Wenhua Wu and Huai Guan and Zhe Liu and Hesheng Wang},
      year={2026},
      eprint={2602.22096},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22096}, 
}
```
