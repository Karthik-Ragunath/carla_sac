# CARLA-PPO-Algorithm

This github repository is aimed at research to perform end-to-end motion planning for Autonomous Driving based on PPO (and SAC) algorithms.

## STEPS

1. Create a conda environment and activate it

```
conda create -n carla_ppo python=3.8
conda activate carla_ppo
```

2. Install pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

3. Install other required packages after 
```
pip install -r requirements.txt
```

4. Download CARLA 0.9.13 package and start in port 2021 for training
Refer - https://carla.readthedocs.io/en/latest/start_quickstart/#carla-0912

5. Setup training and inference configs - Default configs provided in
```
env_config.py
```

6. Start Training
```
python  carla_ppo_train.py --device_id <device_id> --img-stack <image_stack_dimension> \
        --log_seed <log_seed No.> --running_score <max reward to stop training> \
        --context <train_context_id> --num_steps_per_episode <number of steps per episode>
```

Example:
```
python  carla_ppo_train.py --device_id 0 --img-stack 30 \
        --log_seed 1 --running_score 30000 \
        --context train_1 --num_steps_per_episode 150
```

7. Perform Inference
```
python  carla_ppo_inference.py --device_id <device_id> --img-stack <image_stack_dimension> \
        --log_seed <log_seed No.>  --context <train_context_id> \
        --num_steps_per_episode <number of steps per episode>
```

Example
```
python  carla_ppo_train.py --device_id 0 --img-stack 30 \
        --log_seed 1 --context train_1 \
        --num_steps_per_episode 150
```

8. Generate Inference Videos

```
python util_video_visualization.py  --image_dir <image_dir where inference images are stored> \
      --fps <frames per second> --output_dir <inference video output directory>
```

```
python util_video_visualization.py  --image_dir visualization_inference_3/reward \
      --fps 10 --output_dir visualization_videos/reward
```

### Example: Inference on model trained for 5 hrs on Zero Traffic Simulation

[!Alt text](https://github.com/Karthik-Ragunath/carla_sac/blob/feature/ppo_carla/assets/visualization_inference_4_0.mp4 "ZeroTraffic - 5 hours Training")