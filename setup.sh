### nerf_pretrain

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "pose_2" -O --gt_dir dataset/pose_2 --nerf_transfer --nerf_pretrain'
echo 'python main.py --text "pose_2" -O --gt_dir dataset/pose_2 --nerf_transfer --nerf_pretrain --bound 32'
echo 'python main.py --text "pose_2" -O --gt_dir dataset/pose_2 --nerf_transfer --nerf_pretrain --sd_version waifu'

### nerf_transfer

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "a girl is dancing" -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth'
echo 'python main.py --text "a girl is dancing" -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth --bound 32'
echo 'python main.py --text "a girl is dancing" -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth --sd_version waifu'
echo 'python main.py --text "a girl is dancing, high resolution, unreal engine" -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth --sd_version waifu'
echo 'python main.py --text "a girl is dancing" -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth --manually_lr_scheduler --lr 0.00001'
# echo 'python main.py --text "a girl is dancing, high resolution, unreal engine, " -O --gt_dir dataset/pose_2 --nerf_transfer --load_model ./pretrain_models/pose_2_0030.pth'
