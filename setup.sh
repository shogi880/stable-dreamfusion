### nerf_pretrain

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "pose_3_background" -O --gt_dir dataset/pose_3_black_background --nerf_transfer --nerf_pretrain'

### nerf_transfer

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "a girl is dancing" -O --gt_dir dataset/pose_3 --nerf_transfer --load_model pretrain_models/pose_3_0050.pth'
