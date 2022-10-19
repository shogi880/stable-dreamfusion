### nerf_pretrain

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "pose3" -O --gt_dir dataset/pose_3 --nerf_trasfer --nerf_pretrain'

### nerf_transfer

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate
cd code/stable-dreamfusion 
echo 'python main.py --text "a boy" -O --gt_dir dataset/pose_2 --nerf_trasfer --load_model pretrain_models/pose_2_0010.pth'
nohup python main.py --text "a man" -O --gt_dir dataset/pose_3 --nerf_trasfer --load_model pretrain_models/pose_3_0150.pth --seed 150 &!
