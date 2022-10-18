source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5

source ~/.zshrc
conda deactivate

source ~/venv/dreamfusion/bin/activate

echo 'python main.py --text "A hamburger" -O'