import os
import argparse

parameter_dict = {
    'text' : [
        'Elsa in modern disney style', 
        'Simba in modern disney style', 
        'Mickey mouse in modern disney style', 
    ],
    # nerf rendering
    'lambda_entropy' : [1, 0.1, 0.001, 0.00001],
    'lambda_opacity' : [0.1, 0.01],
    'lambda_orient' : [0.001, 0.1],
    'lambda_smooth' : [0.1, 0.01],
        
    'lr' : [0.001, 0.0001, 0.00001],
}

def run_command(cmd):
    print('to run: ', cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', default=None, help="select one parameter to tune")
    parser.add_argument('--seed', type=int, default=0, help="select one parameter to tune")
    parser.add_argument('--pose', type=int, default=1, help="select one parameter to tune")
    opt = parser.parse_args()
        

    if opt.tune == 'lambda_orient':
        # 1e-4, 0, 1e-2, 0 # default
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name lambda_orient_001 --text "Mickey mouse in modern disney style" --transfer_type original --lambda_orient 0.01 --seed {opt.seed}')
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name lambda_orient_005 --text "Mickey mouse in modern disney style" --transfer_type original --lambda_orient 0.05 --seed {opt.seed}')
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name lambda_orient_05 --text "Mickey mouse in modern disney style" --transfer_type original --lambda_orient 0.5 --seed {opt.seed}')
            
    elif opt.tune == 'prompt':
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name prompt_elsa --text "Elsa in modern disney style" --transfer_type original --seed {opt.seed}')
    
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name prompt_simba --text "Simba in modern disney style" --transfer_type original --seed {opt.seed}')
    
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name prompt_mickey --text "Mickey mouse in modern disney style" --transfer_type original --seed {opt.seed}')
             
    elif opt.tune == 'sd_version':
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version CompVis --ex_name sd_version_compvis --text "Mickey mouse in modern disney style" --transfer_type original --seed {opt.seed}')
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version waifu --ex_name sd_version_waifu --text "Mickey mouse in modern disney style" --transfer_type original --seed {opt.seed}')
        
    elif opt.tune == 'pose':
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth -O3 --sd_version disney --ex_name pose_non --text "Mickey mouse in modern disney style" --transfer_type original --seed {opt.seed}')
        
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_2_0050.pth --reload_model -O3 --sd_version disney --ex_name pose_2 --text "Mickey mouse in modern disney style" --transfer_type original --seed {opt.seed}')
        
    elif opt.tune == 'pose_prompt':
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth -O3 --sd_version disney --ex_name pose_1_prompt_Simba --text "Simba in modern disney style" --transfer_type original --seed {opt.seed}')
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_2_0050.pth --reload_model -O3 --sd_version disney --ex_name pose_2_prompt_Simba --text "Simba in modern disney style" --transfer_type original --seed {opt.seed}')
        
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth -O3 --sd_version disney --ex_name pose_1_prompt_Elsa --text "Elsa in modern disney style" --transfer_type original --seed {opt.seed}')
        
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_2_0050.pth --reload_model -O3 --sd_version disney --ex_name pose_2_prompt_Elsa --text "Elsa in modern disney style" --transfer_type original --seed {opt.seed}')
        
    elif opt.tune == 'cycle_tuning':
        run_command(f'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --reload_model -O3 --sd_version disney --ex_name cycle_tuning --text "Mickey mouse in modern disney style" --transfer_type cycle_tuning --seed {opt.seed}')
    
    elif opt.tune == 'tune_sd_at_start':
        run_command(f'python main.py -O --text "Mickey mouse in modern disney style" --ex_name tune_sd_at_start_baseline')
        run_command(f'python main.py -O --text "Mickey mouse in modern disney style" --transfer_type dream_booth --ex_name tune_sd_at_start --tune_sd_at_start --gt_images_path dataset/mickey --class_prompt Mickey')
        
        run_command(f'python main.py -O --text "flower, photo, realistic" --ex_name tune_sd_at_start_baseline')
        run_command(f'python main.py -O --text "flower, photo, realistic" --transfer_type dream_booth --ex_name tune_sd_at_start --tune_sd_at_start --gt_images_path dataset/flower --class_prompt flower')
