#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100-32:2
#SBATCH --output=/jet/home/dsouzare/assn4_out.txt
#SBATCH --error=/jet/home/dsouzare/assn4_err.txt

# activate conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate minitorch

nvidia-smi

cd $PROJECT
pwd
ls
cd llmsys_s24_hw4
pwd
ls

python -m pytest -l -v -k "a4_1_1"
python project/run_data_parallel.py --pytest True --n_epochs 1
python -m pytest -l -v -k "a4_1_2"

python project/run_data_parallel.py --world_size 1 --batch_size 64
python project/run_data_parallel.py --world_size 2 --batch_size 128

python -m pytest -l -v -k "a4_2_1"
python -m pytest -l -v -k "a4_2_2"
python project/run_pipeline.py --model_parallel_mode='model_parallel'
python project/run_pipeline.py --model_parallel_mode='pipeline_parallel'
