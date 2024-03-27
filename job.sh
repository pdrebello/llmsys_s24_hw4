export CUDA_VISIBLE_DEVICES=0,1
python -m pytest -l -v -k "a4_1_1" &> a4_1_1.txt
python project/run_data_parallel.py --pytest True --n_epochs 1 &> data_parallel_test.txt
python -m pytest -l -v -k "a4_1_2" &> a4_1_2.txt

python project/run_data_parallel.py --world_size 1 --batch_size 64 &> data_parallel_1.txt
python project/run_data_parallel.py --world_size 2 --batch_size 128 &> data_parallel_2.txt

python -m pytest -l -v -k "a4_2_1" &> a4_2_1.txt
python -m pytest -l -v -k "a4_2_2" &> a4_2_2.txt
python project/run_pipeline.py --model_parallel_mode='model_parallel' &> model_parallel.txt
python project/run_pipeline.py --model_parallel_mode='pipeline_parallel' &> pipeline_parallel.txt
