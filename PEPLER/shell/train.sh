echo $1
seed=1111
model_name=/uiadapter_qwen/
data_path=./data/
data_name=/reviews.pickle
llm_model=../llms/Qwen2.5-7B/
checkpoint_dir=./checkpoints/
out_file_dir=./outputs/
out_file_name=generated.txt
out_log_dir=./logs/
out_log=train.log

for d_type in ClothingShoesAndJewelry TripAdvisor MoviesAndTV
do
    for d_index in 1 2 3 4 5
    do
        mkdir -p ${out_log_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${out_file_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${checkpoint_dir}${d_type}${model_name}

        echo "data_type: $d_type, data_index: $d_index , seed: $seed"
        TRANSFORMERS_CACHE=./llm/ \
        HF_DATASETS_CACHE=./llm/ \
        CUDA_VISIBLE_DEVICES=$1 python -u ./main.py \
            -data_path ${data_path}${d_type}${data_name} \
            -index_dir ${data_path}${d_type}\/${d_index}\/ \
            -llm_model ${llm_model} \
            -lr 2e-7 \
            -epochs 1 \
            -batch_size 16 \
            -rating_reg 0.01 \
            -bal_reg 0.1 \
            -mlp_size 400 \
            -k 768 \
            -r 8 \
            -lora_modules 7 \
            -acc_steps 1 \
            -seed $seed \
            -cuda \
            -log_interval 200 \
            -checkpoint ${checkpoint_dir}${d_type} \
            -outf ${out_file_dir}${d_type}\/${d_index}${model_name}${out_file_name} \
            -words 20 \
            -model_type uiadapter \
            > ${out_log_dir}${d_type}\/${d_index}${model_name}${out_log}
    done
done
