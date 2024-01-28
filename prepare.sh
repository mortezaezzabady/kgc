datasets=("gt" "em" "fb" "wn")
dirs=("checkpoints" "results" "embeddings" "data")

for dir in ${dirs[*]}; do
        mkdir ${dir}
done
for dataset in ${datasets[*]}; do
    mkdir results/${dataset}
    mkdir embeddings/${dataset}
    mkdir data/${dataset}
    mkdir checkpoints/${dataset}
    mkdir checkpoints/${dataset}/out
    mkdir checkpoints/${dataset}/out/conv
done

