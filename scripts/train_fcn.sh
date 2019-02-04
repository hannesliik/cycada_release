gpu=0
data=gta5
#data=cityscapes
#data2=cyclegta5
model=drn26


crop=768
datadir=~/datasets/cycada_data
batch=1
iterations=100000
lr=1e-3
momentum=0.99
num_cls=19

#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${data}_${model}
mkdir -p results/${data} 

python scripts/train_fcn.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --dataset ${data}  #--dataset ${data2} 
