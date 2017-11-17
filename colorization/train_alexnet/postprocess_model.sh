
# Input 1: GPU_ID
# Input 2: Iteration number (450000 if training has concluded)

# absorb batchnorm
python ./resources/batch_norm_absorb.py --prototxt_in ./train_alexnet/train_val.prototxt --prototxt_out ./train_alexnet/train_val_nobn.prototxt --caffemodel_in ./train_alexnet/models/colornet_iter_${2}.caffemodel --caffemodel_out ./train_alexnet/colornet_iter_${2}_nobn.caffemodel --gpu ${1}

# run rescaling
python ./resources/magic_init/magic_init_mod.py ./train_alexnet/train_val_nobn_rs.prototxt ./train_alexnet/colornet_iter_${2}_nobn_rs.caffemodel -l ./train_alexnet/colornet_iter_${2}_nobn.caffemodel -nit 10 -cs --gpu ${1}

# copy weights into net with FC layers
python ./resources/conv_into_fc.py --gpu ${1} --prototxt_in ./train_alexnet/train_val_nobn.prototxt --prototxt_out ./train_alexnet/train_val_nobn_fc.prototxt --caffemodel_in ./train_alexnet/colornet_iter_${2}_nobn_rs.caffemodel  --caffemodel_out ./train_alexnet/colornet_iter_${2}_nobn_fc_rs.caffemodel
