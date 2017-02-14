
# bash train_alexnet/train_model.sh [[GPU_ID]]

mkdir ./train_alexnet/models
./caffe-colorization/build/tools/caffe train -solver ./train_alexnet/solver0.prototxt -weights ./train_alexnet/mi.caffemodel -gpu ${1}
./caffe-colorization/build/tools/caffe train -solver ./train_alexnet/solver1.prototxt -snapshot ./train_alexnet/models/colornet_iter_240000.solverstate -gpu ${1}
./caffe-colorization/build/tools/caffe train -solver ./train_alexnet/solver2.prototxt -snapshot ./train_alexnet/models/colornet_iter_300000.solverstate -gpu ${1}
./caffe-colorization/build/tools/caffe train -solver ./train_alexnet/solver3.prototxt -snapshot ./train_alexnet/models/colornet_iter_400000.solverstate -gpu ${1}
