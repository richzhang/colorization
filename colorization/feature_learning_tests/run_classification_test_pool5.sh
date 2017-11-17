
# Input 1: path to train_cls.py from VOC-classification repo
# Input 2: [GPU_ID]

mkdir ./feature_learning_tests/classification
python ${1} ./feature_learning_tests/classification/classification_trainval.prototxt ./models/alexnet_release_450000_nobn_rs.caffemodel --gpu ${2} --no-mean --train-from pool5 --output-dir ./feature_learning_tests/classification/pool5/
