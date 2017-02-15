 
mkdir ./feature_learning_tests/linear/models/
./caffe-colorization/build/tools/caffe train -gpu ${1} -solver ./feature_learning_tests/linear/solver.prototxt -weights ./models/alexnet_release_450000_nobn_rs.caffemodel
