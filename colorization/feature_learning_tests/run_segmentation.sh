
mkdir ./feature_learning_tests/segmentation/models
python ./feature_learning_tests/segmentation/solve.py --gpu ${1} --phase 0
python ./feature_learning_tests/segmentation/solve.py --gpu ${1} --phase 1
python ./feature_learning_tests/segmentation/solve.py --gpu ${1} --phase 2
 