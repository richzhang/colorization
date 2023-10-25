import joblib
# from dataset.cielab import CIELabConversion

# cielab = CIELabConversion(
#     buckets_path="resources/buckets_313.npy",
#     buckets_knn_path="resources/buckets_knn.joblib"
# )

buckets_knn = joblib.load("resources/buckets_knn.joblib")