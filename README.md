# HEAPML
HEAPML is a final year BEng project that aims to predict phase formations of high entropy alloys using machine learning models.

This repo contains notebooks that outline the process of training three models (SVM, RF and GBDT).

As of version 0.9.0 of scikit-optomize, all instances of `np.int` in `skopt/space/transformers.py` need to be changed to `int` or `np.int64`/`np.int32` in order to be compatible with version 1.24.2 of numpy
