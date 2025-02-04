# settings.py

DATASET_NAME = "mvsa-mts-v3"
DATASET_PATHS = {
    'mvsa-mts-v3': {
        'train': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3/train.tsv',
        'val': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3/val.tsv',
        'test': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3/test.tsv'
    },
    'mvsa-mts-v3-30': {
        'train': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-30/train.tsv',
        'val': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-30/val.tsv',
        'test': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-30/test.tsv'
    },
    'mvsa-mts-v3-100': {
        'train': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-100/train.tsv',
        'val': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-100/val.tsv',
        'test': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-100/test.tsv'
    },
    'mvsa-mts-v3-1000': {
        'train': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-1000/train.tsv',
        'val': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-1000/val.tsv',
        'test': '/Users/ronengold/Datasets/MVSA-MTS/mvsa-mts-v3-1000/test.tsv'
    }
}

IMAGE_PATH = "/Users/ronengold/Datasets/MVSA-MTS/images"

EMBEDDING_DIM = 200
EMBEDDING_TYPE = "glove"
GLOVE_BASE_PATH = "/Users/ronengold/Datasets/util_models/glove.twitter.27B/"