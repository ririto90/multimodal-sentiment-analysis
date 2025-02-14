# settings.py

DATASET_NAME = "mvsa-mts-v3"
DATASET_PATHS = {
    'mvsa-mts-v3': {
        'train': 'Datasets/MVSA-MTS/mvsa-mts-v3/train.tsv',
        'val': 'Datasets/MVSA-MTS/mvsa-mts-v3/val.tsv',
        'test': 'Datasets//MVSA-MTS/mvsa-mts-v3/test.tsv'
    },
    'mvsa-mts-v3-30': {
        'train': 'Datasets/MVSA-MTS/mvsa-mts-v3-30/train.tsv',
        'val': 'Datasets/MVSA-MTS/mvsa-mts-v3-30/val.tsv',
        'test': 'Datasets/MVSA-MTS/mvsa-mts-v3-30/test.tsv'
    },
    'mvsa-mts-v3-100': {
        'train': 'Datasets/MVSA-MTS/mvsa-mts-v3-100/train.tsv',
        'val': 'Datasets/MVSA-MTS/mvsa-mts-v3-100/val.tsv',
        'test': 'Datasets/MVSA-MTS/mvsa-mts-v3-100/test.tsv'
    },
    'mvsa-mts-v3-1000': {
        'train': 'Datasets/MVSA-MTS/mvsa-mts-v3-1000/train.tsv',
        'val': 'Datasets/MVSA-MTS/mvsa-mts-v3-1000/val.tsv',
        'test': 'Datasets/MVSA-MTS/mvsa-mts-v3-1000/test.tsv'
    }
}

IMAGE_PATH = "Datasets//MVSA-MTS/images"

EMBEDDING_DIM = 200
EMBEDDING_TYPE = "glove"
GLOVE_BASE_PATH = "Datasets//util_models/glove.twitter.27B/"