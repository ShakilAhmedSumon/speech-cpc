import os


DATASET_PATH = "DATA"
SAVED_EMBEDDED_PATH = "embeddings"

if not os.path.exists(SAVED_EMBEDDED_PATH + '/' + DATASET_PATH):
    os.makedirs(SAVED_EMBEDDED_PATH + '/' + DATASET_PATH)