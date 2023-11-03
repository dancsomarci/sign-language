# ASL-fingerspelling

The models aim to develop a sign language translation system from fingerspelling pose data acquired from native signers.

The different stages of the project are separated among python notebooks:

| File       | Purpose |
|------------|-----|
| data_handling.ipynb       | Processing of raw data (cretaing a tfrecord dataset)  |
| encoder_decoder.ipynb      | Preprocessing | Creating and Training models | Saving  |
| testing_asl_fingerspelling.ipynb        | Code for testing and demo application  |

You can also see the progression from week to week in the `archived models` folder.

The rest of the folders contain the trained models with in `tf-savedmodel` format.


## Dataset

https://www.kaggle.com/c/asl-fingerspelling