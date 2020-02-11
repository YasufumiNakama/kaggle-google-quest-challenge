# kaggle-google-quest-challenge
This respository contains my code for competition in kaggle.

60th Place Solution for [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge)

Public score: 0.43938 (64th)  
Private score: 0.40094 (60th)

## Prerequisite
Pull PyTorch image from [NVIDIA GPU CLOUD (NGC)](https://ngc.nvidia.com/)
```
docker login nvcr.io
docker image pull nvcr.io/nvidia/pytorch:20.01-py3
docker run --gpus all -it --ipc=host --name=bert nvcr.io/nvidia/pytorch:20.01-py3
```

## Usage
```
pip install iterative-stratification
pip install category_encoders

# train BERT model
python train_bert.py
```
