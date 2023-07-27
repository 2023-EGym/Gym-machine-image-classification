# Gym-machine-image-classification
CNN 기반의 운동기구 이미지 분류 모델

## Model
> ### Resnet
- [paper](https://arxiv.org/abs/1512.03385)
- [Model reference code](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)

## Dataset
- 직접 각 운동기구 이미지 크롤링 후 데이터셋 생성
- 아래와 같이 train, val, test 각 디렉토리에 나눈 뒤 사용 
```bash
    ├── data
      ├── train
      │   ├── class_1
      │   ├── ...
      ├── val
      │   ├── class_1
      │   ├── ...
      ├── test
      │   ├── class_1
      │   ├── ...

```

## How to run
### Training
```
python main.py --train_dir  --val_dir  --save_path 
```
* --train_dir: train 데이터셋 경로
* --val_dir: val 데이터셋 경로
* --save_path: 학습할 모델을 저장할 경로
### Inference
```
python inference.py --test_dir  --model_path
```
* --test_dir: test 데이터셋 경로
* --model_path: 학습된 모델이 저장된 경로
