# Gym-machine-image-classification
CNN 기반의 운동기구 이미지 분류 모델


## Dataset
- 직접 크롤링한 운동기구 이미지 데이터셋 사용 

## How to run
### Training
```
python main.py --train_dir  --val_dir  --save_path 
```
### Inference
```
python inference.py --test_dir  --model_path
```
### Make tffile 
```
python h5_to_tffile.py --save_path  --model_path
```
