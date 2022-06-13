# Age-and-gender-prediction-using-CNN-and-ViT

## Download site
- [Training Dataset](https://drive.google.com/file/d/1ZSqeOSU7-n2YxK91Ef5g4qOijACBdc0g/view)
- [Pretrained Models](https://drive.google.com/drive/folders/1TEOAdkTWqijFA_UnDfv_R7FF_8AsbuJj?usp=sharing)

## Install reuqired package
- To install the required package run the command below.

```
pip install -r requirements.txt
```

## To training the model
1. Run **preprocessing.ipynb** and produce **train.csv** and **val.csv**
2. Run **main.py** and set some arguments
```
python main.py --data-path './AFAD-Full' --model vit_base_patch16_224 --model-type transformer
```
- model-type can be resnet or transformer
- support model list:
  - vit_base_patch16_224 (v)
  - vit_small_patch16_224 (v)
  - vit_tiny_patch16_224
  - resnet50 (v)
  - resnet34
  - resnet18
  - more model can be seen at [resnet page](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py) and [transformer page](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
  - in our work, we test performance of three models with (v), testing result can be seen at evaluate.ipynb

## To using the face detection
- There are three detection method **cv2**, **mediapipe** and **mtcnn**
- There are three kind of pretrained model provided above in google drive
  - resnet50
  - VIT_small
  - VIT_base 

- Run Real-time detection (only works on Windows)
```
python "face detect.py" -f mediapipe -m VIT_small
```
- We can also detect a single picture instead of make a detection screen
```
python "face detect.py" -f mediapipe -m VIT_small -p ./picture.png
```
