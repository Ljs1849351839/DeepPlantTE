# DeepPlantTE

## Environment Setup
The running environment can be created using Conda:
```
conda env create -f environment.yml
conda activate DeepPlantTE
```

## Model Weights
The trained model weight file should be placed in the model/ directory:
```
model/best_deepplantte_model.pth
```

## Prediction/Testing
To evaluate DeepPlantTE using the trained model, run:
```
python src/test_deepplantte-final.py
```

## Help

For specific usage parameters, run:
```
python src/test_deepplantte-final.py -h
```

## Data Availability
The training, validation and test datasets used in this study are available at: https://pan.quark.cn/s/7ac5e8cdfe96
