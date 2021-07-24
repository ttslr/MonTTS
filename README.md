# MonTTS




## 1) Environment Preparation

This project uses `conda` to manage all the dependencies, you should install [anaconda](https://anaconda.org/) if you have not done so. 

```bash
# Clone the repo
git clone https://github.com/ttslr/CAccentTTS.git
cd $PROJECT_ROOT_DIR
```

### Install dependencies
```bash
conda env create -f environment.yml
```

### Activate the installed environment
```bash
conda activate accenttts
```



## 2) Data Preparation


### Text Normalization


### Duration Extraction



## 3) MonTTS Model Training

The training and validation data should be specified in text files, see `data/filelists` for examples.

```bash
python train.py
```


## 4) Vocoder Training

The training and validation data should be specified in text files, see `data/filelists` for examples.

```bash
python train.py
```

## 5) Tensorboard (View training progress)
You should find a dir `log` in all of your output dirs, that is the `LOG_DIR` you should use below.

```bash
tensorboard --logdir=${LOG_DIR}
```

## 6) MonTTS Model Inference
You can find pre-trained models in the [Links](#Links) section.

```bash
python synthesis.py
```


## Links

- Pre-trained models: [link to model](https://drive.google.com/xxx)
- Training data (Mongolian): [link to training data](https://xxx)
- Demo: [link to audio samples](https://ttslr.github.io/CAccentTTS/demo)


## Author
[Rui Liu](https://ttslr.github.io)<br> 
E-mail: liurui_imu@163.com

## Citation
Please kindly cite the following paper if you use this code repository in your work,


```
xxx
```


