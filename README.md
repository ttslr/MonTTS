# MonTTS: A Real-time and High-fidelity Mongolian TTS Model with Complete Non-autoregressive Mechanism 
# (MonTTS：完全非自回归的实时、高保真蒙古语语音合成模型)
 

## 0) Environment Preparation

This project uses `conda` to manage all the dependencies, you should install [anaconda](https://anaconda.org/) if you have not done so. 

```bash
# Clone the repo
git clone https://github.com/ttslr/MonTTS.git
cd $PROJECT_ROOT_DIR
```

### Install dependencies
```bash
conda env create -f Environment/environment.yaml
```

### Activate the installed environment
```bash
conda activate montts
```

## 1) Prepare MonSpeech Dataset

Prepare our MonSpeech dataset in the following format:
```
|- MonSpeech/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```

Where `metadata.csv` has the following format: `id|transcription`. This is a ljspeech-like format.

You can find MonSpeech in the [Links](#Links) section.

## 2) Preprocessing

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

To reproduce the steps above:
```
tensorflow-tts-preprocess --rootdir /home/rui/MonSpeech  --outdir ./dump_mon --config preprocess/mon_preprocess.yaml --dataset mon
```

```
tensorflow-tts-normalize --rootdir ./dump_mon --outdir ./dump_mon --config preprocess/mon_preprocess.yaml --dataset mon
```

 



## 3) Training MonTTS from scratch with MonSpeech dataset

Based on the script [`train_fastspeech2.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech2/train_fastspeech2.py).

 
This example code show you how to train MonTTS from scratch with Tensorflow 2 based on custom training loop and tf.function. 

  
Here is an example command line to training MonTTS from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump_mon/train/ \
  --dev-dir ./dump_mon/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2-mon.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump_mon/stats_f0.npy \
  --energy-stat ./dump_mon/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode.

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/fastspeech2/exp/train.fastspeech2-mon.v1/checkpoints/ckpt-100000
```

If you want to finetune a model, use `--pretrained` like this with your model filename
```bash
--pretrained pretrained.h5
```


 

## 4) Vocoder Training


First, you need training generator with only stft loss:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump_mon/train/ \
  --dev-dir ./dump_mon/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan-mon.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:


```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump_mon/train/ \
  --dev-dir ./dump_mon/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan-mon.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --resume ./examples/hifigan/exp/train.hifigan-mon.v1/checkpoints/ckpt-100000
```

## 5) Tensorboard 
You should find a dir `log` in all of your output dirs, that is the `LOG_DIR` you should use below.

```bash
tensorboard --logdir=${LOG_DIR}
```

For example, you can follow below example command lines to access the tensrobords to check the training progress:


```bash
tensorboard --logdir examples/fastspeech2/exp/train.fastspeech2-mon-exp1
```


![image](https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)

```bash
tensorboard --logdir examples/hifigan/exp/train.hifigan-mon.v1
```

![image](https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)

## 6) MonTTS Model Inference

You can follow below example command line to generate synthesized speeh for given text in 'dump_mon/inference.txt' using Griffin-Lim and trained HiFi-GAN vocoder:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/fastspeech2/inference_fastspeech2-mon.py \
    --outdir prediction/mon_inference_fastspeech2 \
    --infile dump_mon/inference.txt  \
    --tts_ckpt examples/fastspeech2/exp/train.fastspeech2-mon.v1/checkpoints/model-200000.h5 \
    --vocoder_ckpt  examples/hifigan/exp/train.hifigan-mon.v1/checkpoints/generator-420000.h5 \
    --stats_path dump_mon/stats.npy \
    --dataset_config preprocess/mon_preprocess.yaml \
    --tts_config examples/fastspeech2/conf/fastspeech2.v1.yaml \
    --vocoder_config examples/hifigan/conf/hifigan.v1.yaml \
    --lan_json dump_mon/mon_mapper.json 
```

You can find pre-trained models in the [Links](#Links) section.


The synthesized speech will save to `prediction/mon_inference_fastspeech2` folder.

## 7) Online Demo

You can also run `demo_server.py` to build a online demo.


```bash
python examples/fastspeech2/demo_server.py
```

Note that you need to point your browser at localhost:9000 and then type what you want to synthesize.


![image](https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)

## Links

- Please contact the [Author](#Author) to access the MonSpeech corpus and Pre-trained models
- Demo: [link to synthesized audio samples](https://github.com/ttslr/MonTTS/tree/main/prediction/mon_inference_fastspeech2)


## Author
[Rui Liu](https://ttslr.github.io)<br> 
E-mail: liurui_imu@163.com

## Citation
Please kindly cite the following paper if you use this code repository in your work,


```
 @inproceedings{liu2021montts,
  title={MonTTS: A Real-time and High-fidelity Mongolian TTS Model with Complete Non-autoregressive Mechanism (in Chinese)},
  author={Rui, Liu and Shiyin, Kang and Jingdong, Li amd Feilong, Bao and Guanglai, Gao},
  booktitle={JOURNAL OF CHINESE INFORMATION PROCESSING (中文信息学报)},
  year={2021}
}

```


