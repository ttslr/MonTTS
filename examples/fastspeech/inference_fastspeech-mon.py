import glob
 
import time, os
import argparse
import yaml
import soundfile as sf
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

from tensorflow_tts.utils import TFGriffinLim, griffin_lim_lb

# %config InlineBackend.figure_format = 'svg'



from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)




#mel_spec = np.load("dump_mon/valid/norm-feats/mon_00043568-norm-feats.npy")
#print(mel_spec.shape). # len, 80

#gt_wav = np.load("../dump_mon/train/wavs/LJ001-0007-wave.npy")

 

def tts_inference(args, ttsmodel, vocoder, processor):
    with open(args.infile, 'r') as f:
        for line in f:
            input_ids = processor.text_to_sequence(line)
            #input_ids = tf.expand_dims(input_ids, 0)
            masked_mel_before, masked_mel_after, duration_outputs = ttsmodel.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            #speaker_ids=tf.zeros(shape=[tf.shape(input_ids)[0]], dtype=tf.int32),
            #speed_ratios=tf.ones(shape=[tf.shape(input_ids)[0]], dtype=tf.float32),
            )
            filename = 'test'
            mel_outputs = masked_mel_after
            syn_gl(args, mel_outputs, filename)
            syn_vocoder(args, vocoder, mel_outputs,  filename)
     


 

def syn_gl(args, mel, filename):
    # Single file

    mel = mel.numpy()
    mel = mel[0]
    print(mel.shape) # [len, 80]
    config = yaml.load(open(args.dataset_config), Loader=yaml.Loader)
    griffin_lim_lb(mel, args.stats_path, config, 32, args.outdir, wav_name= filename + '-gl') # [mel_len] -> [audio_len]

def syn_vocoder(args, vocoder, mel, filename):
    audio_after = vocoder.inference(mel)[0, :, 0]
    # save to file
    sf.write(os.path.join(args.outdir, filename + '-vocoder.wav'), audio_after, 22050, "PCM_16")

def main():
    """Running decode tacotron-2 mel-spectrogram."""
    parser = argparse.ArgumentParser(
        description="Decode mel-spectrogram from folder ids with trained Tacotron-2 "
        "(See detail in tensorflow_tts/example/tacotron2/decode_tacotron2.py)."
    )
    parser.add_argument(
        "--outdir", 
        default = "prediction/mon_inference",
        type=str, required=True, help="directory to save generated speech."
    )
    parser.add_argument(
        "--infile",  
        default = "dump_mon/inference.txt",
        type=str, required=True, help="inference text."
    )
    parser.add_argument(
        "--tts_ckpt", 
        default = "examples/tacotron2/exp/train.tacotron2-mon-exp1/checkpoints/model-10000.h5",
        type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--vocoder_ckpt", 
        default = "examples/parallel_wavegan/exp/train.parallel_wavegan-mon.v1/checkpoints/generator-95000.h5",
        type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--stats_path",
        default="dump_mon/stats.npy",
        type=str,
        required=True,
        help="stats path",
    )
    parser.add_argument(
        "--dataset_config",
        default="preprocess/mon_preprocess.yaml",
        type=str,
        required=True,
        help="dataset_config path",
    )
    parser.add_argument(
        "--tts_config",
        default='examples/tacotron2/conf/tacotron2-mon-exp1.yaml',
        type=str,
        required=True,
        help="tts_config path",
    )
    parser.add_argument(
        "--vocoder_config",
        default='examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml',
        type=str,
        required=True,
        help="vocoder_config path",
    )
    parser.add_argument(
        "--lan_json",
        default="dump_mon/mon_mapper.json",
        type=str,
        required=True,
        help="language json  path",
    )

    args = parser.parse_args()
    
    # initialize fastspeech model.
    tts_config = AutoConfig.from_pretrained(args.tts_config)
    ttsmodel = TFAutoModel.from_pretrained(
	    config=tts_config,
	    pretrained_path=args.tts_ckpt
	)


    # initialize melgan model
    vocoder_config = AutoConfig.from_pretrained(args.vocoder_config)
    vocoder = TFAutoModel.from_pretrained(
	    config=vocoder_config,
	    pretrained_path=args.vocoder_ckpt
	)

    processor = AutoProcessor.from_pretrained(pretrained_path=args.lan_json)


    tts_inference(args, ttsmodel, vocoder, processor)
    print('ok')


if __name__ == '__main__':
	main()
