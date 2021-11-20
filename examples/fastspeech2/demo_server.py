#!/usr/bin/env python

import argparse
import falcon
import os
import glob
 
import time, io
import argparse
import yaml
import soundfile as sf
import tensorflow as tf
import numpy as np


import scipy.io.wavfile

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)





html_body = '''<html>
<head>
<title>MonTTS Demo</title>
<link rel="shortcut icon" href="#" />
</head>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = tts_inference(args, req.params.get('text'), ttsmodel, vocoder, processor)
    res.content_type = 'audio/wav'


api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())



def tts_inference(args, line, ttsmodel, vocoder, processor):
    #print('qqqqq')
    input_ids = processor.text_to_sequence(line)
    #print(input_ids)
    mel_before, mel_after, duration_outputs, _, _ = ttsmodel.inference(
       input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
       speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
       speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
       f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
       energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    
    mel_outputs = mel_after
    return syn_vocoder(args, vocoder, mel_outputs, line )
     


  

def syn_vocoder(args, vocoder, mel, line ):
    audio_after = vocoder.inference(mel)[0, :, 0]
    #print("wav...")
    out = io.BytesIO()
    time_str = int(time.time())
    outpath = 'examples/fastspeech2/demo-log/' + ''.join(line.split(' ')) + '-' + str(time_str) + '.wav'
    # save to file

    sf.write(outpath, audio_after, 22050, "PCM_16")
    audio_after *= 32767 / max(0.01, np.max(np.abs(audio_after)))
    scipy.io.wavfile.write(out, 22050, audio_after.numpy().astype(np.int16))
    #print('syn ok..')
    return out.getvalue()



if __name__ == '__main__':
    from wsgiref import simple_server
    parser = argparse.ArgumentParser(
        description="Decode mel-spectrogram from folder ids with trained Tacotron-2 "
        "(See detail in tensorflow_tts/example/tacotron2/decode_tacotron2.py)."
    )
    parser.add_argument(
        "--tts_ckpt", 
        default = "examples/fastspeech2/exp/train.fastspeech2-mon.v1/checkpoints/model-200000.h5",
        type=str, required=False, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--vocoder_ckpt", 
        default = "examples/parallel_wavegan/exp/train.parallel_wavegan-mon.v1/checkpoints/generator-400000.h5",
        type=str, required=False, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--stats_path",
        default="dump_mon/stats.npy",
        type=str,
        required=False,
        help="stats path",
    )
    parser.add_argument(
        "--dataset_config",
        default="preprocess/mon_preprocess.yaml",
        type=str,
        required=False,
        help="dataset_config path",
    )
    parser.add_argument(
        "--tts_config",
        default='examples/fastspeech2/conf/fastspeech2.v1.yaml',
        type=str,
        required=False,
        help="tts_config path",
    )
    parser.add_argument(
        "--vocoder_config",
        default='examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml',
        type=str,
        required=False,
        help="vocoder_config path",
    )
    parser.add_argument(
        "--lan_json",
        default="dump_mon/mon_mapper.json",
        type=str,
        required=False,
        help="language json  path",
    )
    parser.add_argument('--port', type=int, default=9000)
    print('aaa')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
