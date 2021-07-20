# Convert TensorFlow* XLNet Model to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_XLNet_From_Tensorflow}

Pre-trained models for XLNet (Bidirectional Encoder Representations from Transformers) are
[publicly available](https://github.com/zihangdai/xlnet).

## Supported Models

Currently, the following models from the [pre-trained XLNet model list](https://github.com/zihangdai/xlnet#pre-trained-models) are supported:

* **[`XLNet-Large, Cased`](https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip)**
* **[`XLNet-Base, Cased`](https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip)**

## Download the Pre-Trained Base XLNet Model

Download and unzip an archive with the [XLNet-Base, Cased](https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip).

After the archive is unzipped, the directory `cased_L-12_H-768_A-12` is created and contains the following files:
* TensorFlow checkpoint (`xlnet_model.ckpt`) containing the pre-trained weights (which is actually 3 files)
* sentence piece model (`spiece.model`) used for (de)tokenization 
* config file (`xlnet_config.json`) which specifies the hyperparameters of the model 

To get pb-file from the archive contents, you need to do the following.

1. Run commands

```sh
   cd ~
   mkdir XLNet-Base
   cd XLNet-Base
   git clone https://github.com/zihangdai/xlnet
   wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
   unzip cased_L-12_H-768_A-12.zip
   mkdir try_save
```

   

2. Save and run the following script:

```python
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework import graph_io

import model_utils
import xlnet

LENGTHS = 50
BATCH = 1
OUTPUT_DIR = '~/XLNet-Base/try_save/'
INIT_CKPT_PATH = '~/XLNet-Base/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt'
XLNET_CONFIG_PATH = '~/XLNet-Base/xlnet_cased_L-12_H-768_A-12/xlnet_config.json'

FLags = namedtuple('FLags', 'use_tpu init_checkpoint')
FLAGS = FLags(use_tpu=False, init_checkpoint=INIT_CKPT_PATH)

xlnet_config = xlnet.XLNetConfig(json_path=XLNET_CONFIG_PATH)
run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.1, dropatt=0.1,)


sentence_features_input_idx = tf.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='input_ids')
sentence_features_segment_ids = tf.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='seg_ids')
sentence_features_input_mask = tf.placeholder(tf.float32, shape=[LENGTHS, BATCH], name='input_mask')

with tf.Session() as sess:
    xlnet_model = xlnet.XLNetModel(xlnet_config=xlnet_config, run_config=run_config,
                                   input_ids=sentence_features_input_idx,
                                   seg_ids=sentence_features_segment_ids,
                                   input_mask=sentence_features_input_mask)

    sess.run(tf.global_variables_initializer())
    model_utils.init_from_checkpoint(FLAGS, True)

    # Save the variables to disk.
    saver = tf.train.Saver()

    # Saving checkpoint
    save_path = saver.save(sess, OUTPUT_DIR + "model.ckpt")

    # Freezing model
    outputs = ['model/transformer/dropout_2/Identity']
    graph_def_freezed = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)

    # Saving non-frozen and frozen  model to pb
    graph_io.write_graph(sess.graph.as_graph_def(), OUTPUT_DIR, 'model.pb', as_text=False)
    graph_io.write_graph(graph_def_freezed,OUTPUT_DIR, 'model_frozen.pb',
                         as_text=False)

    # Write to tensorboard
    with tf.summary.FileWriter(logdir=OUTPUT_DIR, graph_def=graph_def_freezed) as writer:
        writer.flush()
```

The script should save into `~/XLNet-Base/xlnet`.

## Download the Pre-Trained Large XLNet Model

Download and unzip an archive with the [XLNet-Large, Cased](https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip).

After the archive is unzipped, the directory `cased_L-12_H-1024_A-16` is created and contains the following files:

* TensorFlow checkpoint (`xlnet_model.ckpt`) containing the pre-trained weights (which is actually 3 files)
* sentence piece model (`spiece.model`) used for (de)tokenization 
* config file (`xlnet_config.json`) which specifies the hyperparameters of the model 

To get pb-file from the archive contents, you need to do the following.

1. Run commands

```sh
   cd ~
   mkdir XLNet-Large
   cd XLNet-Large
   git clone https://github.com/zihangdai/xlnet
   wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
   unzip cased_L-24_H-1024_A-16.zip
   mkdir try_save
```



2. Save and run the following script:

```python
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework import graph_io

import model_utils
import xlnet

LENGTHS = 50
BATCH = 1
OUTPUT_DIR = '~/XLNet-Large/try_save'
INIT_CKPT_PATH = '~/XLNet-Large/cased_L-24_H-1024_A-16/xlnet_model.ckpt'
XLNET_CONFIG_PATH = '~/XLNet-Large/cased_L-24_H-1024_A-16/xlnet_config.json'

FLags = namedtuple('FLags', 'use_tpu init_checkpoint')
FLAGS = FLags(use_tpu=False, init_checkpoint=INIT_CKPT_PATH)

xlnet_config = xlnet.XLNetConfig(json_path=XLNET_CONFIG_PATH)
run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.1, dropatt=0.1,)


sentence_features_input_idx = tf.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='input_ids')
sentence_features_segment_ids = tf.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='seg_ids')
sentence_features_input_mask = tf.placeholder(tf.float32, shape=[LENGTHS, BATCH], name='input_mask')

with tf.Session() as sess:
    xlnet_model = xlnet.XLNetModel(xlnet_config=xlnet_config, run_config=run_config,
                                   input_ids=sentence_features_input_idx,
                                   seg_ids=sentence_features_segment_ids,
                                   input_mask=sentence_features_input_mask)

    sess.run(tf.global_variables_initializer())
    model_utils.init_from_checkpoint(FLAGS, True)

    # Save the variables to disk.
    saver = tf.train.Saver()

    # Saving checkpoint
    save_path = saver.save(sess, OUTPUT_DIR + "model.ckpt")

    # Freezing model
    outputs = ['model/transformer/dropout_2/Identity']
    graph_def_freezed = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)

    # Saving non-frozen and frozen  model to pb
    graph_io.write_graph(sess.graph.as_graph_def(), OUTPUT_DIR, 'model.pb', as_text=False)
    graph_io.write_graph(graph_def_freezed,OUTPUT_DIR, 'model_frozen.pb',
                         as_text=False)

    # Write to tensorboard
    with tf.summary.FileWriter(logdir=OUTPUT_DIR, graph_def=graph_def_freezed) as writer:
        writer.flush()
```

The script should save into `~/XLNet-Large/xlnet`.



## Convert frozen TensorFlow XLNet Model to IR

To generate the XLNet Intermediate Representation (IR) of the model, run the Model Optimizer with the following parameters:
```sh
python3 mo.py --input_model path-to-model/model_frozen.pb  --input "input_mask[50 1],input_ids[50 1],seg_ids[50 1]" --log_level DEBUG --disable_nhwc_to_nchw
```

