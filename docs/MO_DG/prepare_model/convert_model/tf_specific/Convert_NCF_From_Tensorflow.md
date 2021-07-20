# Convert Neural Collaborative Filtering Model from TensorFlow* to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_NCF_From_Tensorflow}

This tutorial explains how to convert Neural Collaborative Filtering (NCF) model to Intermediate Representation (IR).

[Public TensorFlow NCF model](https://github.com/tensorflow/models/tree/master/official/recommendation) does not contain
 pretrained weights. To convert this model to the IR:
 1. Use [the instructions](https://github.com/tensorflow/models/tree/master/official/recommendation#train-and-evaluate-model) from this repository to train the model. 
 2. Freeze the inference graph you get on previous step in `model_dir` following 
the instructions from the Freezing Custom Models in Python* section of 
[Converting a TensorFlow* Model](../Convert_Model_From_TensorFlow.md). 
Run the following commands:
```python
import tensorflow as tf
from tensorflow.python.framework import graph_io

sess = tf.Session()
saver = tf.train.import_meta_graph("/path/to/model/model.meta")
saver.restore(sess, tf.train.latest_checkpoint('/path/to/model/'))

frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, \
                                                      ["rating/BiasAdd"])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
```
where `rating/BiasAdd` is an output node.

 3. Convert the model to the IR.If you look at your frozen model, you can see that 
it has one input that is split to four `ResourceGather` layers.

![NCF model beginning](../../../img/NCF_start.png)

 But as the Model Optimizer does not support such data feeding, you should skip it. Cut 
the edges incoming in `ResourceGather`s port 1:
```sh
python3 mo_tf.py --input_model inference_graph.pb \
--input 1:embedding/embedding_lookup,1:embedding_1/embedding_lookup,\
1:embedding_2/embedding_lookup,1:embedding_3/embedding_lookup \
--input_shape [256],[256],[256],[256]
```
Where 256 is a `batch_size` you choose for your model.

Alternatively, you can do steps 2 and 3 in one command line:
```sh
python3 mo_tf.py --input_meta_graph /path/to/model/model.meta \
--input 1:embedding/embedding_lookup,1:embedding_1/embedding_lookup,\
1:embedding_2/embedding_lookup,1:embedding_3/embedding_lookup \
--input_shape [256],[256],[256],[256] --output rating/BiasAdd
```

