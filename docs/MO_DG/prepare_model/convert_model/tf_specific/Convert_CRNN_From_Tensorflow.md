# Convert CRNN* Models to the Intermediate Representation (IR) {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_CRNN_From_Tensorflow}

This tutorial explains how to convert a CRNN model to Intermediate Representation (IR).

On GitHub*, you can find several public versions of TensorFlow\* CRNN model implementation. This tutorial explains how to convert the model from
the [https://github.com/MaybeShewill-CV/CRNN_Tensorflow](https://github.com/MaybeShewill-CV/CRNN_Tensorflow) repository to IR. If you
have another implementation of CRNN model, you can convert it to IR in similar way: you need to get inference graph and run the Model Optimizer on it.

**To convert this model to the IR:**

**Step 1.** Clone this GitHub repository and checkout the commit:
    1. Clone reposirory:
```sh
 git clone https://github.com/MaybeShewill-CV/CRNN_Tensorflow.git
```
    2. Checkout necessary commit:
```sh
git checkout 64f1f1867bffaacfeacc7a80eebf5834a5726122
```

**Step 2.** Train the model using framework or use the pretrained checkpoint provided in this repository.

**Step 3.** Create an inference graph:
    1. Go to the `CRNN_Tensorflow` directory with the cloned repository:
```sh
cd path/to/CRNN_Tensorflow
```
    2. Add `CRNN_Tensorflow` folder to `PYTHONPATH`.
       * For Linux\* OS:
```sh
export PYTHONPATH="${PYTHONPATH}:/path/to/CRNN_Tensorflow/"
```
       * For  Windows\* OS add `/path/to/CRNN_Tensorflow/` to the `PYTHONPATH` environment variable in settings.
    3. Open the `tools/demo_shadownet.py` script. After `saver.restore(sess=sess, save_path=weights_path)` line, add the following code:
```python
from tensorflow.python.framework import graph_io
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['shadow/LSTMLayers/transpose_time_major'])
graph_io.write_graph(frozen, '.', 'frozen_graph.pb', as_text=False)
```
    4. Run the demo with the following command:
```sh
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999
```
   If you want to use your checkpoint, replace the path in the `--weights_path` parameter with a path to your checkpoint.
    5. In the `CRNN_Tensorflow` directory, you will find the inference CRNN graph `frozen_graph.pb`. You can use this graph with the OpenVINO&trade; toolkit
     to convert the model into IR and run inference.

**Step 4.** Convert the model into IR:
```sh
python3 path/to/model_optimizer/mo_tf.py --input_model path/to/your/CRNN_Tensorflow/frozen_graph.pb
```




