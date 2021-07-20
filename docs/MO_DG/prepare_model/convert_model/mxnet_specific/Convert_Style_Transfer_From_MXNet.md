# Converting a Style Transfer Model from MXNet*  {#openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_Convert_Style_Transfer_From_MXNet}

The tutorial explains how to generate a model for style transfer using the public MXNet\* neural style transfer sample.
To use the style transfer sample from OpenVINO&trade;, follow the steps below as no public pre-trained style transfer model is provided with the OpenVINO toolkit.

#### 1. Download or clone the repository with an MXNet neural style transfer sample: [Zhaw's Neural Style Transfer repository](https://github.com/zhaw/neural_style).

#### 2. Prepare the environment required to work with the cloned repository:
1. Install packages dependency:<br>
```sh
sudo apt-get install python-tk
```

2. Install Python\* requirements:
```sh
pip3 install --user mxnet
pip3 install --user matplotlib
pip3 install --user scikit-image
```

#### 3. Download the pre-trained [VGG19 model](https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params) and save it to the root directory of the cloned repository because the sample expects the model `vgg19.params` file to be in that directory.<br>

#### 4. Modify source code files of style transfer sample from cloned repository.<br>

1. Go to the `fast_mrf_cnn` subdirectory.
```sh
cd ./fast_mrf_cnn
```

2. Open the `symbol.py` file and modify the `decoder_symbol()` function. Replace.
```py
def decoder_symbol():
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, num_filter=256, kernel=(3,3), pad=(1,1), stride=(1, 1), name='deco_conv1')
```
with the following code:<br>
```py
def decoder_symbol_with_vgg(vgg_symbol):
    data = mx.sym.Convolution(data=vgg_symbol, num_filter=256, kernel=(3,3), pad=(1,1), stride=(1, 1), name='deco_conv1')
```

3. Save and close the `symbol.py` file.

4. Open and edit the `make_image.py` file:
Modify the `__init__()` function in the `Maker` class. Replace:<br>
```py
decoder = symbol.decoder_symbol()
```
with the following code:<br>
```py
decoder = symbol.decoder_symbol_with_vgg(vgg_symbol)
```

5. To join the pre-trained weights with the decoder weights, make the following changes:
After the code lines for loading the decoder weights:<br>
```py
args = mx.nd.load('%s_decoder_args.nd'%model_prefix)
auxs = mx.nd.load('%s_decoder_auxs.nd'%model_prefix)
```
add the following line:<br>
```py
arg_dict.update(args)
```

6. Use `arg_dict` instead of `args` as a parameter of the `decoder.bind()` function. Replace the line:<br>
```py
self.deco_executor = decoder.bind(ctx=mx.cpu(), args=args, aux_states=auxs)
```
with the following:<br>
```py
self.deco_executor = decoder.bind(ctx=mx.cpu(), args=arg_dict, aux_states=auxs)
```
7. Replace all `mx.gpu` with `mx.cpu` in the `decoder.bind()` function.
8. To save the result model as a `.json` file, add the following code to the end of the `generate()` function in the `Maker` class:<br>
```py
self.vgg_executor._symbol.save('{}-symbol.json'.format('vgg19'))
self.deco_executor._symbol.save('{}-symbol.json'.format('nst_vgg19'))
```
9. Save and close the `make_image.py` file.

#### 5. Run the sample with a decoder model according to the instructions from the `README.md` file in the cloned repository.
For example, to run the sample with the pre-trained decoder weights from the `models` folder and output shape, use the following code:<br>
```py
import make_image
maker = make_image.Maker('models/13', (1024, 768))
maker.generate('output.jpg', '../images/tubingen.jpg')
```
Where `'models/13'` string is composed of the following sub-strings: 
* `'models/'` - path to the folder that contains .nd files with pre-trained styles weights and `'13'`
*  Decoder prefix: the repository contains a default decoder, which is the 13_decoder. 

You can choose any style from [collection of pre-trained weights](https://pan.baidu.com/s/1skMHqYp). The `generate()` function generates `nst_vgg19-symbol.json` and `vgg19-symbol.json` files for the specified shape. In the code, it is [1024 x 768] for a 4:3 ratio, and you can specify another, for example, [224,224] for a square ratio.

#### 6. Run the Model Optimizer to generate an Intermediate Representation (IR):

1. Create a new directory. For example:<br>
```sh
mkdir nst_model
```
2. Copy the initial and generated model files to the created directory. For example, to copy the pre-trained decoder weights from the `models` folder to the `nst_model` directory, run the following commands:<br>
```sh
cp nst_vgg19-symbol.json nst_model
cp vgg19-symbol.json nst_model
cp ../vgg19.params nst_model/vgg19-0000.params
cp models/13_decoder_args.nd nst_model
cp models/13_decoder_auxs.nd nst_model
```
> **NOTE**: Make sure that all the `.params` and `.json` files are in the same directory as the `.nd` files. Otherwise, the conversion process fails.

3. Run the Model Optimizer for MXNet. Use the `--nd_prefix_name` option to specify the decoder prefix and `--input_shape` to specify input shapes in [N,C,W,H] order. For example:<br>
```sh
python3 mo.py --input_symbol <path/to/nst_model>/nst_vgg19-symbol.json --framework mxnet --output_dir <path/to/output_dir> --input_shape [1,3,224,224] --nd_prefix_name 13_decoder --pretrained_model <path/to/nst_model>/vgg19-0000.params
```
4. The IR is generated (`.bin`, `.xml` and `.mapping` files) in the specified output directory and ready to be consumed by the Inference Engine. 
