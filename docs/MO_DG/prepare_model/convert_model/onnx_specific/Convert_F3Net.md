# Convert PyTorch* F3Net to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_F3Net}

[F3Net](https://github.com/weijun88/F3Net): Fusion, Feedback and Focus for Salient Object Detection

## Download and Convert the Model to ONNX*

To download the pretrained model or train the model yourself, refer to the 
[instruction](https://github.com/weijun88/F3Net/blob/master/README.md) in the F3Net model repository. Firstly, 
convert the model to ONNX\* format. Create and run the script with the following content in the `src`
directory of the model repository:
```python
import torch

from dataset import Config
from net import F3Net

cfg = Config(mode='test', snapshot=<path_to_checkpoint_dir>)
net = F3Net(cfg)
image = torch.zeros([1, 3, 352, 352])
torch.onnx.export(net, image, 'f3net.onnx', export_params=True, do_constant_folding=True, opset_version=11)
```
The script generates the ONNX\* model file f3net.onnx. The model conversion was tested with the repository hash commit `eecace3adf1e8946b571a4f4397681252f9dc1b8`.

## Convert ONNX* F3Net Model to IR

```sh
./mo.py --input_model <MODEL_DIR>/f3net.onnx
```