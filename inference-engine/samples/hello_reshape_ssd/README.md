# Hello Reshape SSD C++ Sample {#openvino_inference_engine_samples_hello_reshape_ssd_README}

This topic demonstrates how to run the Hello Reshape SSD application, which does inference using object detection
networks like SSD-VGG. The sample shows how to use [Shape Inference feature](../../../docs/IE_DG/ShapeInference.md).

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

You can use the following command to do inference on CPU of an image using a trained SSD network:
```sh
./hello_reshape_ssd <path_to_model>/ssd_300.xml <path_to_image>/500x500.bmp CPU 3
```

## Sample Output

The application renders an image with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

## See Also
* [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
* [Model Downloader](@ref omz_tools_downloader)
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
