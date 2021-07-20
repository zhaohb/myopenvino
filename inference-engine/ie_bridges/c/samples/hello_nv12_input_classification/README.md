# Hello NV12 Input Classification C Sample {#openvino_inference_engine_ie_bridges_c_samples_hello_nv12_input_classification_README}

This topic describes how to run the Hello NV12 Input Classification sample application.
The sample demonstrates how to use the new NV12 automatic input pre-processing API of the Inference Engine in your applications.
Refer to [Integrate the Inference Engine New Request API with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.

## How It Works

Upon the start-up, the sample application reads command-line parameters, loads a network and sets an
image in the NV12 color format to an Inference Engine plugin. When inference is done, the
application outputs data to the standard output stream.

The sample accepts an uncompressed image in the NV12 color format. To run the sample, you need to
convert your BGR/RGB image to NV12. To do this, you can use one of the widely available tools such
as FFmpeg\* or GStreamer\*. The following command shows how to convert an ordinary image into an
uncompressed NV12 image using FFmpeg:
```sh
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

> **NOTE**:
>
> * Because the sample reads raw image files, you should provide a correct image size along with the
>   image path. The sample expects the logical size of the image, not the buffer size. For example,
>   for 640x480 BGR/RGB image the corresponding NV12 logical image size is also 640x480, whereas the
>   buffer size is 640x720.
> * The sample uses input autoresize API of the Inference Engine to simplify user-side
>   pre-processing.
> * By default, this sample expects that network input has BGR channels order. If you trained your
>   model to work with RGB order, you need to reconvert your model using the Model Optimizer tool
>   with `--reverse_input_channels` argument specified. For more information about the argument,
>   refer to **When to Reverse Input Channels** section of
>   [Converting a Model Using General Conversion Parameters](../../../../../docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

To run the sample, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the
> Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

You can perform inference on an NV12 image using a trained AlexNet network on CPU with the following command:
```sh
./hello_nv12_input_classification_c <path_to_model>/alexnet_fp32.xml <path_to_image>/cat.yuv 640x480 CPU
```

## Sample Output

The application outputs top-10 inference results.
