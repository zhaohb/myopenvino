// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier : Apache-2.0
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <opencv_c_wraper.h>
#include <c_api/ie_c_api.h>

struct classify_res {
    size_t class_id;
    float probability;
};

void classify_res_sort(struct classify_res *res, size_t n) {
    size_t i, j;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            if (res[i].probability < res[j].probability) {
                struct classify_res temp = res[i];
                res[i] = res[j];
                res[j] = temp;
            } else if (res[i].probability == res[j].probability && res[i].class_id > res[j].class_id) {
                struct classify_res temp = res[i];
                res[i] = res[j];
                res[j] = temp;
            }
        }
    }
}

struct classify_res *output_blob_to_classify_res(ie_blob_t *blob, size_t *n) {
    dimensions_t output_dim;
    IEStatusCode status = ie_blob_get_dims(blob, &output_dim);
    if (status != OK)
        return NULL;

    *n = output_dim.dims[1];

    struct classify_res *cls = (struct classify_res *)malloc(sizeof(struct classify_res) * (*n));
    if (!cls) {
        return NULL;
    }

    ie_blob_buffer_t blob_cbuffer;
    status = ie_blob_get_cbuffer(blob, &blob_cbuffer);
    if (status != OK) {
        free(cls);
        return NULL;
    }
    float *blob_data = (float*) (blob_cbuffer.cbuffer);

    size_t i;
    for (i = 0; i < *n; ++i) {
        cls[i].class_id = i;
        cls[i].probability = blob_data[i];
    }

    return cls;
}

void print_classify_res(struct classify_res *cls, size_t n, const char *img_path) {
    printf("\nImage %s\n", img_path);
    printf("\nclassid probability\n");
    printf("------- -----------\n");
    size_t i;
    for (i = 0; i < n; ++i) {
        printf("%zu       %f\n", cls[i].class_id, cls[i].probability);
    }
}

int main(int argc, char **argv) {
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (argc != 4) {
        printf("Usage : ./hello_classification <path_to_model> <path_to_image> <device_name>\n");
        return EXIT_FAILURE;
    }

    const char *input_model = argv[1];
    const char *input_image_path = argv[2];
    const char *device_name = argv[3];
    ie_core_t *core = NULL;
    ie_network_t *network = NULL;
    ie_executable_network_t *exe_network = NULL;
    ie_infer_request_t *infer_request = NULL;
    char *input_name = NULL, *output_name = NULL;
    ie_blob_t *imgBlob = NULL, *output_blob = NULL;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 1. Load inference engine instance -------------------------------------

    IEStatusCode status = ie_core_create("", &core);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    status = ie_core_read_network(core, input_model, NULL, &network);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------

    status = ie_network_get_input_name(network, 0, &input_name);
    if (status != OK)
        goto err;
    /* Mark input as resizable by setting of a resize algorithm.
     * In this case we will be able to set an input blob of any shape to an infer request.
     * Resize and layout conversions are executed automatically during inference */
    status |= ie_network_set_input_resize_algorithm(network, input_name, RESIZE_BILINEAR);
    status |= ie_network_set_input_layout(network, input_name, NHWC);
    status |= ie_network_set_input_precision(network, input_name, U8);
    if (status != OK)
        goto err;

    // --------------------------- Prepare output blobs ----------------------------------------------------
    status |= ie_network_get_output_name(network, 0, &output_name);
    status |= ie_network_set_output_precision(network, output_name, FP32);
    if (status !=OK)
        goto err;

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Loading model to the device ------------------------------------------
    ie_config_t config = {NULL, NULL, NULL};
    status = ie_core_load_network(core, network, device_name, &config, &exe_network);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request -------------------------------------------------
    status = ie_exec_network_create_infer_request(exe_network, &infer_request);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Prepare input --------------------------------------------------------
    /* Read input image to a blob and set it to an infer request without resize and layout conversions. */
    c_mat_t img;
    image_read(input_image_path, &img);

    dimensions_t dimens = {4, {1, (size_t)img.mat_channels, (size_t)img.mat_height, (size_t)img.mat_width}};
    tensor_desc_t tensorDesc = {NHWC, dimens, U8};
    size_t size = img.mat_data_size;
    //just wrap IplImage data to ie_blob_t pointer without allocating of new memory
    status = ie_blob_make_memory_from_preallocated(&tensorDesc, img.mat_data, size, &imgBlob);
    if (status != OK) {
        image_free(&img);
        goto err;
    }
    //infer_request accepts input blob of any size

    status = ie_infer_request_set_blob(infer_request, input_name, imgBlob);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference --------------------------------------------------------
    /* Running the request synchronously */
    status = ie_infer_request_infer(infer_request);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Process output ------------------------------------------------------
    status = ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    if (status != OK) {
        image_free(&img);
        goto err;
    }
    size_t class_num;
    struct classify_res *cls = output_blob_to_classify_res(output_blob, &class_num);

    classify_res_sort(cls, class_num);

    // Print classification results
    size_t top = 10;
    if (top > class_num) {
        top = class_num;
    }
    printf("\nTop %zu results:\n", top);
    print_classify_res(cls, top, input_image_path);

    // -----------------------------------------------------------------------------------------------------

    free(cls);
    ie_blob_free(&output_blob);
    ie_blob_free(&imgBlob);
    image_free(&img);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_name_free(&input_name);
    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);

    return EXIT_SUCCESS;
err:
    if (core)
        ie_core_free(&core);
    if (network)
        ie_network_free(&network);
    if (input_name)
        ie_network_name_free(&input_name);
    if (output_name)
        ie_network_name_free(&output_name);
    if (exe_network)
        ie_exec_network_free(&exe_network);
    if (infer_request)
        ie_infer_request_free(&infer_request);
    if (imgBlob)
        ie_blob_free(&imgBlob);
    if (output_blob)
        ie_blob_free(&output_blob);

    return EXIT_FAILURE;
}
