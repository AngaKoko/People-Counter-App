# Project Write-Up

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

ssd_mobilenet_v2_coco_2018_03_29 TensorFlow model was used for the project and the model can be downloaded from this link http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

Model was coverted to IR using this command:

`python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json`

## Explaining Custom Layers

Custom layers are layers not in the list of known layers in the Intel® Distribution of OpenVINO™ toolkit. The Inference Engine will consider these layers as unsupported and  report an error when these layers are loaded into a device plugin from the input model IR files. An extension for both Model Optimizer and Inference Engine is needed when implementing custom layers for a pre-trained model in the Intel® Distribution of OpenVINO™ toolkit. 

The extension needed by the ModelOptimizer to extract information from input model, optimize the model, and finally output optimized model in the model IR for the Inference Engine to run the model are:
1. Custom Layer Extractor 
2. Custom Layer Operation

For Inference Engine the extension to be implemented is based on the target device. It could be either:
1. Custom Layer CPU Extension
    A compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.

2. Custom Layer GPU Extension
    OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

                                pre-conversion  |   post-conversion
`Accuracy                    |       NA          |       NA`
`Model Size                  |     69.7mb        |     33mb`
`FPS                         |      4.06         |      5.26`
`Latency (ms)                |     235.28        |     186.17`
`Total Excution time (ms)    |    20457.53       |    20329.06`
`Batch                       |       1           |       1`
`Streams                     |       1           |       1`

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

1. Recording the number of homeless people in an area. 
2. Determine the item getting the most attention in a museum or store
3. Count the number of people in a room.
4. Determine which task takes the most time or which staff spends the most time performing a specific task. 

Each of these use cases would be useful because the app provides data to help users make better decision

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. 

Poor lighting will reduce visibility of objects in the image making it difficult for models to detect objects. The accuracy of the model is important for object detection. A model with lower accuracy can classify a cat as a dog, or a lamp stand as human. This will affect output data from the app. Also imputed image size is important because the model works with a particular image size. We will have to resize the input image size to the size the model can work with.


