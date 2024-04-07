# Edge TPU Object Tracker Example

Fork and modifications of Coral examples. Modified to do demo of delective object tracking and centroid depth extraction.

## Installation

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2.  Clone this Git repo onto your computer:

    ```
    git clone https://github.com/jc-cr/example-object-tracker.git

    cd example-object-tracker/
    ```

3.  Download the models:

    ```
    sh download_models.sh
    ```

    These models will be downloaded to a new folder
    ```models```.

## Run the detections

Importantly, you should have the latest TensorFlow Lite runtime installed
(as per the [Python quickstart](
https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
using the ```pip3 show tflite_runtime``` command.

1. CD into the gstreamer folder
    ```
    cd gstreamer
    ```

2.  Install the GStreamer libraries and Trackers:

    ```
    bash install_requirements.sh
    ```
3.  Run the detection model with Sort tracker
    ```
    python3 detect.py --tracker sort --target person --threshold 0.25 --videosrc /dev/video4
    ```

In the above command we use `/dev/video41 to access the RGB stream from Intel 435i.
If usign other depth camera, you could find available video sources using the command `v4l2-ctl --list-devices --verbose`



## Contents

  * __gstreamer__: Python examples using gstreamer to obtain camera stream. These
    examples work on Linux using a webcam, Raspberry Pi with
    the Raspicam, and on the Coral DevBoard using the Coral camera. For the
    former two, you will also need a Coral USB Accelerator to run the models.

    This demo provides the support of an Object tracker. After following the setup 
    instructions in README file for the subfolder ```gstreamer```, you can run the tracker demo:


## Models

For the demos in this repository you can change the model and the labels
file by using the flags flags ```--model``` and
```--labels```. Be sure to use the models labeled _edgetpu, as those are
compiled for the accelerator -  otherwise the model will run on the CPU and
be much slower.


For detection you need to select one of the SSD detection models
and its corresponding labels file:

```
mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
```
