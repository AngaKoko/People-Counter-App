"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from fysom import *
from imutils.video import FPS

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Browser and OpenCV Window toggle
Browser_ON = False

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

fsm = Fysom({'initial': 'empty',
             'events': [
                 {'name': 'enter', 'src': 'empty', 'dst': 'standing'},
                 {'name': 'exit',  'src': 'standing',   'dst': 'empty'}]})


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    parser.add_argument("-o", "--output", type=str, default="LOCAL",
                        help="Output window local or Web Server (use -o WEB)"
                        "(LOCAL by default)")
    return parser

def performance_counts(perf_count):
    """
    print information about layers of the model.

    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))

def ssd_out(frame, result):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    exit_count = 0

    entre_ROI_xmin = 400
    entre_ROI_ymin = 450
    exit_ROI_xmin = 550
    exit_ROI_ymin = 410

    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold 
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 2)
            if xmin < entre_ROI_xmin and ymax < entre_ROI_ymin: 
                 if fsm.current == "empty":
                    current_count = current_count + 1 
                    fsm.enter()

            if xmin > exit_ROI_xmin and ymax < exit_ROI_ymin:
                if fsm.current == "standing":
                    # Change the state to exit - fsm state change
                    fsm.exit()
                    exit_count = exit_count +1
                
    return frame, current_count, exit_count

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), 
        interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes

def connect_mqtt():
    ###: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    global prob_threshold, initial_w, initial_h

    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device)

    # Get a Input shape #
    n, c, h, w = infer_network.get_input_shape()
    
    # Handle the input stream#

    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0
    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input
    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
        
    prob_threshold = args.prob_threshold

    # Flag for the input image
    single_image_mode = False

    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    duration = 0

    fps = FPS().start()

    #  Loop until stream is over ###
    while cap.isOpened():
        #  Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        initial_h = frame.shape[0]
        initial_w = frame.shape[1]
        key_pressed = cv2.waitKey(50)
    
        # Start asynchronous inference for specified request ###
        image = cv2.resize(frame, (w, h))
        #  Pre-process the image as needed ###
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

         ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(image)
        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id)
            
            ### TODO: Extract any desired stats from the results ###
            frame, current_count, exit_count = ssd_out(frame, result)
            inf_time_message = "Inference time: {:.3f}ms" .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > exit_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
                duration = 0

            if current_count < exit_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                client.publish("person", json.dumps({"count": current_count}))
            
            last_count = current_count

            cv2.putText(frame, "Total cound: "+str(total_count), (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Duration: "+str(duration), (15, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            if key_pressed == 27:
                break

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        ### Send the frame to the FFMPEG server ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        else:
            if args.output == "WEB":
                # Push to FFmpeg server
                sys.stdout.buffer.write(frame)

                sys.stdout.flush()
            else:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                #Break if escape key pressed
                if key_pressed == 27:
                    break
        
        fps.update()
    
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
    fps.stop()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
