import argparse
import collections
import common
import gstreamer
import numpy as np
import os
import re
import svgwrite
import time


import pyrealsense2 as rs
import numpy as np
import sys

Object = collections.namedtuple('Object', ['id', 'score', 'bbox', 'centroid'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

class RealsenseDepthExtractor:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Configure the pipeline to stream depth information
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.dev = None

        try:
            # Start the pipeline once with the configured streams
            self.profile = self.pipeline.start(self.config)
            self.dev = self.profile.get_device()

            depth_sensor = self.dev.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)
            print("Depth extractor initialized and pipeline started.")
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {str(e)}")

    def stop_device(self):
        try:
            self.pipeline.stop()
            self.dev.hardware_reset()
            print("RealSense pipeline stopped.")
        except Exception as e:
            print(f"Failed to stop RealSense pipeline: {str(e)}")

    def get_depth_at_point(self, x, y):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                print("No depth frame available.")
                return None
            depth = depth_frame.get_distance(x, y)
            return depth * 1000  # Convert meters to millimeters
        except Exception as e:
            print(f"Error retrieving depth data: {str(e)}")
            return None


class DetectionVisualizer:
    def __init__(self):
        self.boxes = None
        self.category_ids = None
        self.scores = None
        self.centroid_depth = 0
        self.depth_extractor = RealsenseDepthExtractor()

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def generate_svg(self, src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag):
        dwg = svgwrite.Drawing('', size=src_size)
        src_w, src_h = src_size
        inf_w, inf_h = inference_size
        box_x, box_y, box_w, box_h = inference_box
        scale_x, scale_y = src_w / box_w, src_h / box_h

        for y, line in enumerate(text_lines, start=1):
            self._add_text_with_shadow(dwg, 10, y*20, line)

        if trackerFlag and (np.array(trdata)).size:
            for td in trdata:
                x0, y0, x1, y1, trackID = td[0].item(), td[1].item(
                ), td[2].item(), td[3].item(), td[4].item()
                overlap = 0
                for ob in objs:
                    dx0, dy0, dx1, dy1 = ob.bbox.xmin.item(), ob.bbox.ymin.item(
                    ), ob.bbox.xmax.item(), ob.bbox.ymax.item()
                    area = (min(dx1, x1)-max(dx0, x0))*(min(dy1, y1)-max(dy0, y0))
                    if (area > overlap):
                        overlap = area
                        obj = ob

                # Relative coordinates.
                x, y, w, h = x0, y0, x1 - x0, y1 - y0

                # Absolute coordinates, input tensor space.
                x, y, w, h = int(x * inf_w), int(y *
                                                inf_h), int(w * inf_w), int(h * inf_h)
                # Subtract boxing offset.
                x, y = x - box_x, y - box_y

                # Scale to source coordinate space.
                x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y

                percent = int(100 * obj.score)

                label = '{}% {} ID:{}'.format(
                    percent, labels.get(obj.id, obj.id), int(trackID))

                self._add_text_with_shadow(dwg, x, y - 5, label)

                # Drawing centroid for each object
                centroid_x = obj.centroid[0]
                centroid_y = obj.centroid[1]
                centroid_x_scaled = centroid_x * src_w  # Scale to source width
                centroid_y_scaled = centroid_y * src_h  # Scale to source height

                # Extract depth at the centroid
                self.centroid_depth = self.depth_extractor.get_depth_at_point(
                    int(centroid_x_scaled), int(centroid_y_scaled))

                if self.centroid_depth is None:
                    self.centroid_depth = 0

                # Convert centroid coordinates to source space
                # Drawing the centroid on the SVG with a larger radius and a highly visible color
                dwg.add(dwg.circle(center=(centroid_x_scaled, centroid_y_scaled), r=10, fill='blue'))
                dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                                fill='none', stroke='red', stroke_width='2'))

                # Add text with centorid coordinates below the centroid
                self._add_text_with_shadow(dwg, centroid_x_scaled + 10, centroid_y_scaled + 20, 'Centroid: ({}, {}, {})'.format(
                    int(centroid_x_scaled), int(centroid_y_scaled), int(self.centroid_depth)), font_size=15)
        
        # If no tracker is used, dont draw the tracking data
        else:
            for obj in objs:
                x0, y0, x1, y1 = list(obj.bbox)
                # Relative coordinates.
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
                # Absolute coordinates, input tensor space.
                x, y, w, h = int(x * inf_w), int(y *
                                                inf_h), int(w * inf_w), int(h * inf_h)
                # Subtract boxing offset.
                x, y = x - box_x, y - box_y
                # Scale to source coordinate space.
                x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
                percent = int(100 * obj.score)
                label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
                self._add_text_with_shadow(dwg, x, y - 5, label)

                # Drawing centroid for each object
                centroid_x = obj.centroid[0]
                centroid_y = obj.centroid[1]
                centroid_x_scaled = centroid_x * src_w  # Scale to source width
                centroid_y_scaled = centroid_y * src_h  # Scale to source height

                # Extract depth at the centroid
                self.centroid_depth = self.depth_extractor.get_depth_at_point(
                    int(centroid_x_scaled), int(centroid_y_scaled))

                # Convert centroid coordinates to source space
                # Drawing the centroid on the SVG with a larger radius and a highly visible color
                dwg.add(dwg.circle(center=(centroid_x_scaled, centroid_y_scaled), r=10, fill='yellow'))

                dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                        fill='none', stroke='red', stroke_width='2'))

                
                # Add text with centorid coordinates below the centroid
                self._add_text_with_shadow(dwg, centroid_x_scaled + 10, centroid_y_scaled + 20, 'Centroid: ({}, {}, {})'.format(
                    int(centroid_x_scaled), int(centroid_y_scaled), int(self.centroid_depth)), font_size=15)

        return dwg.tostring()

    def get_output(self, interpreter, score_threshold, top_k, image_scale=1.0):
        """Returns list of detected objects."""
        self.boxes = common.output_tensor(interpreter, 0)
        self.category_ids = common.output_tensor(interpreter, 1)
        self.scores = common.output_tensor(interpreter, 2)

        return [self._make(i) for i in range(top_k) if self.scores[i] >= score_threshold]

    def user_callback(self, input_tensor, src_size, inference_box, mot_tracker):
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = self.get_output(interpreter, args.threshold, args.top_k)
        end_time = time.monotonic()
        detections = []  # np.array([])

        if args.target:
            objs = [obj for obj in objs if labels.get(obj.id) == args.target]

        for n in range(0, len(objs)):
            element = []  # np.array([])
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            element.append(objs[n].centroid[0])
            element.append(objs[n].centroid[1])
            element.append(objs[n].score)  # print('element= ',element)
            detections.append(element)  # print('dets: ',dets)

        detections = np.array(detections)
        trdata = []
        trackerFlag = False

        if detections.any():
            if mot_tracker != None:
                trdata = mot_tracker.update(detections)
                trackerFlag = True
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(fps_counter))), ]

        if len(objs) != 0:

            return self.generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag)

    def _add_text_with_shadow(self, dwg, x, y, text, font_size=20):

        dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
        dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))

    def _make(self, index):
        ymin, xmin, ymax, xmax = self.boxes[index]

        centroid_x = (xmin + xmax) / 2
        centroid_y = (ymin + ymax) / 2
        centroid = (centroid_x, centroid_y)

        return Object(
            id=int(self.category_ids[index]),
            score=self.scores[index],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                    ymin=np.maximum(0.0, ymin),
                    xmax=np.minimum(1.0, xmax),
                    ymax=np.minimum(1.0, ymax)),
            centroid=centroid)


if __name__ == '__main__':

    try:
        detection_viz = DetectionVisualizer()

        default_model_dir = '../models'
        default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        default_labels = 'coco_labels.txt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='.tflite model path',
                            default=os.path.join(default_model_dir, default_model))
        parser.add_argument('--labels', help='label file path',
                            default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('--top_k', type=int, default=3,
                            help='number of categories with highest score to display')
        parser.add_argument('--threshold', type=float, default=0.1,
                            help='classifier score threshold')
        parser.add_argument('--videosrc', help='Which video source to use. ',
                            default='/dev/video0')
        parser.add_argument('--videofmt', help='Input video format.',
                            default='raw',
                            choices=['raw', 'h264', 'jpeg'])
        parser.add_argument('--tracker', help='Name of the Object Tracker To be used.',
                            default=None,
                            choices=[None, 'sort'])
        parser.add_argument(
            '--target', help='Target object label to track', type=str)

        args = parser.parse_args()

        print('Loading {} with {} labels.'.format(args.model, args.labels))
        interpreter = common.make_interpreter(args.model)
        interpreter.allocate_tensors()
        labels = detection_viz.load_labels(args.labels)

        w, h, _ = common.input_image_size(interpreter)
        inference_size = (w, h)
        # Average fps over last 30 frames.
        fps_counter = common.avg_fps_counter(30)

        result = gstreamer.run_pipeline(detection_viz.user_callback,
                                        src_size=(640, 480),
                                        appsink_size=inference_size,
                                        trackerName=args.tracker,
                                        videosrc=args.videosrc,
                                        videofmt=args.videofmt)

    except Exception as e:
        detection_viz.depth_extractor.stop_device()
        print(f"Error stopping device: {str(e)}")

    finally:
        detection_viz.depth_extractor.stop_device()
        print('Cleanup complete, exiting.')
        sys.exit(0)