import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import math

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

W= 640
H = 480
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

if device_product_line == 'L500':
    W= 960
    H = 540
else:
    W= 640
    H = 480
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()


### Prepare the YOLOv5 object Detector module
print("loading the model")
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom

names = model.names
COLORS = np.random.uniform(0, 255, size=(len(names), 3))

def getResults(output, detection_threshold = 0.2):
    detection_results = output.pred[0].cpu()
    pred_classes = np.array(detection_results[:, -1])
    # get score for all the predicted objects
    pred_scores = np.array(detection_results[:, -2])
    # get all the predicted bounding boxes
    pred_bboxes = np.array(detection_results[:, :4])
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    classes_ids = detection_results[:, -1][pred_scores >= detection_threshold]
    classes = [names[int(id)] for id in classes_ids]
    scores = pred_scores[pred_scores >= detection_threshold]
    return boxes, classes, scores

def draw_boxes(image, boxes, classes, heights=None, depths = None):
    # read the image with OpenCV
    #image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[names.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        if heights is None:
            if depths is None:
                label = ""
            else:
                label =  " : D.: {:.2f} m.".format(depths[i])
        else:
            if depths is None:
                label =  " : H. :{:.2f} m.".format(heights[i])
            else:
                label =  " : H.: {:.2f} m. | D.: {:.2f} m. ".format(heights[i], depths[i])

        cv2.putText(image, "{}{}".format(classes[i],label), (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image



def getHeight(boxes):
    def processBox(bbox):
        obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
        zs = obj_points[:, 2]
        ys = obj_points[:, 1]
        z = np.median(zs)
        ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1)))
        my = np.amin(ys, initial=1)
        My = np.amax(ys, initial=-1)
        height = (My - my)
        return height
    heights = []
    for b in boxes:
        heights.append(processBox(b))
    return heights

def getDepth(boxes):
    def processBox(bbox):
        x, y = int((bbox[0] + bbox[2]) /2), int((bbox[1] + bbox[3])/2)
        depth = depth_frame.get_distance(x, y)
        dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth)
        distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
        return distance
    depths_objects = []
    for b in boxes:
        depths_objects.append(processBox(b))
    return depths_objects
# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames) ## NEW

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        points = point_cloud.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        output = model(color_image)
        boxes, classes, scores = getResults(output, detection_threshold=0.2)
        heights = getHeight(boxes)
        depths_objects = getDepth(boxes)

        #color_image = draw_boxes(color_image, boxes, classes, heights=heights, depths=depths_objects)
        color_image = draw_boxes(color_image, boxes, classes, depths=depths_objects)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
