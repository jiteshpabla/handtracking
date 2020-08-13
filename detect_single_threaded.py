from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument(
        '-frame',
        '--frame_path',
        dest='frame_path',
        default=0,
        help='Frames folder path.')
    parser.add_argument(
        '-filename',
        '--video',
        dest='video_name',
        default=0,
        help='Video name.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2
    
    #cannot open window in google colab
    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    ret= True
    count=0
    while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #frame may not be read properly; a do-while type implemetation to avoid crashes
        ret, image_np = cap.read()
        #to save the original image
        temp_image= image_np
        if ret:
            # image_np = cv2.flip(image_np, 1)
            if not image_np is None:
                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    #print("converting to RGB")
                except:
                    print("Error converting to RGB")

            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(image_np,
                                                          detection_graph, sess)
            
            count += 1
            # draw bounding boxes on frame
            detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                             scores, boxes, im_width, im_height,
                                             image_np, temp_image, count, args.frame_path, args.video_name)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time
        else:
            continue
