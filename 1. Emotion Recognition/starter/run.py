import cv2
import os
import argparse
import sys
import logging

from starter.flowexecutor import FlowExecutor


def __main__(argv):
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

    logging.info("OpenCV version: %s", cv2.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to the image or folder")
    parser.add_argument("-v", "--video", help="path to the video or camera id", default=0)
    args = parser.parse_args()
    flow_executor = FlowExecutor()
    if args.image:
        image_process(args.image, flow_executor)
    elif args.video or len(argv) == 0:
        read_videostream(args.video, flow_executor)
    else:
        raise ValueError('Illegal Argument! Try again.')


def image_process(source, flow_executor):
    if not os.path.isdir(source):
        execute(cv2.imread(source), flow_executor, source)
    else:
        for filename in os.listdir(source):
            image_name = os.path.join(source, filename)
            img = cv2.imread(image_name)
            if img is not None:
                execute(img, flow_executor, image_name)

    cv2.waitKey(0)


def read_videostream(option, flow_executor):
    if os.path.isfile(option):
        cap = cv2.VideoCapture(option)
    else:
        cap = cv2.VideoCapture(int(option))

    try:
        while cap.isOpened():
            # Take each frame
            _, frame = cap.read()
            execute(frame, flow_executor, 'video')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def execute(image, flow_executor, name='Result'):
    image = flow_executor.execute(image)
    cv2.imshow(name, image)


if __name__ == "__main__":
    __main__(sys.argv[1:])
