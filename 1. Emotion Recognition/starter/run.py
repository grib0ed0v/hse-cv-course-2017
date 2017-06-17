#!/usr/bin/python
import cv2, os
import argparse, sys

#from preprocessor.chain import ProcessorChain
from starter.flowexecutor import FlowExecutor

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to the image or folder")
    parser.add_argument("-v", "--video", help="path to the video or camera id", default=0)
    args = parser.parse_args()
    if args.image:
        image_process(args.image)
    elif args.video or len(argv)==0:
        read_videostream(args.video)
    else:
        print ('Illegal Argument! Try again.')

def image_process(source):
    flow = FlowExecutor()

    if (os.path.isdir(source) == False):
        image =flow.execute(cv2.imread(source))
        cv2.imshow(source, image)
        cv2.waitKey(0)
    else:
        for filename in os.listdir(source):
            # print filename
            img = cv2.imread(os.path.join(source, filename))
            if img is not None:
                flow.execute(img)


def read_videostream(option):
    flow = FlowExecutor()
    if os.path.isfile(option):
        cap = cv2.VideoCapture(option)
    else:
        id = int(option)
        cap = cv2.VideoCapture(id)
    while (1):
        # Take each frame
        _, frame = cap.read()
        flow.execute(frame)
    #exeption
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])