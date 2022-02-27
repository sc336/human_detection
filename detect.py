#!./venv/bin/python
import logging
from time import sleep
from datetime import datetime
from argparse import ArgumentParser
import cv2
import imutils
import numpy as np
import chime

# Command line arguments

parser = ArgumentParser()
parser.add_argument('-b', '--beep', help='Beep when someone new is detected', action='store_true')
parser.add_argument('-s', '--save', help='Save each frame with a new detection', action='store_true')
parser.add_argument('-c', '--camera', help='Specify which camera to use (default 0)', type=int, default=0)
parser.add_argument('-d', '--delay', help='Specify a delay in seconds between checks (reduces power consumption)', type=int, default=0)
parser.add_argument('-r', '--rotate', help='Number of times to rotate image anticlockwise by 90 degrees', type=int, default=0)
parser.add_argument('-t', '--threshold', help='Detection threshold: higher is stricter (fewer false positives, more false negatives)', type=float, default=0.8)

ROTATION = [None, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.cv2.ROTATE_180, cv2.cv2.ROTATE_90_CLOCKWISE]

args = parser.parse_args()

# Logging

log_handlers = []
log_formatter = logging.Formatter('%(asctime)s : %(funcName)s : %(message)s')
quick_log = logging.FileHandler(f'log')
quick_log.setFormatter(log_formatter)
log_handlers += [quick_log]
logging.basicConfig(handlers=log_handlers, level=logging.DEBUG)

# chime config
chime.theme('mario')

# HOG config

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect(frame):
    """
    Detect any humans within frame using the configured HOG SVM algorithm and return
    (frame with persons highlighted in a box, number of persons detected)
    """
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03, hitThreshold=args.threshold)

    persons = 0
    for ((x,y,w,h), p) in zip(bounding_box_cordinates, weights):
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {persons+1}, probability {p}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        persons += 1

    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {persons}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)
    return frame, persons

def humanDetector(args_passed):
    """
    Run the detection program
    args:
        beep (bool): beep when a new person is detected
        save (bool): save frames when a new person is detected in screens/{timestamp}.png
    """
    global args
    args = args_passed    #FIXME: this seems messy; also validation
    writer = cv2.VideoWriter('output', cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    detectLoop(writer)

def detectLoop(writer):
    """
    Grab the lowest-numbered webcam and loop detection on the frames it provides.
    Output marked frames to writer
    """
    logging.debug(f'Camera {args.camera}')
    video = cv2.VideoCapture(args.camera)
    logging.info('Detecting humans')

    persons = 0
    while True:
        check, frame = video.read()
        if args.rotate in [1, 2, 3]:
            frame = cv2.rotate(frame, ROTATION[args.rotate])
        frame, detected_persons = detect(frame)
        if detected_persons > persons:
            cv2.imwrite('./latest.png', frame)
            logging.info('Detected new person')
            if args.beep:
                chime.info()
            if args.save:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.imwrite(f'screens/{timestamp}.png', frame)

        persons = detected_persons

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        sleep(args.delay)

    #TODO: refactor using with
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    humanDetector(args)
