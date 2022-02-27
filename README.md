# human_detection

A script for turning your webcam into a security camera.  Uses the HOG SVM algorithm from cv2.  Based on code from https://data-flair.training/blogs/python-project-real-time-human-detection-counting/, although at time of writing there were a few bugs there and the functionality was somewhat different.

Use detect(frame) to analyse individual frames; humanDetector(args) to run from another script.  args is a parser.args object but could be any object with .beep and .save flags to check.
