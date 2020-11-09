# video record through webcam

# Resources
# initial github repo:
# https://github.com/pavan097/webcam_video/blob/master/videoCap.py
# more extensive GUI with object detection:
# https://github.com/iRTEX-MIT/OpenCV-Webcam-Recorder-and-Streamer
# managing fps recording and multi-threading:
# https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2

import os
import sys
import re
import cv2
import numpy as np
import time
from datetime import datetime

#presets
GRAY_SCALE = False
COLLECT_VIDEO = True
LIMIT_FRAME_RATE = True
LIMIT_METHOD = 1
# 0 using the time difference to control the fps.
# 1 using a fixed multipule of the camera frame rate. Effectively down sampling.
FPS_LIMIT = 10

HARD_CODE_CAMERA_FPS = False # if you want to input the camera fps
CAMERA_FPS = 30 # sets the hardcoded camera fps if used.

GUESS_FPS_STANDARD = True # uses the measured fps to compare to a standard list of fps values.
N_TEST = 30 # number of frames to use to estimate the camera fps.
STANDARD_FPS_VALUES = [10, 15, 20, 24, 25, 30, 60, 120] # list of standard camera fps

# collection schedule
N_FRAMES = 100

def video_cap():
    c = cv2.VideoCapture(0) # 0 for the inbuilt camera i.e webcam
    # check camera fps
    fps = get_camera_fps(
        cv2VideoCapture=c,
        N_test=N_TEST,
        guess_fps_standard=GUESS_FPS_STANDARD
        )
    print('Cameras max fps is determined to be:',fps)
    # if HARD_CODE_CAMERA_FPS is True then ignore the calculated fps and use the preset CAMERA_FPS
    if HARD_CODE_CAMERA_FPS:
        fps = CAMERA_FPS
    # series of snapshots (for video)
    if COLLECT_VIDEO:
        if LIMIT_FRAME_RATE: # limit the frame rate
            if LIMIT_METHOD == 0:
                # because the clocks arent matched there is frame dropping occuring in
                # this technique.
                # If possible use the non rate limited technique.
                start_time = time.time()
                i = 0
                while i < N_FRAMES:
                    time_elapsed = time.time() - start_time
                    r, frame = c.read()
                    if time_elapsed > 1./FPS_LIMIT:
                        start_time = time.time()
                        save_frame(frame)
                        i += 1
                    # exit the capture loop?
                    if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
                        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
                        # exit the videocap loop if user enters 'q'
                        c.release()
                        cv2.destroyAllWindows()
                        return
            if LIMIT_METHOD == 1:
                # This method reachs the desired fps rate. However requires knowledge of the
                # Camera fps.
                n = 0
                N = fps // FPS_LIMIT
                i = 0
                while i < N_FRAMES:
                    r, frame = c.read()
                    if n % N == 0:
                        save_frame(frame)
                        i += 1
                    # exit the capture loop?
                    if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
                        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
                        # exit the videocap loop if user enters 'q'
                        c.release()
                        cv2.destroyAllWindows()
                        return
                    n += 1
        else: # dont limit the frame rate
            for i in range(N_FRAMES):
                r, frame = c.read()
                save_frame(frame)
                # exit the capture loop?
                if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
                # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
                    # exit the videocap loop if user enters 'q'
                    c.release()
                    cv2.destroyAllWindows()
                    return
        # calculate the framerate based on saved images.
        calc_frame_rate_metrics()
    else:
        # single snapshot
        _, frame = c.read()
        save_frame(frame)
        if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
            # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            c.release()
            cv2.destroyAllWindows()
            return
    cv2.waitKey(1)
    c.release()
    cv2.destroyAllWindows()

# HIDEEN CODE HERE DOES NOT WORK
# def check_camera_fps(cv2VideoCapture):
#     # from https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#     if int(major_ver)  < 3 :
#         fps = cv2VideoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
#         print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
#     else :
#         fps = cv2VideoCapture.get(cv2.CAP_PROP_FPS)
#         print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#     return fps

def get_camera_fps(cv2VideoCapture, N_test: int, guess_fps_standard: bool) -> int:
    """finds the frame per second the camera can record.

    Args:
        cv2VideoCapture (cv2.VideoCapture): the webcam video capturer
        N_test (int): the number of frames to collect and average over.
        guess_fps_standard (bool): if True, the standard fps is determined by squared error.

    Returns:
        int: average fps of the camera.
    """
    fps = calc_camera_fps(cv2VideoCapture, N_test)
    if guess_fps_standard:
        # find the closest standard value fps
        possible_values = np.array(STANDARD_FPS_VALUES)
        squared_error = (possible_values - fps)**2
        min_index = np.argmin(squared_error)
        fps = int(possible_values[min_index])
    return fps


def calc_camera_fps(cv2VideoCapture, N_test: int) -> float:
    """Calculates the average frame rate for N frames collected

    Args:
        cv2VideoCapture (cv2.VideoCapture): the webcam video capturer
        N_test (int): the number of frames to collect and average over.

    Returns:
        float: average fps
    """
    start_time = time.time()
    for i in range(N_test):
        _, _ = cv2VideoCapture.read()

        # # exit the capture loop?
        # if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
        # # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
        #     # exit the videocap loop if user enters 'q'
        #     cv2VideoCapture.release()
        #     cv2.destroyAllWindows()
        #     return
    end_time = time.time()
    elapsed_time = end_time - start_time # total time in sec for N_test frames collected
    period = elapsed_time / (N_test-1)  # average period for N_test-1 periods.
    fps = 1./period # average fps for N_test-1 periods.
    return fps

def save_frame(frame):
    """Show the image, convert to gray-scale if desired.

    Args:
        frame (np.array): 2D numpy array of the image
    """
    if GRAY_SCALE: # capture gray-scale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray
        cv2.imshow('frame :',gray)
        save_image(gray)
    else: # capture RGB images
        cv2.imshow('frame :',frame)
        save_image(frame)

def save_image(img):
    """saves the image to the images folder.

    Args:
        img (np.array): 2D numpy array image

    Raises:
        Exception: Could not write image
    """
    base_path =  os.getcwd()
    # image_name = re.sub(' ','_',str(datetime.now()).split('.')[0])
    image_name = str(datetime.now()).replace(' ','-').replace(':','-').replace('.','-')  #'test'
    images_path = os.path.join(base_path,'images')
    image_path = os.path.join(base_path,'images',image_name+'.png')
    #print('ready to save image:'+image_path)
    if os.path.isdir(images_path):
        # if not cv2.imwrite(r'C:\Users\jesse\OneDrive\Entrepreneurs First\Adaptive Exergames\Video Recording\Test Record\images\test.png',img):
        #     raise Exception("Could not write image")
        if not cv2.imwrite(image_path,img):
            raise Exception("Could not write image")
        #print('image saved')

def calc_frame_rate_metrics():
    """calculate the fps rate that the datarecording achieved.
    """
    base_path =  os.getcwd()
    images_path = os.path.join(base_path,'images')
    file_names = os.listdir(images_path)
    times =[] # from the start of the day
    for file_name in file_names:
        date = file_name[:-4].split('-')
        time = (int(date[-1]) + 1E6 * (int(date[-2]) + 60 * (int(date[-3]) + 60 * (int(date[-4])))))
        times.append(time)
    times_np = np.array(times)
    delta_times = times_np[1:] - times_np[0:-1]
    print(delta_times)
    T = delta_times.mean() # mean period
    delta_T = delta_times.std() # period stardard deviation
    print(T, delta_T)
    fps = 1E6/T
    delta_fps = fps * ( delta_T / T )
    print(fps, delta_fps)

if __name__ == "__main__":
    video_cap()