# video record through webcam

# Resources
# initial github repo:
# https://github.com/pavan097/webcam_video/blob/master/videoCap.py
# more extensive GUI with object detection:
# https://github.com/iRTEX-MIT/OpenCV-Webcam-Recorder-and-Streamer
# managing fps recording and multi-threading:
# https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2

import os
import glob
import sys
import re
import json
import cv2
import numpy as np
import time
from datetime import datetime

# Global Variables

# record details

# webcam feed
WEBCAM = None
IMG_SIZE = None
# presets
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
RECORDING_FPS = None

# implemented states
IMPLEMENTED_STATES = [
    "setup",
    "action_prompt",
    "action_countdown",
    "edge_window",
    "action_record",
    "done"
]

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

def schedule_cap():
    # global variables that are updated
    global RECORDING_FPS
    global WEBCAM
    global IMG_SIZE
    # set the webcama feed
    WEBCAM = cv2.VideoCapture(0) # 0 for the inbuilt camera i.e webcam
    # get camera fps
    fps = get_camera_fps(
        cv2VideoCapture=WEBCAM,
        N_test=N_TEST,
        guess_fps_standard=GUESS_FPS_STANDARD
    )
    print('Cameras max fps is determined to be:', fps)
    # set camera recording fps
    RECORDING_FPS = fps # TODO: currently collect at max frame rate
    # set record method
    # TODO: currently collect at max frame rate
    # get image resolution
    IMG_SIZE = get_camera_resolution(WEBCAM)
    # load Action_Recording_Schedule.json
    with open('Action_Recording_Schedule.json') as schedule_file:
        schedule_data = json.load(schedule_file)
    # unpack schedule structure
    action_types = schedule_data["action_types"]
    state_types = schedule_data["state_types"]
    schedule = schedule_data["schedule"]
    # check schedule structure
    check_action_types(action_types)
    check_state_types(state_types)
    check_schedule_state_and_actions(action_types, state_types, schedule)
    # check for thumbnails
    # create missing thumbnails
    check_thumbnails(action_types)
    # go through the schedule:
    for schedule_item in schedule:
        # TODO: check if the state is action_prompt and action is random
        # execute schedule_item
        execute_schedule_item(schedule_item)
        # TODO: if the state is action_record clear prev_action
    # end and close windows
    cv2.waitKey(1)
    WEBCAM.release()
    cv2.destroyAllWindows()

def execute_schedule_item(item: list):
    state, action, time = tuple(item)
    if state == 'setup':
        execute_setup(action, time)
    elif state == 'action_prompt':
        execute_action_prompt(action, time)
    elif state == 'action_record':
        execute_action_record(action, time)
    elif state == 'done':
        execute_done(action, time)
    else:
        raise Exception("unknown schedule state.")

def check_action_types(action_types: list) -> bool:
    base_path =  os.getcwd()
    action_images_path = os.path.join(base_path,'action_images')
    if not os.path.isdir(action_images_path):
        raise Exception("No action_images folder")
    else:
        file_names = os.listdir(action_images_path)
        action_images = []
        for file_name in file_names:
            action_images.append(file_name.split('.')[0])
        if set(action_types) <= set(action_images + ['random']):
            return True
        else:
            raise Exception("schedule specifies an unknown action type.")

def check_state_types(state_types: list) -> bool:
    if not IMPLEMENTED_STATES:
        raise Exception("No IMPLEMENTED_STATES list")
    else:
        if set(state_types) <= set(IMPLEMENTED_STATES):
            return True
        else:
            raise Exception("schedule specifies an unknown state type.")

def check_schedule_state_and_actions(action_types: list, state_types: list,schedule: list) -> bool:
    for item in schedule:
        if not item[0] in state_types:
            raise Exception("schedule uses an unknown state type.")
        if item[1] == '':
            pass
        elif not item[1] in action_types:
            raise Exception("schedule uses an unknown state action.")
    return True

def check_thumbnails(actions: list) -> bool:
    base_path =  os.getcwd()
    silhouette_path = os.path.join(base_path, 'thumbnails', 'ideal_stance(silhouette).png')
    if not os.path.isfile(silhouette_path):
        raise Exception("No silhouette image")
    for action in actions:
        if action == 'random':
            pass
        else:
            action_thumbnail_path = os.path.join(base_path, 'thumbnails', action +'.png')
            if not os.path.isfile(action_thumbnail_path):
                create_thumbnail(action)

def create_thumbnail(action: str):
    base_path =  os.getcwd()
    action_image_path = glob.glob(os.path.join(base_path,'action_images', action +'.*'))[0]
    img = cv2.imread(action_image_path, cv2.IMREAD_COLOR)
    # TODO: reshape to the same size as IMG_SIZE, adding black space if needed.

def load_silhouette_image():
    base_path =  os.getcwd()
    silhouette_path = os.path.join(base_path, 'thumbnails', 'ideal_stance(silhouette).png')
    img = cv2.imread(silhouette_path, cv2.IMREAD_COLOR)
    try:
        if isinstance(img,np.ndarray):
            return img
    except:
        if img == None:
            raise Exception("silhouette image not loaded properly")
        else:
            raise Exception("unknown error")

def load_action_image(filename: str):
    base_path =  os.getcwd()
    file_path = os.path.join(base_path, 'thumbnails', filename + '.png')
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    try:
        if isinstance(img,np.ndarray):
            return img
    except:
        if img == None:
            raise Exception("action image not loaded properly")
        else:
            raise Exception("unknown error")

def mask_on_black(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_black_mask = b == 0 # TODO: replace with [0,0,0] check
    exclusive_sum = a*b_black_mask + b * (~b_black_mask)
    return exclusive_sum

def execute_setup(action, time):
    # determine N_frames
    N_frame = determine_N_frames(time)
    # load silhouette image
    silhouette_img = load_silhouette_image()
    # for frame in N_frames:
    for i in range(N_frame):
        # set countdown timer
        countdown = int(time * (1 - i/N_frame))
        # record webcam frame
        r, frame = WEBCAM.read()
        # overlap silouette
        frame = mask_on_black(frame, silhouette_img)
        # overlap countdown
        frame = overlay_countdown(frame, countdown)
        # display image
        cv2.imshow('frame :',frame)
        #save_frame(frame)
        # exit the capture loop?
        if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            c.release()
            cv2.destroyAllWindows()
            return

def execute_action_prompt(action, time):
    # determine N_frames
    N_frame = determine_N_frames(time)
    # load action image
    action_img = load_action_image(action)
    # for frame in  N_frames:
    for i in range(N_frame):
        # set countdown timer
        countdown = int(time * (1 - i/N_frame))
        # record webcam frame to set timing
        _, _ = WEBCAM.read()
        # fetch action_image thumbnail
        frame = load_action_image(action) # action_img
        # overlap countdown
        frame = overlay_countdown(frame, countdown)
        # overlay action heading
        # TODO: overlay action heading text.
        # display the image
        cv2.imshow('frame :',frame)
        # save_frame(frame)
        # exit the capture loop?
        if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            c.release()
            cv2.destroyAllWindows()
            return

def execute_action_record(action, time):
    # determine N_frames
    N_frame = determine_N_frames(time)
    # for frame in N_frames:
    for i in range(N_frame):
        # set countdown timer
        countdown = int(time * (1 - i/N_frame))
        # record webcam frame
        r, frame = WEBCAM.read()
        # save frame
        save_frame(frame) # TODO: replace with the over lap extra.
        # overlay countdown
        frame = overlay_countdown(frame, countdown)
        # overlay action heading
        # TODO: overlay action heading text.
        # display the image
        cv2.imshow('frame :',frame)
        # save_frame(frame)
        # exit the capture loop?
        if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            c.release()
            cv2.destroyAllWindows()
            return

def execute_done(action, time):
    # determine N_frames
    N_frame = determine_N_frames(time)
    # load done image
    done_img = load_action_image('done')
    # for frame in N_frames
    for i in range(N_frame):
        # set countdown timer
        countdown = int(time * (1 - i/N_frame))
        # record webcam frame to set timing
        _, _ = WEBCAM.read()
        # fetch action_image thumbnail
        frame = done_img
        # overlap countdown
        # frame = overlay_countdown(frame, countdown)
        # overlay action heading
        # TODO: overlay action heading text.
        # display the image
        cv2.imshow('frame :',frame)
        # save_frame(frame)
        # exit the capture loop?
        if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
        # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            c.release()
            cv2.destroyAllWindows()
            return
        # fetch done thumbnail
        # overlay user_name
        # display image


def determine_N_frames(timelength: float) -> int:
    # calculate the number of frames needed for collection
    # to reach the desired time period.
    N_frames = timelength * RECORDING_FPS
    return N_frames # TODO: change to the calculated value.

def overlay_countdown(img: np.ndarray, num: int) -> np.ndarray:
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (110,350)
    fontScale              = 10
    fontColor              = (200,200,200)
    lineType               = 20
    cv2.putText(
        img, str(num),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType
    )
    return img

def overlay_action_heading(img: np.ndarray, action: str) -> np.ndarray:
    # overlay the action type on the img
    # prefer not in the way
    # return the edited image
    return img # TODO: change to the edited image

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

def get_camera_resolution(cv2VideoCapture) -> tuple:
    r, frame = cv2VideoCapture.read()
    return np.shape(frame)[:2]

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
    image_path = os.path.join(base_path,'images',image_name+'.png') #TODO: differing save paths for user, data, action, batch id
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
    schedule_cap()