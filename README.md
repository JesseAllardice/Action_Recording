# Action_Recording

This project enables the user to easily collect a automatically annotated dataset of human exercise with video. The collected datasets can then be used to train a deep learning model for exercise recognition and repetition rate estimation with the project [Action-Recognition](https://github.com/JesseAllardice/Action-Recognition).
# scheduleCap.py
scheduleCap.py collects a labelled dataset of exercise movements in the form of ordered webcam images.

# Before running scheduleCap.py
- check that the testing is off (TESTING = False)
- that the user id (USER_ID) is correct

# Workout schedule
information is stored and retreved from the file:
Action_Recording_Schedule.json

This includes information such as
action_types, a list of possible actions
state_types, a list of possible states in the schedule.
schedule, list of entries in the form {state: time period}