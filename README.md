# Webcam_Recording

# scheduleCap.py
scheduleCap.py collects a labelled dataset of exercise movements in the form of ordered webcam images.

# Before running scheduleCap.py
- check that the testing short-circuits are off (TESTING = False)
- that the user id (USER_ID) is correct

# Workout schedule
information is stored and retreved from the file:
Action_Recording_Schedule.json

This includes information such as
action_types, a list of possible actions
state_types, a list of possible states in the schedule.
schedule, list of entries in the form {state: time period}