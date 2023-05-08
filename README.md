# PLEASE READ FOR FINAL PROJECT - CSCI3302_LabTeam

WATCH ME FOR FINAL PROJECT VIDEO

[Click for youtube link](https://youtu.be/K8VR4sO6eGA)

(Sorry for going over the desired time limit. We really tried to condense it but were not sure what to cut out without losing value in the video. We hope you still enjoy the video! Also, the video took really long to render due to our old computers, so we had to wait a long time (over 2 hours) for the video to finish processing. We apologise for the slightly late upload.)

ADDITIONALLY... We accidentally hot-keyed both the 'J' and 'K' keys to two different functions in the robotic arm, causing the robotic arm not to behave properly. We fixed this when we were filming the video however we forgot to push this update to our Github before midnight. 
This is happening on lines 1123 and 1190 for the 'J' key and 1140 and 1201 for the 'K' key.  
If you would like to fix this problem so that you can properly test the controller in your WeBots, please change line 1123 to: 

`elif key == ord('A')`

and line 1140 to 

`elif key == ord('D')`

We apologise in advance for this inconvenience however we would appreciate if you implement this change to properly test our code.
