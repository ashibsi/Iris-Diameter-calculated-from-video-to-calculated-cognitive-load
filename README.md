# Iris-Diameter-calculated-from-video-to-calculated-cognitive-load


Participant must be seated comfortably in front of a camera, and their face was recorded while they answered a series of mathematics questions with varying levels of difficulty to induce different levels of cognitive load. No real-time tracking software was used; instead, focused on video-based analysis after data collection.

The recorded videos is processed frame by frame using image processing techniques to detect the pupil and calculate its diameter over time. The average pupil diameter for each difficulty level was then computed and compared. By analyzing how pupil size changed with task complexity, the experiment aimed to verify whether more difficult mathematical problems produced greater pupil dilation, thereby supporting the hypothesis proposed by Cognitive Load Theory.


To run the code use following command :
```python
python iris_diameter_windowed_plot.py --video sample.mp4 --window 3.0 --display
```
Make sure the video is in the same folder as the code and name the code "iris_diameter_windowed_plot.py"for above command to work.And also make sure that you are in same folder as code in terminal before running the command. To do so use -cd commmand.


The name of the video should be replace with Sample.mp4 and the duration at whick the reading is taken b echanged by changing the  3.0 to other values.

The video should be stable with constant gaze to camera for best result. 

After processing a diameter(in pixel) vs time graph will be plotted.
