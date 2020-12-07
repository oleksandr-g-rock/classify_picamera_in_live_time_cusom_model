# Importing the modules
import os
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import numpy as np
import tensorflow as tf
import tensorflow.keras
from PIL import Image, ImageOps
import argparse
import io
import picamera
    
#set main function    
def main():
    
  print("#set classes names")
  classes_names = ['animals', 'other', 'person'] #you can change classes

  print("#load model")
  model = tensorflow.keras.models.load_model('animall_person_other_v2_fine_tuned.h5') #you can change name and path for model

  #start capturing the image from the Picamera
  with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        
        #you can change image size from 299, 299 to any size
        img = Image.open(stream).convert('RGB').resize((299, 299),
                                                         Image.ANTIALIAS)
        #start time
        start_time = time.time()

        #####predict class from video stream block######
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        classes = np.argmax(prediction, axis = 1)
        #####predict class from video stream block######
        
        elapsed_ms = (time.time() - start_time) * 1000
        stream.seek(0)
        stream.truncate()
        
        #block for print result
        print("elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')
        #block for print result
    finally:
      camera.stop_preview()
if __name__ == '__main__':
  main()
