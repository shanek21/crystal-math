import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter():

  def __init__(self):
    self.dirname = os.path.realpath(os.path.dirname(__file__))
    sns.set()
    
  def plot(self, filenames):
    areas = []
    for i, filename in enumerate(filenames):
      area = []
      cap = cv.VideoCapture(self.dirname + '/../media/filtered/' + filename + '_sub.avi')
      num_frames = 115
      start_frame = 9
      stop_frame = num_frames - 10
      frame_count = 0

      while(cap.isOpened()):
        ret, bgr_img = cap.read()
        if ret == True:
          frame_count += 1
          if frame_count < start_frame or frame_count >= stop_frame:
            continue

          mask_img = cv.inRange(bgr_img, np.array([200, 200, 200]), np.array([255, 255, 255]))
          area.append(np.count_nonzero(mask_img))
          # cv.imshow('bgr_window', np.hstack((bgr_img, cv.cvtColor(mask_img, cv.COLOR_GRAY2RGB))))

          k = cv.waitKey(1) & 0xFF
          if k == ord('q'):  # quit
            break
        else:
          break

      frame_interval = 8.6  # seconds
      plt.plot(np.array(range(len(area))) * frame_interval, area)
      areas.append(area)

      cap.release()
      # cv.destroyAllWindows()

    areas = np.array(areas)
    plt.title('Dendrite Area Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Area (pixels)')
    plt.legend(filenames)
    plt.show()


if __name__ == '__main__':
  p = Plotter()
  p.plot(['06_v', '07_v', '08_v', '09_v', '10_v', '12_v'])
