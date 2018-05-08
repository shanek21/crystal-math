import os
import cv2 as cv
import numpy as np


class Filterer():


  def __init__(self, filename, frame_dur=50, save=False):
    self.dirname = os.path.realpath(os.path.dirname(__file__))
    self.filename = filename
    self.frame_dur = frame_dur
    self.save = save

    cv.namedWindow('bgr_window')
    hsv_parameters = np.loadtxt(self.dirname + '/hsv_parameters.csv', delimiter=',', dtype=np.int_)
    self.hsv_lb, self.hsv_ub = hsv_parameters
    cv.createTrackbar('H lb', 'bgr_window', self.hsv_lb[0], 255, self.set_h_lb)
    cv.createTrackbar('H ub', 'bgr_window', self.hsv_ub[0], 255, self.set_h_ub)
    cv.createTrackbar('S lb', 'bgr_window', self.hsv_lb[1], 255, self.set_s_lb)
    cv.createTrackbar('S ub', 'bgr_window', self.hsv_ub[1], 255, self.set_s_ub)
    cv.createTrackbar('V lb', 'bgr_window', self.hsv_lb[2], 255, self.set_v_lb)
    cv.createTrackbar('V ub', 'bgr_window', self.hsv_ub[2], 255, self.set_v_ub)

  # HSV slider callbacks
  def set_h_lb(self, val): self.hsv_lb[0] = val
  def set_h_ub(self, val): self.hsv_ub[0] = val
  def set_s_lb(self, val): self.hsv_lb[1] = val
  def set_s_ub(self, val): self.hsv_ub[1] = val
  def set_v_lb(self, val): self.hsv_lb[2] = val
  def set_v_ub(self, val): self.hsv_ub[2] = val


  def run(self):
    cap = cv.VideoCapture(self.dirname + '/../media/raw/' + self.filename + '.mp4')
    frame_count = 0

    if self.save:
      fourcc = cv.VideoWriter_fourcc(*'XVID')
      out = cv.VideoWriter(self.dirname + '/../media/filtered/' + self.filename + '.avi', fourcc, 20.0, (650, 650))
    
    while(cap.isOpened()):
      ret, bgr_img = cap.read()
      if ret == True:
        frame_count += 1
        if frame_count == cap.get(7) - 10:
          frame = cap.get(7) - 50
          cap.set(1, frame)
          frame_count = frame

        hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
        mask_img = cv.inRange(hsv_img, self.hsv_lb, self.hsv_ub)
        contours = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        contour_img = np.zeros(bgr_img.shape[:2], dtype="uint8")
        cv.drawContours(contour_img, [max(contours, key=cv.contourArea)], 0, (255, 255, 255))

        if self.save:
          out.write(cv.cvtColor(mask_img, cv.COLOR_GRAY2RGB))
        cv.imshow('bgr_window', np.hstack((bgr_img, cv.cvtColor(contour_img, cv.COLOR_GRAY2RGB))))

        k = cv.waitKey(self.frame_dur) & 0xFF
        if k == ord('q'):  # quit
          break
        elif k == ord('s'):  # save
          np.savetxt(self.dirname + '/hsv_parameters.csv', np.array([self.hsv_lb, self.hsv_ub]),
              fmt='%03d', delimiter=',')
          print('HSV parameters saved.')
      else:
        break

    cap.release()
    if self.save:
      out.release()
    cv.destroyAllWindows()


  def run_subtractive(self):
    cap = cv.VideoCapture(self.dirname + '/../media/raw/' + self.filename + '.mp4')
    frame_count = 0
    start = 7
    first_frame = None

    if self.save:
      fourcc = cv.VideoWriter_fourcc(*'XVID')
      out = cv.VideoWriter(self.dirname + '/../media/filtered/' + self.filename + '_sub.avi', fourcc, 20.0, (650, 650))

    while(cap.isOpened()):
      frame_count += 1
      ret, bgr_img = cap.read()
      if ret == True:
        # bgr_img = cv.resize(bgr_img, (0,0), fx=0.5, fy=0.5)
        if frame_count < start:
          continue
        elif frame_count == start:
          first_frame = bgr_img
          hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
          mask_img = cv.inRange(hsv_img, self.hsv_lb, self.hsv_ub)
        else:
          sub_img = cv.subtract(bgr_img, first_frame)
          hsv_img = cv.cvtColor(sub_img, cv.COLOR_BGR2HSV)
          mask_img = cv.inRange(hsv_img, self.hsv_lb, self.hsv_ub)
          kernel = np.ones((1,1),np.uint8)
          mask_img = cv.erode(mask_img,kernel,iterations = 1)
        if self.save:
          out.write(cv.cvtColor(mask_img, cv.COLOR_GRAY2RGB))
        cv.imshow('bgr_window', np.hstack((first_frame, bgr_img, cv.cvtColor(mask_img, cv.COLOR_GRAY2RGB))))
        k = cv.waitKey(self.frame_dur) & 0xFF
        if k == ord('q'):  # quit
          break
        elif k == ord('s'):  # save
          np.savetxt(self.dirname + '/hsv_parameters.csv', np.array([self.hsv_lb, self.hsv_ub]),
              fmt='%03d', delimiter=',')
          print('HSV parameters saved.')
      else:
        # cap.set(1, start)
        break

    cap.release()
    if self.save:
      out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
  filename = '12_v'
  f = Filterer(filename, 1, save=True)
  f.run_subtractive()
