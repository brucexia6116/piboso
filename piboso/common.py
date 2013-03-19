from datetime import timedelta
from timeit import default_timer

class Timer(object):
  def __init__(self):
    self.timer = default_timer
    self.start = None
    self.end = None

  def __enter__(self):
    self.start = self.timer()
    self.end = None
    return self

  def __exit__(self, *args):
    self.end = self.timer()

  @property
  def elapsed(self):
    now = self.timer()
    if self.end is not None:
      self.end - self.start
    else:
      return now - self.start

  def rate(self, count):
    now = self.timer()
    if self.start is None:
      raise ValueError("Not yet started")

    return count / (now - self.start)

  def ETA(self, count, target):
    """
    Linearly estimate the ETA to reach target based on the current rate.
    """
    rate = self.rate(count)
    time_left = timedelta(seconds=int((target-count) / rate))
    return time_left 

import os, errno
def makedir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
