"""
Trained model loading
"""

import bz2, os
from cPickle import load

def load_model(path):
  with bz2.BZ2File(path) as model_f:
    return load(model_f)

def load_default_model():
  path = os.path.join(os.path.dirname(__file__), 'models', 'default')
  return load_model(path)
