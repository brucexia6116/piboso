"""
Trained model loading
"""

import bz2, os
import pkgutil
from cPickle import load, loads

def load_model(path):
  with bz2.BZ2File(path) as model_f:
    return load(model_f)

def load_default_model():
  data = pkgutil.get_data('piboso', 'models/default')
  return loads(bz2.decompress(data))
