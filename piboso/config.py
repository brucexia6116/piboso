"""
This module controls the configuration of the 
piboso tagger
"""

import os
import tempfile
import pkgutil
import atexit
from hydrat import config as hydrat_config
from ConfigParser import SafeConfigParser

DEFAULT_CONFIG_FILE='.pibosorc'

def default_config():
  config = SafeConfigParser()
  config.add_section('paths')
  config.set('paths','treetagger','/path/to/treetagger')
  return config

def load_config(path=None):
  """
  Load configuration from a given path. If none, search in
  default paths
  """
  if path is None:
    if os.path.exists(DEFAULT_CONFIG_FILE):
      path = DEFAULT_CONFIG_FILE
    elif os.path.exists('~/'+DEFAULT_CONFIG_FILE):
      path = '~/'+DEFAULT_CONFIG_FILE
    else:
      raise ValueError("configuration not found")

  config = default_config()
  config.read(path)

  wl_fd, wl_path = tempfile.mkstemp()
  os.write(wl_fd, pkgutil.get_data('piboso', 'data/stopword'))
  os.close(wl_fd)

  def cleanup():
    os.unlink(wl_path)
  atexit.register(cleanup)

  # Process configured options
  hydrat_config.set('parameters','parallel_classify', 'False')
  hydrat_config.set('parameters','parallel_tokenize', 'False')
  hydrat_config.set('debug','allow_empty_instance', 'True')
  hydrat_config.set('debug','pdb_on_invalid_fm', 'False')
  hydrat_config.set('tools','treetagger', config.get('paths','treetagger'))
  hydrat_config.set('paths','stopwords', wl_path)


def write_blank_config(path):
  """
  Write a blank configuration to a given file.
  """
  with open(path, 'w') as out_f:
    default_config().write(out_f)
