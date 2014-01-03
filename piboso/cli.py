#!/usr/bin/env python
"""
Off-the-shelf PIBOSO tagger

Marco Lui, March 2013
"""

import argparse, sys, os
import csv

from piboso.common import Timer
from piboso.model import PIBOSOTagger
from piboso.config import load_config, write_blank_config, DEFAULT_CONFIG_FILE


import logging

logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("abstracts", metavar="FILE", help="do PIBOSO tagging for FILE (can specify multiple)", nargs='*')
  parser.add_argument("--dist","-d", help="output distribution over labels")
  parser.add_argument("--model","-m", help="read model from")
  parser.add_argument("--config","-c", help="read configuration from")
  parser.add_argument("--output","-o", type=argparse.FileType('w'), metavar="FILE", default=sys.stdout, help="output to FILE (default stdout)")
  parser.add_argument("--temp","-t", help="store temporary files in TEMP")
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG)

  try:
    load_config(args.config)
  except ValueError:
    write_blank_config(DEFAULT_CONFIG_FILE)
    parser.error('no configuration found. blank config written to: {0}'.format(DEFAULT_CONFIG_FILE))

  if len(args.abstracts) > 0:
    paths = args.abstracts
  else:
    paths = filter(bool,map(str.strip, sys.stdin))

  abstracts = {p: open(p).readlines() for p in paths}
  logging.info("PIBOSO tagging for {0} files".format(len(abstracts)))

  with Timer() as prog_timer:
    tagger = PIBOSOTagger(path=args.model, tempdir=args.temp)
    output = tagger.classify_batch(abstracts)
    writer = csv.writer(args.output)

    if args.dist:
      output.write_dist(writer)
    else:
      output.write_pred(writer)

    logger.info("wrote output to: {}".format(args.output.name))
    logger.info("completed in {0:.2f}s ({1:.2f} abs/s)".format(prog_timer.elapsed, prog_timer.rate(len(abstracts))))

if __name__ == "__main__":
  main()
