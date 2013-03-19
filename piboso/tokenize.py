"""
Tokenizer for feature sets used in PIBOSO sentence tagging.

Marco Lui, March 2013
"""

import os, sys, argparse
import tarfile
from contextlib import closing
from itertools import islice, groupby

import hydrat.common.extractors as ext
from hydrat.store import Store
from hydrat.proxy import DataProxy

import piboso.features as features
from piboso.common import Timer, makedir
from piboso.corpora import NewDocuments

import multiprocessing as mp


def tokenize(ds, features, store_path, fallback_path):
  class_space = 'ebmcat'
  fallback = Store(fallback_path)

  #print >>sys.stderr,  "=== opening store at {0} ===".format(store_path)
  with closing(Store(store_path, 'a', fallback=fallback, recursive_close=False)) as store:

    print >>sys.stderr,  "=== inducing features ({0}) ===".format(features)
    # Induce all the features for the new test data
    proxy = DataProxy(ds, store=store)
    proxy.inducer.process(proxy.dataset, 
      fms=features,
      sqs=['abstract',],
    )

# This is split in two as some of the features were not declared
# in the dataset layer, and so need to be induced using an external
# tokenize call.
def tokenize_ext(ds, store_path, fallback_path):
  class_space = 'ebmcat'
  fallback = Store(fallback_path)

  print >>sys.stderr,  "=== tokenize_ext for {0} ===".format(store_path)
  with closing(Store(store_path, 'a', fallback=fallback, recursive_close=False)) as store:
    proxy = DataProxy(ds, store=store)

    proxy.tokenstream_name = 'treetaggerlemmapos'
    proxy.tokenize(ext.bigram)

    proxy.tokenstream_name = 'treetaggerpos'
    proxy.tokenize(ext.bigram)
    proxy.tokenize(ext.trigram)
    
    #TODO: Copy in the feature spaces?


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-n','--number',type=int,help='number of files to process')
  parser.add_argument('-p','--parts',type=int,help='number of parts to process')
  parser.add_argument('--feats', default='all', help="feature group to process")
  parser.add_argument("data", metavar="PATH", help="read data from PATH (tgz format, filenames as docid)")
  parser.add_argument('feat_store', help='existing hydrat Store to read feature spaces from')
  parser.add_argument("outdir", help="produce output in DIR", metavar="DIR")
  args = parser.parse_args()

  try:
    features = features.feature_sets[args.feats]
  except KeyError:
    parser.error("unknown feature group: {0}".format(args.feats))

  makedir(args.outdir)

  print >>sys.stderr,  "=== producing output in {0} ===".format(args.outdir)
  with tarfile.open(args.data) as archive, Timer() as t:
    # The records in the tarfile are ordered by part. We only take the file records, 
    # extract the part ID from the filename and use that as a chunk.
    archive_iter = groupby((r for r in archive if r.isfile()), lambda r: r.name.split('/')[1])

    chunks_processed = 0
    files_processed = 0
    for chunk_id, chunk_tarinfos in archive_iter:
      if args.parts and chunks_processed >= args.parts:
        break

      if args.number:
        chunk_tarinfos = islice(chunk_tarinfos, args.number)

      chunk = [archive.extractfile(i) for i in chunk_tarinfos]
      print >>sys.stderr, "==== processing Part {0} ({1} files) ====".format(chunk_id, len(chunk))

      store_path = os.path.join(args.outdir, '{0}.features.{1}.h5'.format(chunk_id, args.feats)) 
      #if os.path.exists(store_path):
      #  print >>sys.stderr, "==== {0} exists, skipping ====".format(store_path)
      #  continue

      # We do the tokenization in a subprocess to avoid Python holding on to memory.
      ds = NewDocuments(chunk)
      for feature in features:
        p = mp.Process(target=tokenize, args=(ds, [feature], store_path, args.feat_store))
        p.start()
        p.join()
        p.terminate()

      p = mp.Process(target=tokenize_ext, args=(ds, store_path, args.feat_store))
      p.start()
      p.join()
      p.terminate()


      files_processed += len(chunk)
      chunks_processed += 1

      print >>sys.stderr,  "**** processed {0} files in {1}s ({2} f/s) ****".format(files_processed, t.elapsed, t.rate(files_processed))
      print >>sys.stderr, open('/proc/self/statm').read() #THIS KEEPS CLIMBING!!
