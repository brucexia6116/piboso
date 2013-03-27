=============
PIBOSO Tagger
=============

This module contains a fully-standalone implementation of the PIBOSO
tagger that won the ALTA2012 Shared task [1]. The features and algorithms
used are described in [2].

Installing
----------
The tagger (including a pre-trained model) is packaged as a Python module
and distributed via pypi. Installing it should be as simple as

  pip install piboso

Dependencies
------------
hydrat [3] - automatically installed by pip
TreeTagger [4] - must be manually installed

Configuration
-------------
The path to the folder in which treetagger is located must be specified
in configuration file. When invoked, `piboso_tag` will attempt to locate
a configuration file at ~/.pibosorc and ./.pibosorc. If neither exists,
it will generate a blank configuration file at ./.pibosorc. The path
to treetagger should be set in this configuration file.

An alternative location for reading the configuration file can be specified
with the `-c` command-line option.

Using the tagger
----------------
The tagger can be invoked with the script `piboso_tag`, that is automatically
installed when the package is installed with pip. The simplest invocation is

  piboso_tag -o <OUTPUT_PATH> <FILE TO TAG> <FILE TO TAG> ...

If no files are specified on the command line, `piboso_tag` will read STDIN
and interpret each line as a path to a file to be tagged. More detailed 
information about invoking `piboso_tag` can be obtained by invoking

  piboso_tag --help

Files are assumed to be sentence tokenized and presented in a sentence-per-line
format. The output produced by `piboso-tag` is in a CSV format, for example:

subsample/1454068-1,background
subsample/1454068-2,background
subsample/1454068-3,outcome
subsample/1454088-1,background
subsample/1454088-2,background
subsample/1454088-3,background
subsample/1454088-4,background

The first item in each record is the path of the file and the sentence number
separated by a dash. Sentences are enumerated from 1. The second item is the 
label assigned to the sentence.

Contact
-------

Marco Lui <mhlui@unimelb.edu.au>


[1] http://alta.asn.au/events/sharedtask2012/ 
[2] http://aclweb.org/anthology-new/U/U12/U12-1019.pdf 
[3] http://hydrat.googlecode.com
[4] http://www.ims.uni-stuttgart.de/projekte/corplex/TreeTagger/ 
