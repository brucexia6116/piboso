from setuptools import setup

setup(
    name='piboso',
    version='1',
    author='Marco Lui',
    author_email='mhlui@unimelb.edu.au',
    packages=['piboso'],
    package_data={'piboso':['piboso/models/*']},
    scripts=['bin/piboso_tag'],
    url='http://pypi.python.org/pypi/piboso/',
    license='LICENSE.txt',
    description='Sentence tagger for biomedical abstracts.',
    long_description=open('README.txt').read(),
    install_requires=[
        "hydrat >= 0.9.5",
    ],
)
