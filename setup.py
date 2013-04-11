from setuptools import setup

setup(
    name='piboso',
    version='1',
    author='Marco Lui',
    author_email='mhlui@unimelb.edu.au',
    packages=['piboso'],
    package_dir={'piboso':'piboso'},
    package_data={'piboso':['models/*','data/stopword']},
    url='http://pypi.python.org/pypi/piboso/',
    license='LICENSE.txt',
    description='Sentence tagger for biomedical abstracts.',
    long_description=open('README.txt').read(),
    install_requires=[
        "hydrat >= 0.9.5",
        "nltk >= 2.0",
    ],
    entry_points={
        'console_scripts': [
            'piboso = piboso.cli:main',
        ],
    },
)
