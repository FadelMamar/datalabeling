from setuptools import setup

def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()
    
setup(
name='datalabeling',
version='0.0.1',
description='datalabeling software based on label-studio',
author='fadel seydou',
author_email='fadel.seydou@gmail.com',
# packages=['datalabeling.cli','datalabeling.arguments',
#           'datalabeling.train','datalabeling.preprocessing'],
install_requires=load_requirements(),
)