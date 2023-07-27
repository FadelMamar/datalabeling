# datalabeling
Repository for data labeling

# Installation for Windows

## Install anaconda
Follow instructions at https://www.anaconda.com/download

## Set upn directory 
- Open terminal
- Create a folder ``` mkidr datalabeling```
- Move to folder ```cd datalabeling```
- Create a subfolder ```mkdir data```
- Copy the images to be labeled inside './datalabeling/data/' 

## Install Label-studio
(Instructions are also available her https://labelstud.io/guide/install.html#Install-with-Anaconda)
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar for your computer and launch it.
- Create a virtual environment: type in ```python -m venv label-studio``` then press 'Enter'
- Activate virtual environment: type in ```label-studio\Scripts\activate.bat``` then press 'Enter'
- Install label-studio: type in ```pip install label-studio``` then press 'Enter'
- Launch label studio: type in ```label-studio``` then press 'Enter'
- Close application: log out of the application then type ```deactivate``` then press 'Enter'

## Launch Label-studio when it is already installed
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar for your computer and launch it.
- Activate virtual environment: type in ```conda activate label-studio``` then press 'Enter'
- Launch label studio: type in ```label-studio start``` then press 'Enter'
- To close application: log out of the application, press ```ctrl+c``` inside the terminal, then type in ```deactivate``` and press 'Enter'

## Create a project in Label studio
https://labelstud.io/guide/setup_project.html#Create-a-project

## Import data to Label studio
https://labelstud.io/guide/tasks.html

## Annotation template for bounding boxes
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="specie1" background="green"/>
    <Label value="specie2" background="blue"/>
  </RectangleLabels>
</View>


## Annotation template for key points
<View>
  <KeyPointLabels name="kp-1" toName="img-1">
    <Label value="specie1" background="red" />
    <Label value="specie2" background="green" />
  </KeyPointLabels>
  <Image name="img-1" value="$img" />
</View>