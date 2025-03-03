# Datalabeling
This repository is used to facilitate data labeling in the context of **automated aerial census of large herbivores**. The project aims to facilitate the labeling of aerial images by integrating a YOLO-based object detector that learns from the annotations. The active learning pipeline is currently semi-automated. It uses *Label studio* as the labeling platform and *Ultralytics* as the machine learning training library.

# Set up a workspace 
- Open a terminal
- Create a folder ``` mkidr workspace```
- Move to folder ```cd workspace```
- Clone repository into the workspace ```git clone https://github.com/FadelMamar/datalabeling.git```

# Creating virtual environments
### Create virtual environment for Label studio UI
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar of your computer and launch it.
- Change directory ```cd ./sourcecode```
- Create a conda virtual environment with ```conda env create -f environment_labelstudio.yml```
- Activate virtual environment:```conda activate label-studio``` 
- Run the command ```pip install -e .```
- Launch label studio: type in ```label-studio``` then press 'Enter'`
- Close application: log out of the application then type ```conda deactivate``` then press 'Enter'
### Create virtual environment for ML backend
- Change directory ```cd ../my_ml_backend```
- Create a conda virtual environment with ```conda env create -f environment_mlbackend.yml```
- Activate virtual environment:```conda activate label-backend``` 
- Change directory ```cd ../sourcecode```
- Run the command ```pip install -e .```


# [Optional] Installing Label-studio for windows
- **Install Anaconda** Follow instructions at https://www.anaconda.com/download
(Instructions are also available here https://labelstud.io/guide/install.html#Install-with-Anaconda)

# [Optional] Installing Label-studio for Mac
(Instructions are available here: https://labelstud.io/guide/install.html#Install-using-Homebrew)
- **Install homebrew** -> Follow instructions here https://brew.sh/
- Open terminal 
- Type in ```brew tap humansignal/tap``` and press "Enter"
- Type in ```brew install humansignal/tap/label-studio``` and press "Enter"
- Launch Label studio by typing ```label-studio``` and pressing "Enter".

# [Optional] Installing Label-studio using Docker (Mac and windows)
(instructions are available here https://labelstud.io/guide/install.html#Install-with-Docker)
- Install Docker ->  https://www.docker.com/ 
- open terminal in your workspace and run ```docker run -it -p 8080:8080 -v ./labeleddata:/label-studio/data heartexlabs/label-studio:latest```

# Launch Label-studio when it is already installed - Windows
- To launch label studio, run the file at ```datalabeling\helper-scripts\run-labelstudio-windows.bat```
- To launch the ML backend, run the file at ```datalabeling\helper-scripts\run-ml-backend-windows.bat```
- Label studio is accessible from the browser at ```http://localhost:8080```
- To close application: close the window

# Launch Label-studio when it is already installed - Linux
- To launch label studio, run the file at ```datalabeling\helper-scripts\run-labelstudio-linux.sh```
- To launch the ML backend, run the file at ```datalabeling\helper-scripts\run-ml-backend-linux.sh```
- To close application: close the window



# Create a project in Label studio
https://labelstud.io/guide/setup_project.html#Create-a-project
https://labelstud.io/guide/setup_project.html#Set-up-annotation-settings-for-your-project

# Import data to Label studio
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

## Annotation template for segmentation
<View>
<PolygonLabels name="segmentation" toName="image"
                 strokeWidth="3" pointSize="small"
                 opacity="0.9">
    <Label value="camp" background="red"/>
    <Label value="notcamp" background="blue"/>
 </PolygonLabels>
 </View>
 
 ## Annotation template for classification
 <View>
 <Choices name="choice" toName="image">
   <Choice value="namp"/>
   <Choice value="notCamp" />
 </Choices>
  </View>
