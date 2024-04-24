# datalabeling
Repository for data labeling

# Set up a workspace 
- Open terminal
- Create a folder ``` mkidr datalabeling```
- Move to folder ```cd datalabeling```
- Create a subfolder ```mkdir data```
- Copy the images to be labeled inside './datalabeling/data/' 

# Install Label-studio for windows
- **Install Anaconda** Follow instructions at https://www.anaconda.com/download
(Instructions are also available here https://labelstud.io/guide/install.html#Install-with-Anaconda)
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar of your computer and launch it.
- Create a conda virtual environment: type in ```conda create -n label-studio``` then press 'Enter'
- Activate virtual environment: type in ```conda activate label-studio``` then press 'Enter'
- Install label-studio: type in ```pip install label-studio``` then press 'Enter'
- Launch label studio: type in ```label-studio``` then press 'Enter'`
- Close application: log out of the application then type ```conda deactivate``` then press 'Enter'

# Launch Label-studio when it is already installed - Windows
- To launch label studio, run the file at ```C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\run-labelstudio.bat```
- To launch the ML backend, run the file at ```C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\run-ml-backend.bat```
- Label studio is accessible from the browser at ```http://localhost:8080```
- To close application: close the window

# Launch Label-studio when it is already installed - Linux
- To launch label studio, run the file at ```C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\run-labelstudio.sh```
- To launch the ML backend, run the file at ```C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\run-ml-backend.sh```
- To close application: close the window

# Instal Label-studio for Mac
(Instructions are available here: https://labelstud.io/guide/install.html#Install-using-Homebrew)
- **Install homebrew** -> Follow instructions here https://brew.sh/
- Open terminal 
- Type in ```brew tap humansignal/tap``` and press "Enter"
- Type in ```brew install humansignal/tap/label-studio``` and press "Enter"
- Launch Label studio by typing ```label-studio``` and pressing "Enter".

# Install Label-studio using Docker (Mac and windows)
(instructions are available here https://labelstud.io/guide/install.html#Install-with-Docker)
- Install Docker ->  https://www.docker.com/ 
- open terminal in your workspace and run ```docker run -it -p 8080:8080 -v ./labeleddata:/label-studio/data heartexlabs/label-studio:latest```

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