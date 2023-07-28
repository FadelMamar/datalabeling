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
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar for your computer and launch it.
- Create a virtual environment: type in ```python -m venv label-studio``` then press 'Enter'
- Activate virtual environment: type in ```label-studio\Scripts\activate.bat``` then press 'Enter'
- Install label-studio: type in ```pip install label-studio``` then press 'Enter'
- Launch label studio: type in ```label-studio``` then press 'Enter'
- Close application: log out of the application then type ```deactivate``` then press 'Enter'
## Launch Label-studio when it is already installed
- Open Anaconda prompt. Type 'anaconda prompt' in the search bar for your computer and launch it.
- Move to your workspace
- Activate virtual environment: type in ```label-studio\Scripts\activate.bat``` then press 'Enter'
- Launch label studio: type in ```label-studio start``` then press 'Enter'
- To close application: log out of the application, press ```ctrl+c``` inside the terminal, then type in ```deactivate``` and press 'Enter'

# Instal Label-studio for Mac
(Instructions are available here: https://labelstud.io/guide/install.html#Install-using-Homebrew)
- **Install homebrew** -> Follow instructions here https://brew.sh/
- Open terminal 
- Type in ```brew tap heartexlabs/tap``` and press "Enter"
- Type in ```brew install heartexlabs/tap/label-studio``` and press "Enter"
- Launch Label studio by typing ```label-studio``` and pressing "Enter".


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