# Google Summer of Code 2021 & 2022
This GitHub repo has been developed from scratch by Özgür Kara under GSoC 2021 and has been extended by Mehul Sinha under GSoC 2022. It consists of the main program and it's [detailed documentation](https://github.com/NISYSLAB/VisualRLComposer/blob/main/documentation.pdf). If there is any issue, you can contact us via ozgurrkara99@gmail.com or mehulsinha73@gmail.com 

# VisualRLComposer
The project aims to develop a GUI for facilitating the experimentation of Reinforcement Learning for the users. Particularly, researchers are able to test and implement their ideas and algorithms of reinforcement learning with the GUI easily even though they are not proficient in coding.

# Features
* PyQt5 based open source Graphical User Interface for visually testing the RL agents
* Dragging and dropping the components allow users to create flows easily
* Flows can be saved (in json format), loaded and new graphs can be opened using toolbar options
* Flows can also be created in DHGWorkflow, saved in GraphML format and loaded in VisualRLComposer as a new graph
* Users can create multiple Workflow graphs by opening a new scene tab
* During testing, the relevant values such as rewards, states and actions are updated in a real-time manner
* There are seven built-in RL environments and reward functions from OpenAI Gym and six RL agents that are imported from stable-baselines3 library
* Program allows users to both perform training and testing. Also, users can save their trained models as well as load their pretrained models according to their preferences
* Users can design custom policies for their RL models using the custom policy network designer 
* The program allows users to train or test models with multi-environment experimentation in parallel 
* The program allows user to keep track of model training via Tensorboard
* Users are able to integrate their custom environments and reward functions to the program by following the procedure in the documentation
* Detailed documentation and demo videos are provided in the GitHub page

# Setup and Installation
In order to install the program to Windows, run the following codes:
```
git clone https://github.com/NISYSLAB/VisualRLComposer
python -m venv env
source env/bin/activate
pip install -r /path/to/requirements.txt
```
After the installation is done, you can run the program by running the following command:
```
python main.py
```

# Demo Videos For Training and Testing

![](https://github.com/NISYSLAB/VisualRLComposer/blob/main/assets/demo1.gif)
![](https://github.com/NISYSLAB/VisualRLComposer/blob/main/assets/demo2.gif)
