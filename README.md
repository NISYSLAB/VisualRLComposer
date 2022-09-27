# NeuroWeaver Integration with VisualRLComposer
This branch of VisualRLComposer has been developed to provide a frontend GUI for the NeuroWeaver project to make and run CnF programs.

# Setup and Installation
In order to install the program, run the following codes:
```
git clone https://github.com/NISYSLAB/VisualRLComposer
cd VisualRLComposer
git checkout neuroweaver
python -m venv env
source env/bin/activate
pip install -r /path/to/requirements.txt
```
After the installation is done, run the following commands to copy rlcomposer to neuroweaver
```
cp /path_to_VisualRLComposer/main.py /path_to_neuroweaver/ppo/
cp -r /path_to_VisualRLComposer/rlcomposer/ /path_to_neuroweaver/
```
To run the RLComposer GUI,
```
cd /path_to_neuroweaver/
source /path_to_VisualRLComposer/env/bin/activate
python ppo/main.py
```

## Debug Notes

```
  File "/home/mehul/Desktop/test/neuroweaver/ppo/ppo_components.py", line 20, in <module>
    from ppo import PPO
ImportError: cannot import name 'PPO' from 'ppo' (/home/mehul/Desktop/test/neuroweaver/ppo/__init__.py)

- Change 'from ppo import PPO' to 'from ppo.ppo import PPO'
```

```
  File "/home/mehul/Desktop/test/neuroweaver/ppo/main.py", line 10, in <module>
    from rlcomposer.main_window import RLMainWindow
ModuleNotFoundError: No module named 'rlcomposer'

- manually add neuroweaver path in main.py
```


