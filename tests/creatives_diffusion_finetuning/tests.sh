#!/bin/bash
python nb2py.py notebooks/demo_walkthrough.ipynb README.py  # convert notebook to py script
python README.py  # run the converted python script
rm README.py  # remove the generated script
python train.py --debug_steps 30
