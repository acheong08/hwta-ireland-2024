import gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

from evaluation import evaluation_function

#env.reset() – To reset the entire environment and obtain the initial values of observation.
#env.render() – Rendering the environment for displaying the visualization of the working setup.
#env.step() – To proceed with an action on the selected environment.
#env.close() – Close the particular render frame of the environment.