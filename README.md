# DRLND-continous-learning

## Table of Contents

1. [Summary](#summary)
2. [Environment](#Environment)
3. [Installation](#installation) 
4. [Getting Started](#GettingStarted)
5. [File Descriptions](#files)
6. [Experiments](#experiments)

##  Summary <a name="summary"></a>

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Distributed Training

For this project, we have provided with two separate versions of the Unity environment:

* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment.


## Environment <a name="Environment"></a>

* *Set-up:* Double-jointed arm which can move to target locations.
* *Goal:* Each agent must move its hand to the goal location, and keep it there.
* *Agents:* The environment contains 20 agents linked to a single Brain.
* *Agent Reward Function (independent):* +0.1 for each timestep agent's hand is in goal location.
* *Brains:*  One Brain with the following observation/action space.

Vector Observation space:* 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.  
        
Vector Action space:* (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. 
           
Visual Observations:* None.

* Reset Parameters: Two, corresponding to goal size, and goal movement speed.

#### Solving the Environment

* Environment Solving Criteria: The target for the agent is to solve the environment by achieving a score of +30 averaged across all 20 agents for 100 consecutive episodes.


## Getting Started <a name="Getting Started"></a>

#### *Step 1:* Clone the repo:
`git clone https://github.com/debjani-bhowmick/DRLND-continous-learning.git` 

#### *Step 2:*Change directory into the repo:
`cd continuous-control`

#### *Step 3:*  Activate the Environment

For details related to setting up the Python environment for this project, please follow the instructions provided in the DRLND GitHub repository[https://github.com/udacity/deep-reinforcement-learning]. These instructions can be found in README.md at the root of the repository. By following these instructions, user will be able to install the required PyTorch library, the ML-Agents toolkit, and a few more Python packages required for this project.

(For Windows users) The ML-Agents toolkit supports Windows 10 currently. In general, ML-Agents toolkit could possibly be used for other versions, however, it has not been tested officially, and we recommend choosing Windows 10. Also, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Further, the specific files to look into in the repository is python/setup.py and requiremnets.txt. The readme provides thorough details related to setting up the environment.




#### *Step 4:* Download the Unity Environment

For this project, you will not need to install Unity - this is because environment has buit for you, and you can download it from one of the links below. You need only select the environment that matches your operating systM

Download the environment from one of the links below. You need only select the environment that matches your operating system:

* Linux: (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip]
* Mac OSX: (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip]
* Windows (32-bit): (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip]
* Windows (64-bit): (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip]

(For Windows users) Check out this link(https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

#### *Step 5:*  Unzip (or decompress) the downloaded file:

* Linux:
`unzip Reacher_Linux.zip`
* Mac OSX:
`unzip -a Reacher.app.zip`
* Windows (32-bit): [PowerShell]
`Expand-Archive -Path Reacher_Windows_x86.zip -DestinationPath .
*Windows (64-bit): [PowerShell]
`Expand-Archive -Path Reacher_Windows_x86_64.zip -DestinationPath .`


#### *Step 5:* Train the model with the notebook:
 Activate the environmnet and play with Continuous_Control.ipynb to see the results 


## File Descriptions <a name="files"></a>
The repo is structured as follows:

* Continuous_Control.ipynb: This is where the DDPG agent is trained.
* ddpg_agent.py: This module implements a class to represent a DDPG agent.
* checkpoint_actor.pth: This is the binary containing the trained neural network weights for Actor.
* checkpoint_critic.pth: This is the binary containing the trained neural network weights for Critic.
* model.py: This module contains the implementation of the Actor and Critic neural networks.
* Report.md: Project report and result analysis.
* README.md: Readme file.
* folder:checkpoints: Contains the models saved during training.
* folder:python: This folder has been directly copied from the original repository of Udacity Deep Reinforcement Learning Nanodegree, and contains the files related to                 installation and set up of the environment.
* folder:Images: Contains screenshots of the results as well as additional images used for this document.


## Experiments <a name="experiments"></a>

Follow the instructions in Continuous_Control.ipynb to get started with training your own agent!

Trained model weights is included for quickly running the agent and seeing the result in Unity ML Agent.

Run the last cell of the notebook Continuous_Control.ipynb.



