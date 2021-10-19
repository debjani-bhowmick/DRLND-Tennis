# DRLND-continous-learning

## Table of Contents

1. [Summary](#summary)
2. [Getting Started](#GettingStarted) 
3. [File Descriptions](#files)
4. [Experiments](#experiments)

##  Summary <a name="summary"></a>

In this project, you will work with the Tennis environment. In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

`Environment Solving Criteria: The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5`

## Getting Started <a name="Getting Started"></a>

#### *Step 1:* Clone the repo:
`git clone https://github.com/debjani-bhowmick/DRLND-continous-learning.git` 

#### *Step 2:*Change directory into the repo:
`cd Tennis`

#### *Step 3:*  Activate the Environment

For details related to setting up the Python environment for this project, please follow the instructions provided in the DRLND GitHub repository[https://github.com/udacity/deep-reinforcement-learning]. These instructions can be found in README.md at the root of the repository. By following these instructions, user will be able to install the required PyTorch library, the ML-Agents toolkit, and a few more Python packages required for this project.

(For Windows users) The ML-Agents toolkit supports Windows 10 currently. In general, ML-Agents toolkit could possibly be used for other versions, however, it has not been tested officially, and we recommend choosing Windows 10. Also, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Further, the specific files to look into in the repository is python/setup.py and requiremnets.txt. The readme provides thorough details related to setting up the environment.




#### *Step 4:* Download the Unity Environment

For this project, you will not need to install Unity - this is because environment has buit for you, and you can download it from one of the links below. You need only select the environment that matches your operating system

Download the environment from one of the links below. You need only select the environment that matches your operating system:

* Linux: (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip]
* Mac OSX: (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip]
* Windows (32-bit): (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip]
* Windows (64-bit): (click here)[https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip]

(For Windows users) Check out this link(https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

Then, place the file in the Tennis/ folder in the repository.

#### *Step 5:*  Unzip (or decompress) the downloaded file:

* Linux:
`unzip Reacher_Linux.zip`
* Mac OSX:
`unzip -a Reacher.app.zip`
* Windows (32-bit): [PowerShell]
`Expand-Archive -Path Reacher_Windows_x86.zip -DestinationPath .
*Windows (64-bit): [PowerShell]
`Expand-Archive -Path Tennis_Windows_x86_64.zip -DestinationPath .`


#### *Step 5:* Train the model with the notebook:

Follow the instructions from Tennis.ipynb


## File Descriptions <a name="files"></a>
The repo is structured as follows:

* Tennis.ipynb: This is where the MADDPG agent is trained.
* maddpg_agent.py: This module implements MADDPG algorithm.
* models/checkpoint_actor_0.pth: This is the binary containing the trained neural network weights for 1st Actor.
* models/checkpoint_critic_0.pth: This is the binary containing the trained neural network weights for 1st Critic.
* models/checkpoint_actor_1.pth: This is the binary containing the trained neural network weights for 2nd Actor .
* models/checkpoint_critic_1.pth: This is the binary containing the trained neural network weights for 2nd Critic.
* model.py: This module contains the implementation of the Actor and Critic neural networks.
* Report.md: Project report and result analysis.
* README.md: Readme file.
* folder:models: Contains the models saved during training.
* folder:python: This folder has been directly copied from the original repository of Udacity Deep Reinforcement Learning Nanodegree, and contains the files related to                 installation and set up of the environment.
* folder:Images: Contains screenshots of the results as well as additional images used for this document.


## Experiments <a name="experiments"></a>

Follow the instructions in Tennis.ipynb to get started with training your own agent!

Trained model weights is included for quickly running the agent and seeing the result in Unity ML Agent.





