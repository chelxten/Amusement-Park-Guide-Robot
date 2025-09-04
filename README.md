# Amusement-Park-Guide-Robot

The repository is organised into several main folders, each serving a different part of the project. The controllers folder contains three Webots robot controllers. 

The first is the drl_controller, which implements training code for Deep Reinforcement Learning for obstacle avoidance. 

The second is the full_SLAM_controller, responsible for mapping the environment and handling localisation. 

The third is the full_WeightAstar_controller, which runs the Weighted A* algorithm alongside with all other modules of the system. This is the main controller.

The images folder holds the textures and pictures used inside the Webots world to represent attractions such as rides, food stalls, or shows. These are mainly placeholders that allow the simulation environment to visually resemble a theme park. 

The protos folder contains default Webots PROTO files, which define reusable models for robots and objects within the simulation.

The worlds folder provides the different simulation environments. 

The main file, highway.wbt, represents the amusement park setting where the integrated system is tested. 

Alongside this, there are two additional environments: drl_space.wbt, a simplified space for training and testing the reinforcement learning controller, and training_world.wbt, another training environment mainly used for experiments with DWA and DRL.

Finally, the results_graphs folder gathers the outputs of the evaluation and analysis conducted in the study. This includes performance plots for DRL and DWA, survey-based end user evaluation figures, fuzzy membership function graphs, global planner evaluation metrics for Weighted A*, and clustering with feedback analysis from participant data. Together, these subfolders document both the technical results of the system and the feedback from user studies.
