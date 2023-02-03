# MDP-DynamicProgramming
Policy Generation for Mobile Robot in obstacle environment using Policy Iteration , Generalised Policy Iteration , and Value Iteration  in both Deterministic , and Stochastic Models. Here process is assumed to be Markov Decision Process , and the problem is solved using Dynamic Programming .

In this programing problem we are going to solve the problem explain in Figure 14.2 of the book “Probabilistic Robotics” using Dynamic Programing. 

The robot can move in 8 directions (4 straight + 4 diagonal). The robot has two model:
    a) Deterministic model, that always executes movements perfectly.
    b) Stochastic model, that has a 20% probability of moving +/-45degrees from the commanded move.
(1 means occupied and 0 means free).
The reward of hitting obstacle is -50.0 .
Reward for any other movement that does not end up at goal is -1.0.
The reward for reaching the goal is 100.0.
The goal location is at W(8,11)
Use gamma =0.95.

You are required to generate the optimal policy for the robot using the following algorithms:

    1) Policy iteration (algorithm on page 80 Sutton and Barto)
    2) Value Iteration (algorithm on page 83 Sutton and Barto)
    3) Generalized Policy Iteration
    

# Outputs 
<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_iteration.gif">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_determinisitc_policy_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/Value_determinisitc_policy_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_deterministic_GPI.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/value_determinisitc_GPI.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_deterministic_value_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/value_determinisitc_value_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_stochastic_policy_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/value_stochastic_policy_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_stochastic_gpi.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/value_stochastic_gpi.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/policy_stochastic_value_iteration.png">

<img src="https://github.com/shivakumar-tekumatla/MDP-DynamicProgramming/blob/master/Outputs/value_stochastic_value_iteration.png">
