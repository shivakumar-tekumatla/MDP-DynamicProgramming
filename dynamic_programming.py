"""
In this programing assignment we are going to solve the problem explain in Figure 14.2 of the book “Probabilistic Robotics” using Dynamic Programing. (You don’t really to refer to that book, everything needed is provided here)
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

"""
import numpy as np 
from itertools import product 
import matplotlib.pyplot as plt
class Robot:
    def __init__(self,world,n_actions =8,gamma= 0.95,goal=(7,10),r_clr = -1.0,r_obs = -50,r_goal = 100,theta=1e-8,deterministic=True) -> None:
        self.world = world #defines the world 
        self.x_limit , self.y_limit = self.world.shape #size of the world 
        self.states = [state for state in product(np.arange(self.x_limit),np.arange(self.y_limit)) if self.world[state]==0] # remove boundaries and obstacles from the states 
        self.n_states = self.x_limit*self.y_limit # number of states 
        self.n_actions = n_actions # number of actions 
        self.gamma = gamma # discount factor 
        self.goal = goal #goal location 
        self.r_clr = r_clr # reward for any movement that doe snot reach goal or does not hit obstacle  
        self.r_goal = r_goal #reward if reaches goal 
        self.r_obs = r_obs #reward if hits obstacle 
        self.theta = theta # tolerance value
        self.deterministic = deterministic # Model is deterministic or stochastic 
        # robot can perform 8 actions. go up , down , left right or four diagonal directions. 
        # for each of these actions, the next state can simply be defined by changing the id of the current state.
        # For example if the robot is at the position (0,0) , and took right action , then it simply moves to (0,1)
        # probability of moving to an another state depends on if the model is stochastic or deterministic 
        self.actions_states = {"u":(-1,0),
                        "d":(1,0),
                        "r":(0,1),
                        "l":(0,-1),
                        "ur":(-1,1),
                        "ul":(-1,-1),
                        "dr":(1,1),
                        "dl":(1,-1)
                        } # for an action and if the model is deterministic , this is the next state 
        
        if self.deterministic:
            self.actions = {key:[(key,1)] for key in self.actions_states} #probability of moving in the same direction is 1 
        else:
            # if stochastic 
            # only has 60% probability of moving in that direction , else 0.2 in either +/- diagonal to that action 
            self.actions = {"u":[("u",0.6),("ul",0.2),("ur",0.2)],
                            "ul":[("ul",0.6),("u",0.2),("l",0.2)],
                            "ur":[("ur",0.6),("u",0.2),("r",0.2)],
                            "d":[("d",0.6),("dl",0.2),("dr",0.2)],
                            "dl":[("dl",0.6),("d",0.2),("l",0.2)],
                            "dr":[("dr",0.6),("d",0.2),("r",0.2)],
                            "l":[("l",0.6),("ul",0.2),("dl",0.2)],
                            "r":[("r",0.6),("ur",0.2),("dr",0.2)],}
        self.state_value = np.zeros_like(self.world) # initial action_value function  
        self.future_state_value = np.zeros_like(self.world)

        # Element wise tuple sum
        self.tuple_sum = lambda state,action:tuple(map(sum,zip(state,self.actions_states[action])))

        # Initialization:  value and policy
        self.init_value_function = np.zeros_like(self.world) 
        self.init_policy = {}
        for state in self.states:
            self.init_policy[state] = {key:1/self.n_actions for key in self.actions.keys()} # for now let's assume the probability of selecting any action at each state is same 

        pass

    def policy_evaulation(self,policy,value_function,delta=0):
        for state in self.states:
            val =0 
            # for each state , get the probability of a given action 
            for action in self.actions:
                pi = policy[state][action] 
                # for a given action , with some probability , get the probability of reaching next state and reward 
                for _a , prob in self.actions[action]:

                    next_state = self.tuple_sum(state,_a)
                    if next_state not in self.states:
                    #if the next state is out of bound 
                        continue
                    reward = self.reward(next_state)
                    val+=pi*prob*(reward+self.gamma*value_function[next_state])
            delta = max(delta,abs(val-value_function[state]))
            value_function[state] = val 
        return value_function, delta 


    def policy_improvement(self,policy,value,stable=True):
        for state in self.states:
            # for this state , get the action that has the better probability of picking 
            old_action = max(policy[state], key=policy[state].get, default=None)  #key with max value 
            action_value = {key:0 for key in self.actions}#np.zeros(self.n_actions) # assume value of each action is zero for now 

            for action in self.actions:
                for _a , prob in self.actions[action]:
                    next_state = self.tuple_sum(state,_a)
                    if next_state not in self.states:
                        #if the next state is out of bound 
                        continue
                    reward = self.reward(next_state)   
                    action_value[_a]+= prob*(reward+self.gamma*value[next_state])   

            best_action = max(action_value, key=action_value.get, default=None)
            if old_action!=best_action:
                stable = False # the action we guessed is not  same as the best one 
            policy[state] = dict.fromkeys(policy[state],0)
            policy[state][best_action] = 1 # set this value to max so this will be chosen in next iteration
        return policy,value,stable 

    def policy_iteration(self,live_plot=False):
        # Initialization
        policy = self.init_policy.copy()
        value = self.init_value_function.copy()
        while True:
            if live_plot:
                self.plot_policy(policy,"Policy Iteration Live Policy Plot")
                self.plot_value(value,"Policy Iteration Live Value Plot")
            # Policy evaluation 
            delta = 0
            while delta <self.theta:
                value,delta = self.policy_evaulation(policy,value,delta=delta)
            # policy improvement 
            policy ,value,stable = self.policy_improvement(policy,value)
            
            if stable:
                break

        return policy,value #self.policy_evaluation(policy) 

    def generalized_policy_iteration(self,live_plot=False):
        # Initialization
        policy = self.init_policy.copy()
        value = self.init_value_function.copy()
        while True:
            if live_plot:
                self.plot_policy(policy," Generalized Policy Iteration Policy plot")
                self.plot_value(value,"Generalized Policy Iteration Value Plot")
            # Policy evaluation 
            # But here we are not waiting for delta to converge 
            value,_ = self.policy_evaulation(policy,value)
            # policy improvement 
            policy ,value,stable = self.policy_improvement(policy,value)
            if stable:
                break
        return policy,value #self.policy_evaluation(policy) 

    def value_iteration(self,live_plot= False):
        value = self.init_value_function.copy() 
        delta =0 
        while True:
            delta =0 
            for state in self.states:
                action_value = {key:0 for key in self.actions}
                for action in self.actions:
                    for _a,prob in self.actions[action]:
                        next_state = self.tuple_sum(state,_a)
                        if next_state not in self.states:
                            #if the next state is out of bound 
                            continue
                        reward = self.reward(next_state)   
                        action_value[_a]+= prob*(reward+self.gamma*value[next_state]) 
                best_val = max(action_value.values()) #action_value[max(action_value, key=action_value.get, default=None)] # best value 
                delta = max(delta,abs(best_val-value[state]))
                value[state] = best_val
            if delta <self.theta:
                break 

        # now we found the best value function iteratively.
        # can we use the same value function to find the optimal policy ? 

        policy = {}
        for state in self.states:
            policy[state] =  {key:0 for key in self.actions.keys()} # for now let's assume the probability of selecting any action zero 
        # this is because there is no policy at all , but only value function 
        # this is same as improving the policy for a given value but , in this case the policy function is just zero 
        policy ,value,_ = self.policy_improvement(policy,value)
        return policy,value


    def reward(self,state):
        # if the robots new state after an action is obstacle the reward is r_obs
        # if the goal then , r_goal 
        # else r_clr 

        if state == self.goal:
            return self.r_goal
        elif self.world[state[:]] == 1:
            return self.r_obs 
        else:
            return self.r_clr 
    
    def plot_policy(self,policy,title):
        plt.title(title)
        gy,gx = self.goal 
        plt.plot(gx,gy,"ro")
        plt.imshow(1-self.world,cmap="gray")
        for state in policy:
            for key in policy[state]:
                if policy[state][key] ==1:
                    y,x = state
                    dy,dx = self.actions_states[key]
                    mul = 0.5
                    plt.arrow(x,y,mul*dx,mul*dy,width=0.05)
                    break 
        plt.show() 

    def plot_value(self,value,title):
        plt.title(title)
        gy,gx = self.goal 
        plt.plot(gx,gy,"ro")
        plt.imshow(1-self.world,cmap="gray")
        plt.imshow(value,cmap="gray")
        plt.show()


if __name__ == "__main__":

    world = np.loadtxt("world.txt") # loads the world 
    goal = (8,11) # this is given in matlab indexing . Change it to python 
    goal = (goal[0]-1,goal[1]-1) 
    robot = Robot(world,goal=goal)
    policy,value = robot.policy_iteration()
    robot.plot_policy(policy,"Control Policy for Deterministic Policy Iteration")
    robot.plot_value(value,"Value Function for Deterministic Policy Iteration")
    policy,value = robot.generalized_policy_iteration()
    robot.plot_policy(policy,"Control Policy for Deterministic Generalized Policy Iteration")
    robot.plot_value(value,"Value Function for Deterministic Generalized Policy Iteration")
    policy,value = robot.value_iteration()
    robot.plot_policy(policy,"Control Policy for Deterministic Value Iteration")
    robot.plot_value(value,"Value Function for Deterministic Value Iteration")
    robot = Robot(world,goal=goal,deterministic=False)
    policy,value = robot.policy_iteration()
    robot.plot_policy(policy,"Control Policy for Stochastic Policy Iteration")
    robot.plot_value(value,"Value Function for Stochastic Policy Iteration")
    policy,value = robot.generalized_policy_iteration()
    robot.plot_policy(policy,"Control Policy for Stochastic Generalized Policy Iteration")
    robot.plot_value(value,"Value Function for Stochastic Generalized Policy Iteration")
    policy,value = robot.value_iteration()
    robot.plot_policy(policy,"Control Policy for Stochastic Value Iteration")
    robot.plot_value(value,"Value Function for Stochastic Value Iteration")





