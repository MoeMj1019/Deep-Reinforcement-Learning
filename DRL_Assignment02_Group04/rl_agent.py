from rl_environment import GridWorld
import numpy as np
import copy

class Agent: 
    '''
    agent class
    '''
    def __init__(self, environment = GridWorld(), actions = []) -> None:
        self.states = []
        self.actions = ['L', 'U', 'R', 'D'] # default actions
        if actions:
            self.actions = actions
        self.actions = list(dict.fromkeys(self.actions)) # make sure there's no duplicates , causes errors for policies
        self.stateValueFunc = {} # one item is ( state , value ) exp. ( (2,2) , 0.6 )
        self.actionValueFunc = {} # one item is ( (state,action) , value ) exp. ( (2,2),'U') , 0.6 )
        self.policy = {} # one item is ( state , probabilities list acording to the action list ) exp. ( (2,2), [0.1,0.2,0.3,0.4] )
        
        for x,y in np.ndindex(environment.grid.shape):
            self.stateValueFunc[(x,y)] = 0

    def takeAction(self, environment:GridWorld, action): # take action (deterministicly) and return new state
        position = environment.nextState(action)
        environment.updateAgentState(position)
        return environment.state

    def reset(self,environment:GridWorld):
        self.states = [environment.state]
        self.policy = {}
        self.stateValueFunc = {}
        # for x,y in np.ndindex(environment.grid.shape):
        self.stateValueFunc= {state : 0 for state in environment.getValidStates()}
        environment.reset()
        
    def updatePolicy(self, environment:GridWorld): # update policy based on state value function
        for state in environment.getValidStates():
            if environment.isValidState(state):
                self.policy[state] = self.argMaxAction(environment,state)
            else:
                self.policy[state] = None

    def policyGenerator(self,environment:GridWorld, explorationRate=0.2): # heuristic policy
        new_policy = {}
        action_count = len(self.actions)
        if action_count == 0: 
            raise Exception('no actions available for the policy : actions are not defined')
        equal_prob = 1 / action_count
        for state in environment.getValidStates():
            x,y = state
            if environment.isTerminal((x,y)): 
                new_policy[state] = np.array([equal_prob for a in self.actions]) 

            if environment.isValidState((x,y)):
                goal = environment.terminalStates.get('goal' , [(x,y)])
                x_distance = x - goal[0][0]
                y_distance = y - goal[0][1]
                one_direction_prob = round(1 - explorationRate + (explorationRate / action_count) , 2)
                tow_direction_prob = round( (1 - explorationRate)/2 + (explorationRate / action_count) , 2)
                non_target_prob = round(explorationRate / action_count , 2)

                if x_distance == 0:
                    if y_distance < 0:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                 (one_direction_prob  if a == 'R' else non_target_prob)\
                        #                     for a in self.actions}
                    else:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                 (one_direction_prob  if a == 'L' else non_target_prob)\
                        #                     for a in self.actions}
                elif y_distance == 0:
                    if x_distance < 0:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'D' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                 (one_direction_prob  if a == 'D' else non_target_prob)\
                        #                     for a in self.actions}
                    else:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'U' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                     (one_direction_prob  if a == 'U' else non_target_prob)\
                        #                     for a in self.actions}
                        
                randomize_check = np.random.choice(4,size=4,replace=False)
                for i in randomize_check:                
                    if x_distance < 0 and y_distance < 0 and i == 0:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'D' or a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                         (tow_direction_prob  if a == 'D' or a == 'R' else non_target_prob)\
                        #                         for a in self.actions}
                    if x_distance < 0 and y_distance > 0 and i == 1:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'D' or a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                         (tow_direction_prob  if a == 'D' or a == 'L' else non_target_prob)\
                        #                         for a in self.actions}
                    if x_distance > 0 and y_distance > 0 and i == 2:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'U' or a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                         (tow_direction_prob  if a == 'U' or a == 'L' else non_target_prob)\
                        #                         for a in self.actions}
                    if x_distance > 0 and y_distance < 0 and i == 3:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'U' or a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                        # {a : \
                        #                         (tow_direction_prob  if a == 'U' or a == 'R' else non_target_prob)\
                        #                         for a in self.actions}
                        
        for state , possible_actions in new_policy.copy().items():
            prob_actions = possible_actions / np.sum(possible_actions)  # normilizing the probabilities
            new_policy[state] = prob_actions

        return new_policy

    def prob_to_determin_policy(self,environment:GridWorld, policy:dict): # need completion
        deterministic_policy = {}
        action_list = self.actions.copy()
        for state , prob_actions in policy.items():
            if environment.isTerminal(state):
                action = np.random.choice(action_list)
            else:
                action = np.random.choice(self.actions, p=prob_actions)
            deterministic_policy[state] = action

        return deterministic_policy

    def argMaxAction(self,environment:GridWorld, state): # approximation of argmax action
        rest_of_actions = self.actions.copy()
        for action in self.actions:
            if environment.lookAhead(state, action) == state:
                rest_of_actions.remove(action)

        if len(rest_of_actions) == 0:
            #print('no action found for Agent::argMaxAction')
            return None
        elif len(rest_of_actions) == 1:
            return rest_of_actions[0]  
        else:
            action = np.random.choice(rest_of_actions)  
            rest_of_actions.remove(action)
            mx_nxt_reward =  self.stateValueFunc[environment.lookAhead(state, action)]
            for a in rest_of_actions:
                nxt_reward = self.stateValueFunc[environment.lookAhead(state, a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            return action

    def evaluatePolicyLoop_MC(self, environment:GridWorld, policy=None, samples=1000, max_steps=200, gamma=0.9):
        try:
            it = iter(policy)
        except TypeError:
            policy = self.policy
        if len(policy) != len(environment.getValidStates()):
            raise ValueError('no valid policy provided for the MC policy evaluation')
        sv_new = np.zeros(shape=environment.grid.shape)
        returns_s = np.zeros(shape=environment.grid.shape)
        occurences_s = np.zeros(shape=environment.grid.shape)
        i = 0
        while i < samples:
            i += 1
            steps = 0
            valid_state_index = np.random.choice(len(environment.validStates))
            state = environment.validStates[valid_state_index]
            episode = [] # sample an episode
            recieved_reward = []
            while steps < max_steps:
                episode.append(state)
                action = np.random.choice(self.actions, p=policy[state])

                # if in a terminal state -> get reward and break
                if environment.isTerminal(state):
                    reward = environment.getReward(state, state, action)
                    recieved_reward.append(reward)
                    break
                # if not in a terminal state -> take step , get reward and continue
                new_state = environment.interact(state, action)
                reward = environment.getReward(state, new_state, action)
                recieved_reward.append(reward)
                state = new_state
                steps += 1

            terminal_state = episode[-1] # either terminal or the last state in the sequence
            acummulated_reward = recieved_reward[-1] # last recieved reward
            returns_s[terminal_state] += acummulated_reward
            occurences_s[terminal_state] += 1
            for index , state in enumerate(reversed(episode[:-1])):
                acummulated_reward = recieved_reward[index] + gamma * acummulated_reward
                if state not in episode[:index]:
                    returns_s[state] += acummulated_reward
                    occurences_s[state] += 1

        occured_states = (occurences_s > 0)
        sv_new[occured_states] = np.divide(returns_s[occured_states], occurences_s[occured_states])
        sv_new = np.around(sv_new, 3)
        self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
        return sv_new
    
    def getConvergedStates(self, sv_old, sv_new , theta = 0.001):
        diff = abs(sv_old-sv_new)
        # return np.where(diff < theta)[0]
        indecies = np.where(diff < theta)
        return tuple(zip(indecies[0], indecies[1]))
    
    def getActionProb(self, state, action): 
        try:
            if len(self.policy) == 0 or state not in self.policy.keys() or action not in self.actions:
                return 1/len(self.actions)
            else:
                return self.policy[state][self.actions.index(action)]
        except IndexError:
            #print('Agent::getActionProb::IndexError')
            #print('state : ', state)
            #print('action : ', action)
            #print('policy : ', self.policy) 
            pass
        
    
    def evaluateValueOfSatate(self, environment:GridWorld, sv_old, gamma):
        value = 0
        expectedRewards = [-np.inf] * len(self.actions)
        for index , action in enumerate(self.actions):
            actionProb = self.getActionProb(environment.state, action) # actionProb = Pi(a|s)
            if actionProb == 0 or action == None:
                continue

            possible_states = environment.reachableStates(environment.state, action)
            expectedReward = 0
            for state in possible_states:
                if environment.isTerminal(environment.state):
                    state_expectedReward = environment.getReward(environment.state, state, action)
                    expectedReward += state_expectedReward
                    continue
                value_state = sv_old[state]
                transistion_Prob = environment.transitionProbFunc(environment.state, state, action)
                state_expectedReward = transistion_Prob * (environment.getReward(environment.state, state, action) +\
                                        gamma * value_state)
                expectedReward += state_expectedReward
            expectedRewards[index] = expectedReward
            value_action = actionProb * expectedReward
            # try:
            #     value_action = actionProb * expectedReward
            # except TypeError:
            #     #print('actionProb = ', actionProb)
            #     #print('expectedReward = ', expectedReward)
            #     pass
            value_action = 0
            value += value_action
        
        if len(self.policy) == 0:
            value = max(expectedRewards)
        return round(value, 3)

    def evaluatePolicyLoop_DP(self, environment:GridWorld, sv_old=None, convergedStates=None , gamma=0.9 , ignore=0.3, **kwargs):
        try:
            it = iter(sv_old)
        except TypeError:
            sv_old = np.zeros(shape=environment.grid.shape)
        if len(sv_old) == 0:
            sv_old = np.zeros(shape=environment.grid.shape)
        try:
            it = iter(convergedStates)
        except TypeError:
            convergedStates = []
        
        sv_new = np.zeros(shape=environment.grid.shape)
        prev_state = environment.state
        for state in environment.getValidStates():
            if ignore >= np.random.uniform(0,1):
                if state in convergedStates:
                    sv_new[state] = sv_old[state]
                    continue
            if environment.isValidState(state): # evaluate value of valid state (can be entered) 
                environment.updateAgentState(state)
                sv_new[state] = self.evaluateValueOfSatate(environment, sv_old, gamma)
        
        environment.updateAgentState(prev_state)
        self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
        return sv_new

    def valueIteration(self,environment:GridWorld , max_iterations=np.inf, gamma = 1, theta = 0.01, **kwargs):
        self.reset(environment) # ensure policy is empty, for compatibility with other algorithems
        sv_old = None
        sv_new = np.zeros(shape=environment.grid.shape)
        convergedStates = np.zeros(0)
        iteratoin = 0
        while len(convergedStates) < environment.grid.size and iteratoin < max_iterations:
            sv_old = sv_new
            sv_new = self.evaluatePolicyLoop_DP(environment, sv_old, convergedStates, gamma, **kwargs)
            convergedStates = self.getConvergedStates(sv_old, sv_new, theta)
            iteratoin += 1

        self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
        self.updatePolicy(environment)
        environment.reset()

        return sv_new , iteratoin



