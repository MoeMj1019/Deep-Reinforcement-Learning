import numpy as np
from rl_environment import GridWorld

class Agent: 
    '''
    agent class
    '''
    def __init__(self, environment = GridWorld(), actions = []) -> None:
        self.states = []
        self.actions = ['L', 'U', 'R', 'D']
        if actions:
            self.actions = actions
        self.stateValueFunc = {} # TODO , change state value function to arrays
        self.actionValueFunc = {} # one item is ( (state,action) , value ) exp. ( (2,2),'U') , 0.6 )
        self.policy = {}
        
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
        for x,y in np.ndindex(environment.grid.shape):
            self.stateValueFunc[(x,y)] = 0
        environment.reset()
        
    def updatePolicy(self, environment:GridWorld): # update policy based on state value function
        for x,y in np.ndindex(environment.grid.shape):
            state = x,y
            if environment.isValidState(state):
                self.policy[(x,y)] = self.argMaxAction(environment,(x,y))
            else:
                self.policy[(x,y)] = None

    def policyGenerator(self,environment:GridWorld, explorationRate=0.2): # heuristic policy
        new_policy = {}
        action_count = len(self.actions)
        if action_count == 0: 
            raise Exception('no actions available for the policy : actions are not defined')
        equal_prob = 1 / action_count
        for x,y in np.ndindex(environment.grid.shape):
            if environment.isTerminal((x,y)): 
                new_policy[(x,y)] = {a : equal_prob  for a in self.actions}

            if environment.isValidState((x,y)):
                goal = environment.terminalStates.get('goal' , [(x,y)])
                x_distance = x - goal[0][0]
                y_distance = y - goal[0][1]
                one_direction_prob = round(1 - explorationRate + (explorationRate / action_count) , 2)
                tow_direction_prob = round( (1 - explorationRate)/2 + (explorationRate / action_count) , 2)
                non_target_prob = round(explorationRate / action_count , 2)

                if x_distance == 0:
                    if y_distance < 0:
                        new_policy[(x,y)] = {a : \
                                        (one_direction_prob  if a == 'R' else non_target_prob)\
                                            for a in self.actions}
                    else:
                        new_policy[(x,y)] = {a : \
                                        (one_direction_prob  if a == 'L' else non_target_prob)\
                                            for a in self.actions}
                elif y_distance == 0:
                    if x_distance < 0:
                        new_policy[(x,y)] = {a : \
                                        (one_direction_prob  if a == 'D' else non_target_prob)\
                                            for a in self.actions}
                    else:
                        new_policy[(x,y)] = {a : \
                                            (one_direction_prob  if a == 'U' else non_target_prob)\
                                            for a in self.actions}
                        
                randomize_check = np.random.choice(4,size=4,replace=False)
                for i in randomize_check:                
                    if x_distance < 0 and y_distance < 0 and i == 0:
                        new_policy[(x,y)] = {a : \
                                                (tow_direction_prob  if a == 'D' or a == 'R' else non_target_prob)\
                                                for a in self.actions}
                    if x_distance < 0 and y_distance > 0 and i == 1:
                        new_policy[(x,y)] = {a : \
                                                (tow_direction_prob  if a == 'D' or a == 'L' else non_target_prob)\
                                                for a in self.actions}
                    if x_distance > 0 and y_distance > 0 and i == 2:
                        new_policy[(x,y)] = {a : \
                                                (tow_direction_prob  if a == 'U' or a == 'L' else non_target_prob)\
                                                for a in self.actions}
                    if x_distance > 0 and y_distance < 0 and i == 3:
                        new_policy[(x,y)] = {a : \
                                                (tow_direction_prob  if a == 'U' or a == 'R' else non_target_prob)\
                                                for a in self.actions}
                        
        for state , possible_actions in new_policy.copy().items():
            prob_actions = np.asarray([possible_actions[self.actions[i]] for i in range(len(self.actions))]).astype('float64')
            prob_actions = prob_actions / np.sum(prob_actions)  # normilizing the probabilities
            for i , prob in enumerate(prob_actions):
                new_policy[state][self.actions[i]] = prob

        return new_policy

    def prob_to_determin_policy(self,environment:GridWorld, policy): # need completion
        deterministic_policy = {}
        action_list = list(list(policy.values())[0].keys())
        for state , possible_actions in policy.items():
            prob_actions = np.asarray([possible_actions[action_list[i]] for i in range(len(action_list))]).astype('float64')
            # prob_actions = prob_actions / np.sum(prob_actions)  # normilizing the probabilities
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
            print('no action found for Agent::argMaxAction')
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

    def evaluatePolicy_MC(self, environment:GridWorld, policy, samples=1000, max_steps=200, gamma=0.9):
        # starting_state = environment.state
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
                prob_actions = np.asarray([policy[state][self.actions[i]] for i in range(len(self.actions))]).astype('float64')
                action = np.random.choice(self.actions, p=prob_actions)

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
        for x,y in np.ndindex(sv_new.shape):
            self.stateValueFunc[(x,y)] = sv_new[x,y]
        return sv_new
    
    def getConvergedStates(self, sv_old, sv_new , theta = 0.001):
        diff = abs(sv_old-sv_new)
        # return np.where(diff < theta)[0]
        indecies = np.where(diff < theta)
        return tuple(zip(indecies[0], indecies[1]))
    
    def getActionProb(self, state, action): # TODO need completion
        return 1
    
    def evaluateValueOfSatate(self, environment:GridWorld, sv_old, gamma):
        value = 0
        expectedRewards = [-np.inf] * len(self.actions)

        for index , action in enumerate(self.actions):
            actionProb = self.getActionProb(environment.state, action) # for compatibility with other algorithems
            if actionProb == 0 or action == None:
                continue

            possible_states = environment.reachableStates(environment.state, action)
            expectedReward = 0
            terminal_marker = 1
            for state in possible_states:
                if len(possible_states) == 1:
                    terminal_marker = 0
                value_state = sv_old[state]
                state_expectedReward = environment.transitionProbFunc(environment.state, state, action) *\
                                        (environment.getReward(environment.state, state, action) +\
                                        gamma * value_state * terminal_marker)
                expectedReward += state_expectedReward
            expectedRewards[index] = expectedReward

            value_action = actionProb * expectedReward
            value += value_action
        
        # if len(self.policy) == 0:
        value = max(expectedRewards)
        return round(value, 3)

    def evaluatePolicy_DP(self, environment:GridWorld, sv_old, convergedStates, gamma=0.9 , ignore=0.3, **kwargs):
        sv_func = np.zeros(shape=environment.grid.shape)
        prev_state = environment.state
        for x,y in np.ndindex(environment.grid.shape):
            state = (x,y)
            if ignore >= np.random.uniform(0,1):
                if state in convergedStates:
                    sv_func[state] = sv_old[state]
                    continue
            if environment.isValidState(state): # evaluate value of valid state (can be entered) 
                environment.updateAgentState(state)
                sv_func[state] = self.evaluateValueOfSatate(environment, sv_old, gamma)
        
        environment.updateAgentState(prev_state)
        for state, value in np.ndenumerate(sv_func):
            self.stateValueFunc[state] = value
        return sv_func

    def valueIteration(self,environment:GridWorld , max_iterations=np.inf, gamma = 1, theta = 0.01, **kwargs):
        self.reset(environment) # ensure policy is empty, for compatibility with other algorithems
        sv_old = None
        sv_new = np.zeros(shape=environment.grid.shape)
        convergedStates = np.zeros(0)
        iteratoin = 0
        while len(convergedStates) < environment.grid.size and iteratoin < max_iterations:
            sv_old = sv_new
            sv_new = self.evaluatePolicy_DP(environment, sv_old, convergedStates, gamma, ignore=kwargs.get('ignore', 0.3))
            convergedStates = self.getConvergedStates(sv_old, sv_new, theta)
            iteratoin += 1

        for state, value in np.ndenumerate(sv_new):
            self.stateValueFunc[state] = value

        self.updatePolicy(environment)
        environment.reset()
        
        return sv_new , iteratoin