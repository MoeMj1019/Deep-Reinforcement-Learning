from rl_environment import GridWorld
import numpy as np
import copy
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(filename)s - %(lineno)d - %(levelname)s : %(message)s',
                     datefmt='%H:%M:%S')
logger = logging.getLogger('my-logger')
# logger.propagate = False
logger.disabled = True

class Agent: 
    '''
    agent class
    '''
    def __init__(self, environment = GridWorld(), actions = []) -> None:
        # self.env = environment
        self.states = []
        self.actions = ['L', 'U', 'R', 'D'] # default actions
        if actions:
            self.actions = actions
        self.actions = list(dict.fromkeys(self.actions)) # make sure there's no duplicates , causes errors for policies
        self.stateValueFunc = {} # one item is ( state , value ) exp. ( (2,2) , 0.6 )
        self.actionValueFunc = {} # one item is ( (state,action) , value ) exp. ( (2,2),'U') , 0.6 )
        self.policy = {} # one item is ( state , probabilities list acording to the action list ) exp. ( (2,2), [0.1,0.2,0.3,0.4] )
        
        for x,y in environment.getValidStates():
            self.stateValueFunc[(x,y)] = 0

    def takeAction(self, environment:GridWorld, action): # take action (deterministicly) and return new state
        position = environment.nextState(action)
        environment.updateAgentState(position)
        return environment.state

    def reset(self,environment:GridWorld):
        self.states = [environment.startState]
        self.policy = {}
        self.stateValueFunc = {}
        # for x,y in np.ndindex(environment.grid.shape):
        self.stateValueFunc= {state : 0 for state in environment.getValidStates()}
        environment.reset()
        
        
    def updatePolicy(self, environment:GridWorld, policy=None): # update policy based on state value function
        if policy == None:
            policy = self.policy

        for state in environment.getValidStates():
            if environment.isValidState(state):
                policy[state] = self.argMaxAction(environment,state)
            else:
                policy[state] = None
        return policy
    

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
                goals = environment.terminalStates.get('goal' , [(x,y)])
                dist_state_goals = np.linalg.norm(np.array(state) - np.array(goals), axis=1)
                goal = goals[np.argmin(dist_state_goals)]
                x_distance = x - goal[0]
                y_distance = y - goal[1]
                one_direction_prob = round(1 - explorationRate + (explorationRate / action_count) , 2)
                tow_direction_prob = round( (1 - explorationRate)/2 + (explorationRate / action_count) , 2)
                non_target_prob = round(explorationRate / action_count , 2)

                if x_distance == 0:
                    if y_distance < 0:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                    else:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                elif y_distance == 0:
                    if x_distance < 0:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'D' 
                                                      else non_target_prob for a in self.actions]) 
                    else:
                        new_policy[(x,y)] = np.array([one_direction_prob if a == 'U' 
                                                      else non_target_prob for a in self.actions]) 
                        
                randomize_check = np.random.choice(4,size=4,replace=False)
                for i in randomize_check:                
                    if x_distance < 0 and y_distance < 0 and i == 0:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'D' or a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                    if x_distance < 0 and y_distance > 0 and i == 1:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'D' or a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                    if x_distance > 0 and y_distance > 0 and i == 2:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'U' or a == 'L' 
                                                      else non_target_prob for a in self.actions]) 
                    if x_distance > 0 and y_distance < 0 and i == 3:
                        new_policy[(x,y)] = np.array([tow_direction_prob if a == 'U' or a == 'R' 
                                                      else non_target_prob for a in self.actions]) 
                        
        for state , possible_actions in new_policy.copy().items():
            prob_actions = possible_actions / np.sum(possible_actions)  # normilizing the probabilities
            new_policy[state] = prob_actions

        return new_policy
    

    def prob_to_determin_policy(self,environment:GridWorld, policy:dict,method='stochastic'): # need completion
        deterministic_policy = {}
        action_list = self.actions.copy()
        for state , prob_actions in policy.items():
            if environment.isTerminal(state):
                action = np.random.choice(action_list)
            else:
                if method == 'stochastic':
                    action = np.random.choice(self.actions, p=prob_actions)
                elif method == 'greedy':
                    action = self.actions[np.argmax(prob_actions)]
                else:  
                    action = np.random.choice(self.actions, p=prob_actions)
            deterministic_policy[state] = action

        return deterministic_policy
    

    def argMaxAction(self,environment:GridWorld, state): # approximation of argmax action
        rest_of_actions = self.actions.copy()
        for action in self.actions:
            if environment.lookAhead(state, action) == state:
                rest_of_actions.remove(action)

        # if len(rest_of_actions) == 0:
        #     #logging.info('no action found for Agent::argMaxAction')
        #     return None
        if len(rest_of_actions) == 1:
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
        

    def evaluatePolicyLoop_MC(self, environment:GridWorld, policy=None, samples=500, max_steps=50, gamma=0.9):
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
            if np.random.uniform(0,1) < 0.25:
                state = environment.startState
            episode = [] # sample an episode
            recieved_rewards = []
            while steps < max_steps:
                steps += 1
                episode.append(state)
                action = np.random.choice(self.actions, p=policy[state])

                # if in a terminal state -> get reward and break
                if environment.isTerminal(state):
                    reward = environment.getReward(state, state, action)
                    recieved_rewards.append(reward)
                    break
                # if not in a terminal state -> take step , get reward and continue
                new_state = environment.interact(state, action)
                reward = environment.getReward(state, new_state, action)
                recieved_rewards.append(reward)
                state = new_state

            terminal_state = episode[-1] # either terminal or the last state in the sequence
            acummulated_reward = recieved_rewards[-1] # last recieved reward
            returns_s[terminal_state] += acummulated_reward
            occurences_s[terminal_state] += 1
            for index , state in enumerate(reversed(episode[:-1])):
                index = len(episode) - index - 2
                acummulated_reward = recieved_rewards[index] + gamma * acummulated_reward
                if state not in episode[:index]:
                    returns_s[state] += acummulated_reward
                    occurences_s[state] += 1

        occured_states = (occurences_s > 0)
        sv_new[occured_states] = np.divide(returns_s[occured_states], occurences_s[occured_states])
        sv_new = np.around(sv_new, 3)
        self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
        return sv_new
    
    
    def getConvergedStates(self, sv_old, sv_new , theta = 0.001, **kwargs):
        diff = abs(sv_old-sv_new)
        # return np.where(diff < theta)[0]
        indecies = np.where(diff < theta)
        return tuple(zip(indecies[0], indecies[1]))
    
    def getActionProb(self, state, action): 
        try:
            if len(self.policy) == 0 or state not in self.policy.keys() or action not in self.actions:
                if len(self.actions) == 0:
                    logging.info('Agent::getActionProb:: agent has no actions avaolable')
                    return 1
                return 1/len(self.actions)
            else:
                return self.policy[state][self.actions.index(action)]
        except IndexError:
            logging.info('Agent::getActionProb::IndexError')
            return 1
            #logging.info('state : ', state)
            #logging.info('action : ', action)
            #logging.info('policy : ', self.policy) 
            pass
        
    
    def evaluateValueOfSatate(self, environment:GridWorld, sv_old, gamma):
        value = 0
        expectedRewards = [-np.inf] * len(self.actions)
        for index , action in enumerate(self.actions):
            actionProb = self.getActionProb(environment.state, action) # actionProb = Pi(a|s)
            # logging.info('evaluateValueOfState: actionProb: ',actionProb)
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
                # print('#################### value_state : ', value_state)
                transistion_Prob = environment.transitionProbFunc(environment.state, state, action)
                state_expectedReward = transistion_Prob * (environment.getReward(environment.state, state, action) +\
                                        gamma * value_state)
                expectedReward += state_expectedReward
            expectedRewards[index] = expectedReward
            # value_action = actionProb * expectedReward
            try:
                value_action = actionProb * expectedReward
            except TypeError:
                logging.info('actionProb = ', actionProb)
                logging.info('expectedReward = ', expectedReward)
                value_action = 0
                pass
            
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
    
# ------------------------------------------------ Util Policy Iteration --------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #

    def isConvergedPolicy(self, policy_old, policy_new, method='close', **kwargs):
        if method == 'exact':
        # Return whether two dicts of arrays are exactly equal
            if policy_old.keys() != policy_new.keys():
                return False
            return all(np.array_equal(policy_old[key], policy_new[key]) for key in policy_old)
    
        else:
        # Return whether two dicts of arrays are roughly equal
            if policy_old.keys() != policy_new.keys():
                return False
            return all(np.allclose(policy_old[key], policy_new[key],rtol=kwargs.get('rtol',1e-2)) for key in policy_old)
        
    # exponentialy wighted average
    def exp_weighted_avg(self,x,y,alpha=0.9):
        return x * alpha + (y * (1-alpha))
        
    
# --------------------------------------------------- DP Policy Iteration -------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #

    def DP_policyIteration(self, environment:GridWorld,policy, max_iterations=np.inf, converged_condition='close',
                           gamma = 0.9, epsilon=0.01,annealing=0.99, **kwargs):
        logging.info('######## ----------- START DP POLICY ITERATION ------------------##########')
        # self.reset(environment)
        prevPolicy = copy.deepcopy(policy)
        improvedPolicy = None
        iteration = 0

        while iteration < max_iterations:
            self.stateValueFunc = {} # reset state value function for the DP evaluation
            improvedPolicy = self.improvePolicy(environment,prevPolicy, gamma=gamma,epsilon=epsilon , **kwargs)
            if self.isConvergedPolicy(prevPolicy,improvedPolicy,method=converged_condition): # if policy is converged
                break
            prevPolicy = improvedPolicy
            iteration += 1
            epsilon *= annealing

        # self.stateValueFunc = { state : value for state, value in np.ndenumerate(self.stateValueFunc) if state in environment.getValidStates()}
        self.policy = improvedPolicy
        return improvedPolicy, iteration

    def improvePolicy(self, environment:GridWorld, policy, gamma = 0.9, **kwargs):
        policy = copy.deepcopy(policy)
        improvedGreedyPolicy = {}
        if len(self.stateValueFunc) == 0:
            self.evaluatePolicy(environment,policy, gamma=gamma, **kwargs)
        improvedGreedyPolicy = self.GreedyPolicy_DP(environment, gamma=gamma, **kwargs)
        # self.policy = improvedGreedyPolicy
        return improvedGreedyPolicy
    
    def evaluatePolicy(self, environment:GridWorld,policy, gamma=0.9, max_iterations=100, evaluationMethod='DP', **kwargs):
        self.policy = copy.deepcopy(policy)
        sv_old = None
        sv_new = np.zeros(shape=environment.grid.shape)
        convergedStates = np.zeros(0)
        iteratoin = 0
        if evaluationMethod == 'DP':
            while len(convergedStates) < environment.grid.size and iteratoin < max_iterations:
                sv_old = sv_new
                sv_new = self.evaluatePolicyLoop_DP(environment,sv_old, convergedStates, gamma, **kwargs)
                # self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
                convergedStates = self.getConvergedStates(sv_old, sv_new, **kwargs)
                iteratoin += 1
        elif evaluationMethod == 'MC':
            while len(convergedStates) < environment.grid.size and iteratoin < max_iterations:
                sv_old = sv_new
                sv_new = self.evaluatePolicyLoop_MC(environment,policy=self.policy, gamma=gamma)
                # self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
                convergedStates = self.getConvergedStates(sv_old, sv_new, **kwargs)
                iteratoin += 1
            # sv_new = self.evaluatePolicyLoop_MC(environment,policy=self.policy, gamma=gamma)
        
        self.stateValueFunc = { state : value for state, value in np.ndenumerate(sv_new) if state in environment.getValidStates()}
        environment.reset()

    def GreedyPolicy_DP(self, environment:GridWorld, gamma=0.9, epsilon=0.2,anneiling_factor=1, **kwargs):     
        # for every valid state , try all actions and choose the one with the highest expected value
        if anneiling_factor != 0:
            epsilon *= anneiling_factor 
        greedyPolicy = {}
        counter = 0
        for state in environment.getValidStates():
            counter += 1
            max_action_value_pair = (None, -np.inf)
            for action in self.actions:
                nextState = environment.lookAhead(state, action)
                possible_states = environment.reachableStates(state, action)
                expectedRward = 0 # action-value function
                for new_state in possible_states:
                    if environment.isTerminal(new_state):
                        action_expectedReward = environment.getReward(state,new_state,action)
                        expectedRward += action_expectedReward
                        continue
                    transitionProb = environment.transitionProbFunc(state, nextState, action)
                    action_expectedReward = transitionProb * (environment.getReward(state,nextState,action)\
                                            + gamma * self.stateValueFunc[nextState])
                    expectedRward += action_expectedReward

                if expectedRward > max_action_value_pair[1]:
                    max_action_value_pair = (action, expectedRward)
            index_of_action = self.actions.index(max_action_value_pair[0])
            greedyPolicy[state] = np.array([( 1 - epsilon ) + ( epsilon/len(self.actions) ) if i == index_of_action 
                             else epsilon/len(self.actions)  for i in range(len(self.actions))])
        # self.policy = policy
        return greedyPolicy


# -------------------------------------------------- MC Policy Iteration --------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #

    def MC_policyIteration(self, environment:GridWorld,policy, max_iterations=np.inf,steps_per_iter=100, converge_condition='close',
                            gamma = 0.9, epsilon=0.2, annealing=0.99, **kwargs):
        logging.info('######## ----------- START MC POLICY ITERATION ------------------##########')
        # self.reset(environment)
        prevPolicy = copy.deepcopy(policy)
        improvedPolicy = None
        iteration = 0

        exploration_rate = kwargs.get('exploration_rate',0.3)

        Q_sa = {}
        returns_sa = {(s,a):[] for s in environment.getValidStates() for a in self.actions}
        # the main loop
        while iteration < max_iterations:
            iteration += 1
            # sample an episode
            # episode is a list of tuples (state, action, reward)
            episode = self.MC_sample_episode(environment, policy, max_steps=steps_per_iter,exploration_rate=exploration_rate, **kwargs)
            # update the Q_values and the returns
            Q_sa , returns_sa = self.MC_evaluate_Qsa(episode, Q_sa, returns_sa, gamma=gamma, **kwargs)
            # update the policy by increasing the probability of the greedy action on behalf of all other actions
            improvedPolicy = self.MC_improvePolicy(environment,Q_values=Q_sa, epsilon=epsilon, **kwargs)

            # check for convergence
            if self.isConvergedPolicy(prevPolicy,improvedPolicy,method=converge_condition,**kwargs): # if policy is converged
                break
            prevPolicy = improvedPolicy
            epsilon *= annealing
            exploration_rate *= annealing

        self.policy = improvedPolicy
        return improvedPolicy, iteration
    
    def MC_sample_episode(self, environment:GridWorld, policy, max_steps=1000, exploration_rate=0.1, **kwargs):
        episode = [] # sample an episode
        steps = 0

        valid_state_index = np.random.choice(len(environment.validStates))
        state = environment.validStates[valid_state_index]
        # start at the start state 20% of the time
        # if np.random.uniform() < 0.2:
        #     state = environment.startState

        while steps < max_steps:
            steps += 1
            # choose an action according to the policy or explore
            if np.random.uniform(0,1) <= exploration_rate:
               action = np.random.choice(self.actions)
            else:
                action = np.random.choice(self.actions, p=policy[state])
            # interact with the environment with the chosen action , get new state and reward and append them to the episode
            new_state = environment.interact(state, action)
            reward = environment.getReward(state, new_state, action)
            episode.append((state,action,reward))
            # if in a terminal state -> break
            if environment.isTerminal(state):
                break
            state = new_state

        return episode
    
    def MC_evaluate_Qsa(self, episode, Q_sa, returns_sa, gamma=0.9, **kwargs):
        # an entry in the episode is a tuple (state, action, reward)
        acummulated_reward = 0
        for index , (s,a,r) in enumerate(reversed(episode[:])):
            index = len(episode) - index - 1
            acummulated_reward = r + gamma * acummulated_reward
            if (s,a) not in [it[:2] for it in episode[:index]]:
                returns_sa[(s,a)] += [acummulated_reward]
                # Q_sa[(s,a)] = self.exp_weighted_avg(Q_sa.get((s,a),0), returns_sa[(s,a)][-1],alpha=0.1) # exponential wighted average
                Q_sa[(s,a)] = np.mean(returns_sa[(s,a)]) # mean
                # Q_sa[(s,a)] = np.sum(returns_sa[(s,a)]) / len(returns_sa[(s,a)]) 
                Q_sa[(s,a)] = np.around(Q_sa[(s,a)], 3)

        return Q_sa, returns_sa

    def MC_improvePolicy(self, environment:GridWorld, Q_values, epsilon , **kwargs):
        greedyPolicy = {}
        for state in environment.getValidStates():
            index_greedy_action = np.argmax([Q_values.get((state,a),0) for a in self.actions])
            greedyPolicy[state] = np.array([( 1 - epsilon ) + ( epsilon/len(self.actions) ) if i == index_greedy_action 
                             else epsilon/len(self.actions)  for i in range(len(self.actions))])
            
            # this is a lazy way to get the best values of the Q_sa function for plotting later on
            self.stateValueFunc[state] = Q_values.get((state,self.actions[index_greedy_action]),0) # TODO change this
        # self.policy = improvedGreedyPolicy
        return greedyPolicy