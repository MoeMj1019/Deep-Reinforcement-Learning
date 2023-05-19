import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class GridWorld: 
    '''
    a grid world class
    a state is a tuple (x,y) 
    '''
    def __init__(self,width = 5, hight = 5, startState=(0,0), terminalState={}, terrain={} ,possible_actions=None,
                actions_vectors=None, setRewards={} , environmentDynamics='stochastic') -> None:
        '''
        a state is a tuple (x,y) 
        width : width of environment
        hight : hight of environment
        startState : initial state of an agent in the environment
        terminalState : dictionary of terminal-type : list of states
        terrain : dictionary of terrain-type : list of states
        .
        .
        '''
        self.hight = hight
        self.width = width
        self.grid = np.zeros(shape=(self.hight,self.width))
        self.startState = startState # for the reseting
        self.state = startState
        self.validStates = tuple()
        if possible_actions is None or actions_vectors is None:
            self.possibleActions = ['L', 'U', 'R', 'D'] # default actions
            self.actionsVectors = {'L': (0, -1), 'U': (-1, 0), 'R': (0, 1), 'D': (1, 0)} # default actions vectors
        else:
            self.possibleActions = possible_actions
            self.actionsVectors = actions_vectors
        self.terminalStates = terminalState # dictionsry/map of terminal state types as keys and positions as values
        self.terrain = terrain # dictionsry/map of terrain types as keys and positions as values
        self.setRewards = setRewards
        self.transitionProb = {}
        self.environmentDynamics = environmentDynamics
        self.terminated = False
        self.step = 0

        if self.environmentDynamics == 'stochastic': # later implementation for stochastic MDP
            pass

        if not self.possibleActions:
            self.possibleActions = ['L', 'U', 'R', 'D']

        ## TODO check validity of the provided states 
        self.__configure()


# ------ Grid configurations and settings ------
    def __configure(self):
        self.grid[:] = 0
        for type , states in self.terminalStates.items():
            for state in states:
                self.grid[state] = 1
        for type , states in self.terrain.items():
            for state in states:
                self.grid[state] = -1
        self.__updateValidStates()
    
    def __updateValidStates(self):
        valid_states_indecies = np.isin(self.grid, [0,1], invert=False).nonzero()
        self.validStates = tuple(zip(valid_states_indecies[0],valid_states_indecies[1]))

    def configurations(self,size=(5,5),config=0): # random
        pass

    def addTerminal(self, *type_value_pairs):
        for pair in  type_value_pairs:
            if pair[0] in self.terminalStates.keys():
                # if pair[1] not in self.terminalStates[pair[0]]:
                newValues = set(self.terminalStates[pair[0]] + pair[1])
                self.terminalStates[pair[0]] = list(newValues)
            else:
                self.terminalStates[pair[0]] = pair[1]
        self.__configure()

    def addTerrain(self, *type_value_pairs):
        for pair in  type_value_pairs:
            if pair[0] in self.terrain.keys():
                # if pair[1] not in self.terrain[pair[0]]:
                newValues = set(self.terrain[pair[0]] + pair[1])
                self.terrain[pair[0]] = list(newValues)
            else:
                self.terrain[pair[0]] = pair[1]
        self.__configure()
    
    def deleteObject(self, *objects_states):
        for obj_state in objects_states:
            for states in self.terminalStates.values():
                if obj_state in states:
                    states.remove(obj_state)
                    break
            for states in self.terrain.values():
                if obj_state in states:
                    states.remove(obj_state)
                    break
        self.__configure()
    
    def configuration(self,**kwargs): # set the configuration of the environment
        self.terminalStates = kwargs.get('terminalStates', self.terminalStates)
        self.terrain = kwargs.get('terrain', self.terrain)
        self.startState = kwargs.get('startState', self.startState)
        self.setRewards = kwargs.get('setRewards', self.setRewards)
        self.possibleActions = kwargs.get('possibleActions', self.possibleActions)
        self.actionsVectors = kwargs.get('actionsVectors', self.actionsVectors)
        self.transitionProb = kwargs.get('transitionProb', self.transitionProb)
        self.environmentDynamics = kwargs.get('environmentDynamics', self.environmentDynamics)
        self.__configure()

# ------ informations about states and grid conditions ------
    def getValidStates(self) -> tuple: # returns a tuple of valid states
        return self.validStates
    
    def isAtTerminal(self): # check if the agent is at a terminal state and mark the environment as terminated
        if self.state in [s for valueList in self.terminalStates.values() for s in valueList]:
            self.terminated = True
            return True
        return False

    def isTerminal(self, state): # check if a state is a terminal state , without marking the environment as terminated
        if state in [s for valueList in self.terminalStates.values() for s in valueList]:
            return True
        return False
    
    def isValidState(self, state):
        return state in self.validStates
    
    # def isInGrid(self, state):
    #     return (state[0] >= 0) and (state[0] <= (self.hight -1)) and (state[1] >= 0) and (state[1] <= (self.width -1))

    def reset(self): # reset the environment to the initial state
        self.terminated = False
        self.state = self.startState
        self.step = 0
        # self.__configure()

    def updateAgentState(self, state): # update the agent state and increase the step counter
        if not self.isValidState(state):
            raise ValueError("Invalid state")
        self.state = state
        self.step += 1

    def nextState(self, action): # returns the next state given the current state and the action
        if action not in self.possibleActions:
            raise ValueError("Invalid action")
        
        nextState = self.state
        if action == "L":
            nextState = (self.state[0], self.state[1] - 1)
        elif action == "U":
            nextState = (self.state[0] - 1, self.state[1])
        elif action == "R":
            nextState = (self.state[0], self.state[1] + 1)
        elif action == "D":
            nextState = (self.state[0] + 1, self.state[1])
        # consider terrain dynamics
        if nextState in self.terrain.get('shortcut' , []):
            nextState = (2*nextState[0]-self.state[0] , 2*nextState[1]-self.state[1])
        if self.isValidState(nextState):
            return nextState
        # otherwise, the next move in not valid and we stay at the same state
        return self.state
    
    def lookAhead(self, state , action):
        current_state = self.state
        self.state = state
        next = self.nextState(action)
        self.state = current_state
        return next
    
    def reachableStates(self, state, theta = 0.01): # returns a list of reachable states from a given state
        for type_list in self.terminalStates.values():
            if state in type_list: # if agent is in terminal state no transition possible
                    return [state]
            
        states_list = [state,
                       (state[0]+1, state[1]),(state[0]-1, state[1]),
                       (state[0], state[1]+1),(state[0], state[1]-1)] # TODO ? generelize 
        x , y = state
        for s in states_list.copy():
            if s in self.terrain.get('shortcut',[]): # handel shortcut terrain
                states_list.append(( 2*s[0] - x , 2*s[1] -y ))
        
        for s in states_list.copy():
            # if not self.isValidState(s) or self.transitionProbFunc(state,s,action) > theta:
            if not self.isValidState(s) or s in self.terrain.get('shortcut',[]):
                states_list.remove(s)

        return states_list

    def transitionProbFunc(self, old_state, new_state , action):
        for type_list in self.terminalStates.values():
            if old_state in type_list: # if agent is in terminal state no transition possible
                if old_state == new_state:
                    return 1
                return 0
        # stochastic case
        if self.environmentDynamics == 'stochastic':
            trans_prob = 0
            if new_state == self.lookAhead(old_state, action): # the state aligns with the action
                trans_prob +=  0.9
            # delta_s = self.state - state
            # x , y = self.state
            # perpendicular_states = [(x + delta_s[1], y + delta_s[0]),(x - delta_s[1], y - delta_s[0])]
            up_down = ['U','D']
            left_right = ['L','R']
            if action in up_down: # for (up,down)
                for a in left_right: # perpendicular states check 
                    if new_state == self.lookAhead(old_state, a):
                        trans_prob += 0.05
            if action in left_right: # for (left,right)
                for a in up_down: # perpendicular states check
                    if new_state == self.lookAhead(old_state, a):
                        trans_prob += 0.05
            # else any other state
            return trans_prob
        # deterministic case
        elif self.environmentDynamics == 'deterministic':
            if new_state == self.lookAhead(old_state, action): # the state aligns with the action
                return 1
            return 0
        else:
            return 0
        
    def interact(self, state, action): # return the next state according to the transition probabilities
        possible_states = self.reachableStates(state, action)
        prob_states = np.asarray([self.transitionProbFunc(state, new_s, action) for new_s in possible_states]).astype('float64')
        # prob_states = prob_states / np.sum(prob_states) # normilizing the probabilities
        try:
            new_state_index = np.random.choice(len(possible_states), p=prob_states)
        except ValueError: ## ERORO 
            print('ValueError, GridWorld::interact :', state,action,possible_states,prob_states)
        new_state = possible_states[new_state_index]

        return new_state

    def getReward(self, old_state, new_state, action): # returns the reward for a given transition
        # if agent is at terminal states
        if old_state in self.terminalStates.get('goal',[]):
            return self.setRewards.get('goal',1)
        if old_state in self.terminalStates.get('negative_goal',[]):
            return self.setRewards.get('negative_goal',-1)
        # else deafult reward
        return self.setRewards.get('default',-0.02)
