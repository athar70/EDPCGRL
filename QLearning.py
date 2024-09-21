import pandas as pd
import numpy as np
import random
import logging
from Environment import Environment
from State import State
from Action import Action
import ManageCONST
from ManageSubjectIDs import ManageSubjectIDs

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent:
    def __init__(self, manageSubjectIDs):
        """Initialize the Agent with the given subject manager."""
        self.CONST = ManageCONST.readCONST()
        self.manageSubjectIDs = manageSubjectIDs
        self.env = Environment(self.manageSubjectIDs)
        self.createQTable()
        self.rollout = {'State': [], 'Action': [], 'Reward': []}
        self.lastState = None
        self.allStates = []

    def resetLastState(self):
        """Reset the last state to None."""
        self.lastState = None

    def getRolloutStates(self):
        """Return the list of states in the current rollout."""
        return self.rollout['State']

    def getRolloutActions(self):
        """Return the list of actions in the current rollout."""
        return self.rollout['Action']

    def getRolloutRewardItem(self, index):
        """Return the reward for a specific item in the rollout."""
        return self.rollout['Reward'][index]

    def addListStates(self, state):
        """Add a state to the list of all states."""
        self.allStates.append(state)

    def getListStates(self):
        """Return the list of all states."""
        return self.allStates

    def getListStressLevel(self):
        """Return the stress levels for all states."""
        return [s.getStress() for s in self.allStates]

    def getRolloutStatesItem(self, index):
        """Return a specific state from the rollout."""
        return self.rollout['State'][index]

    def getRolloutActionsItem(self, index):
        """Return a specific action from the rollout."""
        return self.rollout['Action'][index]

    def getRolloutReward(self):
        """Return all rewards from the rollout."""
        return self.rollout['Reward']

    def setRolloutLastStates(self):
        """Set the last state in the rollout."""
        if self.rollout['State']:
            self.lastState = self.rollout['State'][-1]

    def getRolloutLastStates(self):
        """Return the last state from the rollout."""
        return self.lastState

    def resetRollout(self):
        """Reset the rollout data."""
        self.rollout = {'State': [], 'Action': [], 'Reward': []}

    def addRolloutStates(self, new_state):
        """Add a new state to the rollout."""
        self.rollout['State'].append(new_state)

    def addRolloutAction(self, new_action):
        """Add a new action to the rollout."""
        self.rollout['Action'].append(new_action)

    def addRolloutReward(self, new_reward):
        """Add a new reward to the rollout."""
        self.rollout['Reward'].append(new_reward)

    def addRolloutStatesToDF(self):
        """Add the stress levels from the rollout to a DataFrame."""
        listStress = self.getListStressLevel()
        dic = {'subject_id': int(self.numperson), 'stress': listStress}
        self.df_stress = self.df_stress.append(dic, ignore_index=True)

    def resetQTable(self):
        """Reset the Q-table."""
        del self.QTable
        self.createQTable()

    def createQTable(self):
        """Create the Q-table with random initial values."""
        initialTable = self.CONST["QLearning_params"]["InitialTable"]
        if initialTable == 'random':
            random_numbers = np.random.randint(-100, 100, size=(len(self.env.listStates()), len(self.env.listActions()))) / 100
            self.QTable = pd.DataFrame(random_numbers, index=self.env.listStates(), columns=self.env.listActions())
        else:  # Initialize Q-table with zeros
            self.QTable = pd.DataFrame(0, index=self.env.listStates(), columns=self.env.listActions())

        # Set invalid actions in specific states to NaN
        for i, state in enumerate(self.QTable.index):
            for j, action in enumerate(self.QTable.columns):
                st = State([0] * len(self.CONST["ATTR"]), self.manageSubjectIDs)
                st.decodeStates(state)
                ac = Action()
                ac.decodeAction(action)
                next_st = self.env.nextState(st, ac)
                if next_st is None:
                    self.QTable.loc[state, action] = None

    def start(self):
        """Return the initial state for the agent."""
        if self.getRolloutLastStates() is None:
            initial_state = State([0] * len(self.CONST["ATTR"]), self.manageSubjectIDs)
            initial_state.initialRandomState()
            return self.env.findInStates(initial_state)
        return self.getRolloutLastStates()

    def runRollout(self, currState, policy='epsilonGreedy', epsilon=0.2, initial_epsilon=0.6, end_epsilon=0.1):
        """Perform a rollout starting from the current state."""
        maxRollout = self.CONST["QLearning_params"]["MaxRollout"]
        self.resetRollout()
        for _ in range(maxRollout):
            newState = None
            while newState is None:
                if policy == 'epsilonGreedy':
                    action = self.epsilonGreedy(currState, epsilon=epsilon)
                elif policy == 'decayEpsilonGreedy':
                    action = self.decayEpsilonGreedy(currState, initial_epsilon=initial_epsilon, end_epsilon=end_epsilon)
                newState = self.env.nextState(currState, action)

            reward = self.env.calculateReward(currState)
            self.addRolloutStates(currState)
            self.addRolloutAction(action)
            self.addRolloutReward(reward)
            self.addListStates(currState)
            self.setRolloutLastStates()

            # Termination conditions
            if currState.hitGoalStress():
                return

            currState = newState

    def randomActionAvailable(self, state):
        """Return a random valid action available for the given state."""
        state_code = state.codeStates()
        random_action = self.QTable.loc[state_code].dropna().sample(n=1)
        action = Action()
        action.decodeAction(str(random_action.index[0]))
        return action

    def epsilonGreedy(self, currState, epsilon=0.2):
        """Select an action using epsilon-greedy strategy."""
        action = Action()
        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = self.randomActionAvailable(currState)
        else:
            # Exploit: select action with maximum value
            state_code = currState.codeStates()
            best_action = self.QTable.loc[state_code].idxmax()
            action.decodeAction(best_action)
        return action

    def decayEpsilonGreedy(self, currState, initial_epsilon=0.6, end_epsilon=0.1):
        """Select an action using decay epsilon-greedy strategy."""
        action = Action()
        r = max(0, (self.Epochs - self.step) / self.Epochs)
        epsilon = (initial_epsilon - end_epsilon) * r + end_epsilon

        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = self.randomActionAvailable(currState)
        else:
            # Exploit: select action with maximum value
            state_code = currState.codeStates()
            best_action = self.QTable.loc[state_code].idxmax()
            action.decodeAction(best_action)
        return action

    def parseRollout(self, alpha=0.6, gamma=0.8):
        """Update the Q-table based on the rollout data."""
        for t in range(len(self.getRolloutStates()) - 2, -1, -1):  # Go backward
            new_reward = (
                self.QTable.loc[self.getRolloutStatesItem(t).codeStates(), self.getRolloutActionsItem(t).codeAction()] +
                alpha * (self.getRolloutRewardItem(t) + gamma * (np.nanmax(self.QTable.loc[self.getRolloutStatesItem(t + 1).codeStates()]) - 
                self.QTable.loc[self.getRolloutStatesItem(t).codeStates(), self.getRolloutActionsItem(t).codeAction()]))
            )
            
            # Clip new_reward to be within [-1, 1]
            self.QTable.loc[self.getRolloutStatesItem(t).codeStates(), self.getRolloutActionsItem(t).codeAction()] = max(-1, min(1, new_reward))

    def runQLearning(self, policy='epsilonGreedy', epsilon=0.2, initial_epsilon=0.6, end_epsilon=0.1):
        """Run the Q-learning algorithm."""
        gamma = self.CONST["QLearning_params"]["gamma"]
        alpha = self.CONST["QLearning_params"]["alpha"]
        self.step = 0
        self.resetLastState()
        self.env.resetInitial()
        currState = self.start()

        while self.step < self.CONST["Horizon"] and not currState.hitGoalStress() and not currState.hitMaxStress():
            self.runRollout(currState, policy=policy, epsilon=epsilon, initial_epsilon=initial_epsilon, end_epsilon=end_epsilon)
            self.parseRollout(gamma=gamma, alpha=alpha)
            self.step += 1
            currState = self.start()

        if currState.hitGoalStress():
            self.manageSubjectIDs.couldHitGoalStress()

def runQLearning(fullPass=10):
    """
    Run the Q-learning agent for a specified number of iterations.

    Parameters:
    - fullPass (int): Number of iterations the agent will run.
    """
    manageSubjectIDs = ManageSubjectIDs()
    CONST =  ManageCONST.readCONST()

    # Parameters for Q-learning
    policy = CONST["QLearning_params"]["policy"]
    epsilon = CONST["QLearning_params"]["epsilon"]
    initial_epsilon = CONST["QLearning_params"]["initial_epsilon"]
    end_epsilon = CONST["QLearning_params"]["end_epsilon"]

    for i in range(fullPass):
        logging.info(f"Starting iteration {i + 1} of {fullPass} with method Genetic Algorithm")
        agent = Agent(manageSubjectIDs)
        pass_before = manageSubjectIDs.getPassThroughSubjects()
        
        while manageSubjectIDs.getPassThroughSubjects() == pass_before:
            manageSubjectIDs.nextSubjectID()
            agent.runQLearning(policy=policy, epsilon=epsilon, initial_epsilon=initial_epsilon, end_epsilon=end_epsilon)

    # Log results
    logging.info("Final Results:")
    logging.info(f"Percentage Hitting Goal: {manageSubjectIDs.percentage_hit_goal} -- "
                 f"Mean: {manageSubjectIDs.getMeanPercentageHitGoal()} -- "
                 f"Std: {manageSubjectIDs.getStdPercentageHitGoal()}")
    logging.info(f"Avg Presented Spiders: {manageSubjectIDs.avg_presented_spiders} -- "
                 f"Mean: {manageSubjectIDs.getMeanAvgPresentedSpiders()} -- "
                 f"Std: {manageSubjectIDs.getStdAvgPresentedSpiders()}")
    print(len(manageSubjectIDs.percentage_hit_goal))

#runQLearning(3)
