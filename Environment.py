import pandas as pd
import ManageCONST
from State import State 
from Action import Action
import numpy as np
import random 
from scipy.stats import norm
import copy
from ManageSubjectIDs import ManageSubjectIDs
import logging
import itertools

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Environment:
    def __init__(self, manageSubjectIDs):  
        """Initialize the environment for the genetic algorithm."""
        self.CONST = ManageCONST.readCONST()
        self.manageSubjectIDs = manageSubjectIDs
        self.populationSize = self.CONST["GA_params"]["PopulationSize"]
        self.allStates = []
        self.resetInitial()
        

    def reset(self):
        """Reset the states and actions."""
        self.states = []
        self.actions = []

    def resetInitial(self):
        """Reset the environment to its initial state."""
        self.reset()
        self.initialStates()
        self.initialActions()
        self.population = []

    def initialStates(self):
        """Initialize all possible states based on the attribute ranges."""
        maxAttr = [value - 1 for value in self.CONST['rangeATTR']]
        # Generate all combinations of attribute values using itertools.product
        for attr in itertools.product(*(range(maxValue + 1) for maxValue in maxAttr)):
            state = State(list(attr), self.manageSubjectIDs)
            self.states.append(state)

    def listStates(self):
        """Return a list of state representations as strings."""
        return [s.codeStates() for s in self.states]

    def listActions(self):
        """Return a list of action representations as strings."""
        return [a.codeAction() for a in self.actions]

    def getActions(self):
        """Return the list of actions."""
        return self.actions

    def initialActions(self):
        """Initialize actions for each attribute with possible changes."""
        for i in range(len(self.CONST['ATTR'])):
            for change in self.CONST["GA_params"]["PossibleActions"]:  # Possible actions
                act = Action(i, change)
                self.actions.append(act)

    def findInStates(self, state):
        """Find and return a state from the list of states."""
        for st in self.states:
            if st.codeStates() == state.codeStates():
                return st
        return None

    def addPopulationToAllStates(self):
        """Add the current population to the list of all states."""
        self.population = list(filter(None, self.population))  # Remove None
        for state in self.population:
            self.allStates.append(state)

    def getListStates(self):
        """Return the list of all states."""
        return self.allStates

    def nextState(self, currState, action):
        """Return the next state based on the current state and action."""
        attr = currState.getAttributes()
        changeAttr = action.getChangeAttributes()
        newAttr = [attr[i] + changeAttr[i] for i in range(len(attr))]
        
        # Validate new attributes
        if any(n < 0 or n >= self.CONST['rangeATTR'][i]-1 for i, n in enumerate(newAttr)):
            return None
        
        newState = State(newAttr, self.manageSubjectIDs)
        return self.findInStates(newState)

    def getNormalDistribution(self):
        """Generate a normal distribution for reward calculations."""
        xAxis = np.arange(0, self.CONST['MaxSTRESS'] + 1, 1)
        mean = self.CONST['TargetSTRESS']
        sd = self.CONST['MaxSTRESS'] / 2
        y = norm.pdf(xAxis, mean, sd)

        # Normalize the output range to [-1, 1]
        oldMin, oldMax = np.min(y), np.max(y)
        newMax, newMin = 1, -1
        result = (((y - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
        return result
    
    def calculateReward(self, currState):
        """Calculate the reward for the current state."""
        stressLevel = currState.getStress()
        rewards = self.getNormalDistribution()
        return rewards[stressLevel]

    def getNeighbours(self, currState):
        """Get neighboring states for the current state based on possible actions."""
        neighbours = []
        for act in self.getActions():
            st = self.nextState(currState, act)
            if st is not None:
                neighbours.append(st)
        return neighbours

    def addListToPopulation(self, newStates):
        """Add new states to the population if they don't already exist."""
        for state in newStates:
            st = self.findInStates(state)
            if st and all(pop.codeStates() != st.codeStates() for pop in self.population):
                self.population.append(st)

    def sortPopulation(self):
        """Sort the population based on calculated rewards."""
        self.population = list(filter(None, self.population))  # Remove None
        self.population.sort(key=self.calculateReward, reverse=True)

    def selection(self):
        """Select the top states from the population."""
        self.sortPopulation()
        if len(self.population) > self.populationSize:
            self.population = self.population[:self.populationSize]

    def chooseRandomPopulation(self):
        """Randomly sample from the population."""
        if len(self.population) > self.populationSize:
            self.population = random.sample(self.population, self.populationSize)
        self.sortPopulation()

    def mutation(self, probability=0.1):
        """Mutate the population based on a specified mutation probability."""
        numMutations = int(probability * self.populationSize)
        children = []
        for _ in range(numMutations):
            rn = np.random.randint(0, len(self.population))
            mutationState = copy.deepcopy(self.population[rn])
            mutationState.changeOneAttrRandomly()
            children.append(mutationState)
        self.addListToPopulation(children)

    def mutationChildren(self, children, probability=0.1):
        """Mutate a list of children based on a specified probability."""
        for child in children:
            if random.random() < probability:
                child.changeOneAttrRandomly()
        return children

    def chooseParent(self):
        """Select a parent from the population based on reward probabilities."""
        sumReward = sum(self.calculateReward(p) for p in self.population)
        randomNum = random.random() * sumReward
        cumulativeReward = 0
        for num, p in enumerate(self.population):
            cumulativeReward += self.calculateReward(p)
            if cumulativeReward >= randomNum:
                return self.population[num]

    def crossover(self, probability=0.8):
        """Perform crossover between parents to create new children."""
        numCrossovers = int(probability * self.populationSize)
        children = []
        for _ in range(numCrossovers): 
            parent1 = copy.deepcopy(self.chooseParent())
            parent2 = copy.deepcopy(self.chooseParent())

            parent1.changeHalfAttr(parent2.getAttributes())
            parent2.changeHalfAttr(parent1.getAttributes())

            children.append(parent1)
            children.append(parent2)

        mutation_prob = self.CONST["GA_params"]["MutationProbability"]
        children = self.mutationChildren(children, mutation_prob)
        self.addListToPopulation(children)

    def setStressLevelSeen(self, currState):
        """Set the stress level seen for the current state."""
        if currState is not None:
            for st in filter(None, self.allStatesPerson):  # Remove None
                if currState.codeStates() == st.codeStates():
                    currState.stressLevel = st.getStress()
                    return
            currState.getStress()

    def start(self):
        """Initialize the first state randomly."""
        curr = State(None, self.manageSubjectIDs)
        curr.initialRandomState()
        return self.findInStates(curr)

    def runGA(self):
        """Run the genetic algorithm for the defined number of steps."""
        print(self.manageSubjectIDs.SubjectID)
        self.step = 0
        self.resetInitial()
        curr = self.start()
        self.population.append(curr)

        neighbours = self.getNeighbours(curr)
        self.addListToPopulation(neighbours)
        self.chooseRandomPopulation()

        crossover_prob = self.CONST["GA_params"]["CrossoverProbability"]
        while (self.step < self.CONST["Horizon"]) and (not self.population[0].hitGoalStress()) and (not self.population[0].hitMaxStress()):
            self.crossover(crossover_prob)
            self.selection()
            self.step += 1

        if self.population[0].hitGoalStress():
            self.manageSubjectIDs.couldHitGoalStress()

    def runGreedy(self):
        """Run the greedy algorithm to find an optimal state over a defined number of steps."""
        print(self.manageSubjectIDs.SubjectID)
        
        self.step = 0  
        self.resetInitial()  # Reset the environment to its initial state
        curr = self.start()  # Get the starting state

        while (self.step < self.CONST["Horizon"]) and not (curr.hitGoalStress() or curr.hitMaxStress()):
            self.step += 1 
            
            # Get all neighboring states
            neighbours = self.getNeighbours(curr)
            
            # Find the best neighbor that improves the current state
            for neighbour in neighbours:
                if self.calculateReward(neighbour) > self.calculateReward(curr):
                    curr = neighbour  # Update current state to the better neighbor
            
            # Check if the goal stress level is reached
            if curr.hitGoalStress():
                self.manageSubjectIDs.couldHitGoalStress()

    def runRandom(self):
        """Run the random algorithm to explore states over a defined number of steps."""
        # Print the current subject ID for tracking
        print(self.manageSubjectIDs.SubjectID)
        
        self.step = 0  
        self.resetInitial()  # Reset the environment to its initial state
        curr = self.start()  # Get the starting state

        while (self.step < self.CONST["Horizon"]) and not (curr.hitGoalStress() or curr.hitMaxStress()):
            self.step += 1 
            
            # Get all neighboring states
            neighbours = self.getNeighbours(curr)
            
            # Select a random neighbor
            rn = np.random.randint(0, len(neighbours))
            curr = neighbours[rn]  # Update current state to the randomly chosen neighbor
            
            # Check if the goal stress level is reached
            if curr.hitGoalStress():
                self.manageSubjectIDs.couldHitGoalStress()

                
            