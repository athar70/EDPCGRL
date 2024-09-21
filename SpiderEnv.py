import gym
from gym import spaces
import numpy as np
# import cv2
import random
import time
from collections import deque
from scipy.stats import norm
import ManageCONST
from ManageSubjectIDs import ManageSubjectIDs

class SpiderEnv(gym.Env):
    def __init__(self, manageSubjectIDs):
        super(SpiderEnv, self).__init__()
        self.CONST = ManageCONST.readCONST()
        self.reward_range = (0, 1) 
        self.States = {}
        self.PresentNewSpider = 0
        self.ManageSubjectIDs = manageSubjectIDs
        #change one action at a time
        #action 1-12: decrease or increase one attribute
        self.action_space = spaces.Discrete(2*len(self.CONST['ATTR']),) 

        max_values = [value - 1 for value in self.CONST['rangeATTR']]
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.CONST['ATTR'])), high=np.array(max_values), dtype=np.int16)

    def step(self, action):
        valid_actions = self.getValidActions()

        if action in valid_actions: #valid action 
            action = self.actionToArray(action)
            self.next_observation = np.add(action , self.observation)
            
            self.next_observation = self.next_observation.astype(np.int16)
            self.observation = self.next_observation

        self.step_num += 1

        stress = self.getStress()
        self.reward = self.calculateReward(stress)
        

        if self.step_num >= self.CONST["Horizon"] -1:
            print("horizon reached")
            self.done = True
        
        if self.hitGoalStress() or self.hitMaxStress():
            self.done = True

        self.info = {}

        return self.observation, self.reward, self.done, self.info
        
    def reset(self):
        if self.ManageSubjectIDs.getSubjectID() != -1:
            self.ManageSubjectIDs.addPresentedSpiders(self.PresentNewSpider)
        self.ManageSubjectIDs.nextSubjectID()
        
        self.step_num = 0
        self.States = {}
        self.PresentNewSpider = 0
        self.done = False
        self.observation = self.initialState()
        stress = self.getStress()
        self.reward = self.calculateReward(stress)
        return self.observation
       

    def actionToArray(self, action):
        new_action = np.zeros(len(self.CONST['ATTR']))
        #decrease or incease one attribute
        #odd numbers: decrease, even numbers: increase
        new_action[action//2] = 1
        if action%2 == 1:
            new_action *= -1
        return new_action
    
    # return an array of valid actions
    def getValidActions(self, observation=[]):
        if len(observation) == 0:
            observation = self.observation
        available_actions = []

        for action_ in range(self.action_space.n):
            action = self.actionToArray(action_)
            next_observation = np.add(action , observation)
            Min_attr = np.zeros(len(self.CONST['ATTR']))
            Max_attr = [value - 1 for value in self.CONST['rangeATTR']]
            next_observation = np.clip(next_observation, Min_attr , Max_attr)

            can_act = not((observation == next_observation).all())
            if can_act:
                available_actions.append(action_)

        # print("available_actions: ", available_actions)
        return available_actions
    
    def addStates(self):
        obs = str(self.observation)
        self.States[obs] = self.stressLevel

    def getStress(self):
        obs = str(self.observation)
        #if see this state before, we do not need to show it to the user again
        #Assume that the stress from the same spider remain the same
        if obs in self.States:
            self.stressLevel = self.States[obs]
        else:
            self.calculateStressLevel() 
            self.addStates()
        return self.stressLevel

    def calculate_stress(self, coefficients, attributes, observation):
        return sum(coefficients[i] * observation[i] for i in attributes)


    def calculateStressLevel(self):
        self.PresentNewSpider += 1
  
        def calculate_stress(coefficients, observation):
            return sum(coefficients[i] * observation[i] for i in range(len(coefficients)))

        def calculate_max_stress(coefficients, max_attributes):
            return sum(coefficients[i] * max_attributes[i] for i in range(len(coefficients)))

        # Calculate stress values
        coefficients = np.array(self.ManageSubjectIDs.getSubjectCoeff())
        stress = calculate_stress(coefficients, self.observation)

        # Calculate max stress values
        Max_Attr = [value - 1 for value in self.CONST['rangeATTR']]
        max_stress = calculate_max_stress(coefficients, Max_Attr)

        # Calculate stress level
        min_stress = 0
        stress_level = stress / (max_stress - min_stress)
        stress_level = round(stress_level * self.CONST['MaxSTRESS'])

        # Clamp stress level between 0 and 10
        stress_level = max(0, min(stress_level, self.CONST['MaxSTRESS']))
        self.stressLevel = stress_level


    def hitMaxStress(self):
        if (self.getStress() == self.CONST['MaxSTRESS']):
            print("Hit Max Stress")
            return True
        else:
            return False

    def hitGoalStress(self):
        if (self.getStress() == self.CONST['TargetSTRESS']):
            print("Find the Target!")
            self.ManageSubjectIDs.couldHitGoalStress()
            return True
        else:
            return False

    def initialState(self):
        spiderAttributes = np.zeros(len(self.CONST['ATTR']))
        for i, range_val in enumerate(self.CONST['rangeATTR']):
            avg_value = range_val // 2
            min_value = 0
            max_value = range_val - 1 
            if self.CONST["StartState"] == 'AVG':
                spiderAttributes[i] = avg_value 
            elif self.CONST["StartState"]  == 'MAX':
                spiderAttributes[i] = max_value
            elif self.CONST["StartState"]  == 'MIN':
                spiderAttributes[i] = min_value

        spiderAttributes = spiderAttributes.astype(np.int16)
        return spiderAttributes

    def getNormalDistribution(self):
        # normal distribution between 0 and 10 with 1 steps.
        x_axis = np.arange(0, self.CONST['MaxSTRESS'] + 1, 1)
        Mean = self.CONST['TargetSTRESS']
        SD = self.CONST['MaxSTRESS'] / 2
        y = norm.pdf(x_axis,Mean,SD)
        #change output range to -1 and 1
        OldMin, OldMax = np.min(y), np.max(y)
        NewMax, NewMin = 1, -1
        result = (((y - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin) ) + NewMin
        return result

    def calculateReward(self, stress):
        rewards = self.getNormalDistribution()
        reward = rewards[stress]
        return reward
