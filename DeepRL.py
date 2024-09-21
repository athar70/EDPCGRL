import logging
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
import torch as th
from SpiderEnv import SpiderEnv  # Custom environment class
import gym
import numpy as np
from ManageSubjectIDs import ManageSubjectIDs  # Custom class for managing subject IDs


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(method, env):
    """
    Create and return a model based on the specified method.
    
    Parameters:
    - method (str): The name of the Deep RL algorithm ('PPO', 'A2C', 'DQN').
    - env (gym.Env): The environment for the model to interact with.
    
    Returns:
    - model: The initialized model for the specified RL algorithm.
    """

    if method == 'PPO':
        # PPO with an MLP Policy and a small number of n_steps for faster updates
        return PPO("MlpPolicy", env, verbose=0, n_steps=2)
    elif method == 'A2C':
        # Advantage Actor-Critic (A2C) with MLP Policy
        return A2C("MlpPolicy", env, verbose=0)
    elif method == 'DQN':
        # DQN with MLP Policy
        return DQN("MlpPolicy", env)
    else:
        raise ValueError("Unsupported method. Choose from 'PPO', 'A2C', or 'DQN'.")


def runDeepRL(method='DQN', full_pass=10):
    """
    Run the Deep Reinforcement Learning algorithm with the specified method.
    
    Parameters:
    - method (str): The RL algorithm to use ('DQN' by default).
    - full_pass (int): Number of full iterations or passes through the process.
    
    Logs performance metrics for the experiment.
    """
    manage_subject_ids = ManageSubjectIDs()

    for i in range(full_pass):
        logging.info(f"Starting iteration {i+1} of {full_pass} with method {method}")

        # Initialize the custom environment with the subject manager
        env = SpiderEnv(manage_subject_ids)
        #manage_subject_ids.resetSubjectID()
        state = env.reset()  # Start the first episode
        done = False
        pass_before = manage_subject_ids.getPassThroughSubjects()

        # Create the model based on the chosen method
        try:
            model = create_model(method, env)
        except ValueError as e:
            logging.error(e)
            return
        
        # Run the learning process until the subject pass-through state changes
        while manage_subject_ids.getPassThroughSubjects() == pass_before:
            model.learn(total_timesteps=300)
        

    
    # Log final results after the learning process
    logging.info("Final Results:")
    logging.info(f"Percentage Hitting Goal: {manage_subject_ids.percentage_hit_goal} -- "
                 f"Mean: {manage_subject_ids.getMeanPercentageHitGoal()} -- "
                 f"Std: {manage_subject_ids.getStdPercentageHitGoal()}")
    logging.info(f"Avg Presented Spiders: {manage_subject_ids.avg_presented_spiders} -- "
                 f"Mean: {manage_subject_ids.getMeanAvgPresentedSpiders()} -- "
                 f"Std: {manage_subject_ids.getStdAvgPresentedSpiders()}")
    print(len(manage_subject_ids.percentage_hit_goal))
