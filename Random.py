import logging
from SpiderEnv import SpiderEnv
import gym
import numpy as np
from ManageSubjectIDs import ManageSubjectIDs


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#This one use gym environmnet
def runRandom(full_pass=10):
    """
    Run the random agent in the Spider environment for a specified number of full passes.
    
    Parameters:
    - full_pass (int): Number of iterations (episodes) the random agent will run.
    """
    manage_subject_ids = ManageSubjectIDs()

    for i in range(full_pass):
        logging.info(f"Starting iteration {i+1} of {full_pass} with method Random")

        # Initialize the environment with the subject manager
        env = SpiderEnv(manage_subject_ids)
        #manage_subject_ids.resetSubjectID()
        state = env.reset()  # Start the first episode
        done = False
        pass_before = manage_subject_ids.getPassThroughSubjects()
        
        #while not manageSubjectIDs.lastSubject():
        while manage_subject_ids.getPassThroughSubjects() == pass_before:
            print(manage_subject_ids.getSubjectID())
            # Take a random action from the environment's action space
            action = env.action_space.sample() 
            state, reward, done, info = env.step(action)
            
            if done:
                #resetting environment... go to the next subject
                state = env.reset()  # Reset after episode finishes

    # Log results
    logging.info("Final Results:")
    logging.info(f"Percentage Hitting Goal: {manage_subject_ids.percentage_hit_goal} -- "
                 f"Mean: {manage_subject_ids.getMeanPercentageHitGoal()} -- "
                 f"Std: {manage_subject_ids.getStdPercentageHitGoal()}")
    logging.info(f"Avg Presented Spiders: {manage_subject_ids.avg_presented_spiders} -- "
                 f"Mean: {manage_subject_ids.getMeanAvgPresentedSpiders()} -- "
                 f"Std: {manage_subject_ids.getStdAvgPresentedSpiders()}")
    print(len(manage_subject_ids.percentage_hit_goal))