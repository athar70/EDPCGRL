import ManageCONST
from ManageSubjectIDs import ManageSubjectIDs
import logging
from Environment import Environment

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def runGA(fullPass=10):
    """
    Run the genetic algorithm in the my environment for a specified number of full passes.
    
    Parameters:
    - fullPass (int): Number of iterations (episodes) the genetic method will run.
    """
    manageSubjectIDs = ManageSubjectIDs()

    for i in range(fullPass):
        logging.info(f"Starting iteration {i + 1} of {fullPass} with method Genetic Algorithm")

        pass_before = manageSubjectIDs.getPassThroughSubjects()
        while manageSubjectIDs.getPassThroughSubjects() == pass_before:
            manageSubjectIDs.nextSubjectID()
            env = Environment(manageSubjectIDs)  # Initialize the environment
            env.reset()  # Start the first episode
            env.runGA()

    # Log results
    logging.info("Final Results:")
    logging.info(f"Percentage Hitting Goal: {manageSubjectIDs.percentage_hit_goal} -- "
                 f"Mean: {manageSubjectIDs.getMeanPercentageHitGoal()} -- "
                 f"Std: {manageSubjectIDs.getStdPercentageHitGoal()}")
    logging.info(f"Avg Presented Spiders: {manageSubjectIDs.avg_presented_spiders} -- "
                 f"Mean: {manageSubjectIDs.getMeanAvgPresentedSpiders()} -- "
                 f"Std: {manageSubjectIDs.getStdAvgPresentedSpiders()}")
    print(len(manageSubjectIDs.percentage_hit_goal))

#runGA(10)
