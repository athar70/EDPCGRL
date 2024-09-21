import pandas as pd
import ManageCONST
import numpy as np
import logging 

class ManageSubjectIDs:
    def __init__(self):
        self.SubjectID = -1
        self.CONST = ManageCONST.readCONST()

        # Load subjects' coefficients from CSV
        self.SubjectsCoeff = pd.read_csv(self.CONST["Subjects_data_path"])
        self.numSubjects = len(self.SubjectsCoeff)

        # Initialize arrays to track stress goals and presented spiders
        self.hitGoalStress = np.zeros(self.numSubjects)
        self.presentedSpiders = np.zeros(self.numSubjects)

        # Counter for how many times subjects have been passed through
        self.pass_through_subjects = 0

        # Lists to store percentage of goals hit and average spiders presented
        self.percentage_hit_goal = []
        self.avg_presented_spiders = []

    def nextSubjectID(self):
        self.SubjectID += 1
        # Reset if we exceed the number of subjects
        if self.SubjectID >= self.numSubjects:
            self.resetSubjectID()
            self.pass_through_subjects += 1

    def receiveMaxPassThroughSubjects(self, full_pass=10):
        # Check if the maximum number of passes through subjects has been reached
        return self.pass_through_subjects >= full_pass
    
    def getPassThroughSubjects(self):
        # Return the current number of pass-throughs
        return self.pass_through_subjects
    
    def lastSubject(self):
        # Check if the last subject has been reached
        return self.SubjectID >= self.numSubjects

    def resetSubjectID(self):
        # Reset subject ID and record statistics
        self.SubjectID = -1
        self.percentage_hit_goal.append(self.getPercentageHitGoal())
        self.avg_presented_spiders.append(self.getAvgPresentedSpiders())

        logging.info(f"Iteration {self.pass_through_subjects} results:")
        logging.info(f"Goal Stress: {self.hitGoalStress}")
        logging.info(f"Presented Spiders: {self.presentedSpiders}")

        # Reset arrays for next pass-through
        self.hitGoalStress.fill(0)
        self.presentedSpiders.fill(0)

    def getSubjectCoeff(self):
        # Return coefficients for the current subject
        return self.SubjectsCoeff.iloc[self.SubjectID]

    def getSubjectID(self):
        # Return the current subject ID
        return self.SubjectID

    def couldHitGoalStress(self):
        # Mark the current subject as having hit the goal stress
        self.hitGoalStress[self.SubjectID] = 1

    def addPresentedSpiders(self, _presntedSpiders):
        # Add the number of spiders presented to the current subject
        self.presentedSpiders[self.SubjectID] = _presntedSpiders

    def getPercentageHitGoal(self):
        # Calculate and return the percentage of subjects that hit the goal
        return (np.sum(self.hitGoalStress) / len(self.hitGoalStress)) * 100

    def getAvgPresentedSpiders(self):
        # Calculate and return the average number of spiders presented
        return np.mean(self.presentedSpiders)

    def getMeanAvgPresentedSpiders(self):
        # Return the mean of average presented spiders across all pass-throughs
        return np.mean(self.avg_presented_spiders) if self.avg_presented_spiders else 0

    def getStdAvgPresentedSpiders(self):
        # Return the standard deviation of average presented spiders
        return np.std(self.avg_presented_spiders) if self.avg_presented_spiders else 0

    def getMeanPercentageHitGoal(self):
        # Return the mean of percentage hit goals across all pass-throughs
        return np.mean(self.percentage_hit_goal) if self.percentage_hit_goal else 0

    def getStdPercentageHitGoal(self):
        # Return the standard deviation of percentage hit goals
        return np.std(self.percentage_hit_goal) if self.percentage_hit_goal else 0