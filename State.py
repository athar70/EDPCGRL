import numpy as np
import ManageCONST

class State:
    def __init__(self, spiderAttributes, manage_subject_ids):
        """Initialize the State with given spider attributes and read constants."""
        self.CONST = ManageCONST.readCONST()
        self.manageSubjectIDs = manage_subject_ids
        self.spiderAttributes = spiderAttributes
        self.resetStressLevel()

    def initialRandomState(self):
        """Generate initial random state based on specified start conditions."""
        spiderAttributes = np.zeros(len(self.CONST['ATTR']), dtype=np.int16)

        for i, rangeVal in enumerate(self.CONST['rangeATTR']):
            avgValue = rangeVal // 2
            minValue = 0
            maxValue = rangeVal - 1
            
            if self.CONST["StartState"] == 'AVG':
                spiderAttributes[i] = avgValue
            elif self.CONST["StartState"] == 'MAX':
                spiderAttributes[i] = maxValue
            elif self.CONST["StartState"] == 'MIN':
                spiderAttributes[i] = minValue

        self.spiderAttributes = spiderAttributes
        self.resetStressLevel()
        return spiderAttributes

    def codeStates(self):
        """Convert the state attributes to a comma-separated string."""
        return ','.join(map(str, self.spiderAttributes))

    def getStress(self):
        """Retrieve the current stress level, calculating if necessary."""
        if self.stressLevel == -1:
            self.calculateStressLevel()
        return self.stressLevel

    def getAttributes(self):
        """Return the current spider attributes."""
        return self.spiderAttributes

    def resetStressLevel(self):
        """Reset the stress level to an initial state."""
        self.stressLevel = -1

    def setStressLevel(self, stress):
        """Set the stress level to a specified value."""
        self.stressLevel = stress

    def changeOneAttrRandomly(self):
        """Randomly change one attribute of the spider."""
        index = np.random.randint(0, len(self.spiderAttributes))
        maxAttr = [a + 1 for a in self.CONST['rangeATTR']]
        newValue = np.random.randint(0, maxAttr[index])
        self.spiderAttributes[index] = newValue
        self.resetStressLevel()

    def changeHalfAttr(self, newValsAttr):
        """Change half of the spider attributes to new values."""
        halfLen = len(self.CONST['ATTR']) // 2
        self.spiderAttributes[:halfLen] = newValsAttr[:halfLen]
        self.resetStressLevel()

    def decodeStates(self, strStateName):
        """Extract attribute values from a comma-separated string."""
        attributes = list(map(int, strStateName.split(',')))
        self.spiderAttributes = attributes

    def calculateStressLevel(self):
        """Calculate the current stress level based on spider attributes."""
        self.manageSubjectIDs.presentedSpiders[self.manageSubjectIDs.SubjectID] += 1
        coefficients = np.array(self.manageSubjectIDs.getSubjectCoeff())
        
        # Calculate current stress based on attributes
        stress = np.dot(coefficients, self.spiderAttributes)

        # Calculate maximum possible stress
        maxAttr = [value - 1 for value in self.CONST['rangeATTR']]
        maxStress = np.dot(coefficients, maxAttr)

        # Calculate stress level as a proportion
        stressLevel = stress / max(maxStress, 1)  # Prevent division by zero
        stressLevel = round(stressLevel * self.CONST['MaxSTRESS'])

        # Clamp stress level between 0 and MaxSTRESS
        self.stressLevel = max(0, min(stressLevel, self.CONST['MaxSTRESS']))

    def hitMaxStress(self):
        """Check if the current stress level has reached the maximum."""
        if self.getStress() == self.CONST['MaxSTRESS']:
            print("Hit Max Stress")
            return True
        return False

    def hitGoalStress(self):
        """Check if the current stress level has reached the target."""
        if self.getStress() == self.CONST['TargetSTRESS']:
            print("Find the Target!")
            return True
        return False
