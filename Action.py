import random
import ManageCONST

class Action:
    def __init__(self, numAttr=0, change=0):
        """Initialize the Action with the specified attribute index and change value."""
        self.CONST = ManageCONST.readCONST()
        self.changeAttribute = [0] * len(self.CONST['ATTR'])
        self.changeAttribute[numAttr] += change

    def initialRandomAction(self):
        """Generate a random action by randomly selecting an attribute and changing it by +1 or -1."""
        loc = random.randint(0, len(self.CONST['ATTR']) - 1)  # Random index for attribute
        val = random.choice(self.CONST["GA_params"]["PossibleActions"])  # Randomly choose to decrease (-1) or increase (+1)
        self.changeAttribute[loc] = val

    def codeAction(self):
        """Convert the action attributes to a comma-separated string."""
        return ','.join(map(str, self.changeAttribute))  # Use join for cleaner string conversion

    def decodeAction(self, strActionName):
        """Set the action attributes based on a comma-separated string input."""
        actionValues = strActionName.split(',')
        self.changeAttribute = [int(a) for a in actionValues]  # Convert string values to integers

    def getChangeAttributes(self):
        """Return the list of change attributes."""
        return self.changeAttribute
