import json

def readCONST():
    json_file = open('CONSTANTS.json')
    CONST = json.load(json_file)
    return CONST

def writeCONST(data):
    with open('CONSTANTS.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def updateStartState(new_state):
    CONST = readCONST()
    CONST['StartState'] = new_state
    writeCONST(CONST)

def updateTargetStress(target_stress):
    CONST = readCONST()
    CONST['TargetSTRESS'] = target_stress
    writeCONST(CONST)