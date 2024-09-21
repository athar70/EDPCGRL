import argparse
import ManageCONST 
from Random import runRandom
from Genetic import runGA
from Greedy import runGreedy
from QLearning import runQLearning
from DeepRL import runDeepRL

def validate_target_stress(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"TargetSTRESS must be an integer, got '{value}'")
    if not (0 < ivalue < 10):
        raise argparse.ArgumentTypeError(f"TargetSTRESS must be between 0 and 10 [1-9], got '{ivalue}'")
    return ivalue


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Update the StartState and TargetStress in CONSTANTS.json")
    parser.add_argument('--start_state', type=str, 
                        choices=['MIN', 'AVG', 'MAX'], default=None, help="The StartState value to set --- it can be 'MIN', 'MAX', 'AVG'")
    parser.add_argument('--target_stress', type=validate_target_stress,
                        default=None, help="The new StartState value to set -- it should be a value between (0-10)")
    parser.add_argument('--method', type=str, 
                        choices=['Random', 'Greedy', 'GA', 'QLearning', 'DQN', 'PPO', 'A2C'],
                        default='Random', 
                        help="The Method to run --- it can be 'Random', 'Greedy', 'GA', 'QLearning', or deep RL methods: 'DQN', 'PPO', 'A2C'")
    
    Full_pass_over_all_subjects = 10
    # Parse arguments
    args = parser.parse_args()

    # Read current constants
    CONST = ManageCONST.readCONST()
    # Update StartState if provided
    if args.start_state is not None:
        ManageCONST.updateStartState(args.start_state)
        print(f"StartState has been updated to: {args.start_state}")
    else:
        print(f"No new StartState provided. Current StartState is: {CONST['StartState']}")

    # Update TargetSTRESS if provided
    if args.target_stress is not None:
        ManageCONST.updateTargetStress(args.target_stress)
        print(f"TargetSTRESS has been updated to: {args.target_stress}")
    else:
        print(f"No new TargetSTRESS provided. Current TargetSTRESS is: {CONST['TargetSTRESS']}")

    if args.method == 'Random':
        runRandom(Full_pass_over_all_subjects)

    if args.method == 'Greedy':
        runGreedy(Full_pass_over_all_subjects)
    
    if args.method == 'GA':
        runGA(Full_pass_over_all_subjects)
    
    if args.method == 'QLearning':
        runQLearning(Full_pass_over_all_subjects)

    ##Added deep RL methods
    elif args.method == 'DQN':
        runDeepRL('DQN', Full_pass_over_all_subjects)
    elif args.method == 'PPO':
        runDeepRL('PPO', Full_pass_over_all_subjects)
    elif args.method == 'A2C':
        runDeepRL('A2C', Full_pass_over_all_subjects)

if __name__ == "__main__":
    main()