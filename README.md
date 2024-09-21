# Arachnophobia exposure therapy using EDPCGRL

## Description

This project is an implementation of my paper titled **"Arachnophobia exposure therapy using experience-driven procedural content generation via reinforcement learning (EDPCGRL)"** and extends its concepts by providing a command-line interface for managing and executing various optimization algorithms. Users can configure parameters like the initial state and target stress, and choose from several methods, including Random, Greedy, Genetic Algorithms (GA), Q-Learning, and Deep Reinforcement Learning (DQN, PPO, A2C).

### Abstract

Personalized therapy, in which a therapeutic practice is adapted to an individual patient, leads to better health outcomes. Typically, this is accomplished by relying on a therapist's training and intuition along with feedback from a patient. While there exist approaches to automatically adapt therapeutic content to a patient, they rely on hand-authored, pre-defined rules, which may not generalize to all individuals. In this paper, we propose an approach to automatically adapt therapeutic content to patients based on physiological measures. We implement our approach in the context of arachnophobia exposure therapy, and rely on experience-driven procedural content generation via reinforcement learning (EDPCGRL) to generate virtual spiders to match an individual patient. 

In this initial implementation, and due to the ongoing pandemic, we make use of virtual or artificial humans implemented based on prior arachnophobia psychology research. Our EDPCGRL method is able to more quickly adapt to these virtual humans with high accuracy in comparison to existing, search-based EDPCG approaches.

For more details, refer to the paper: [Arachnophobia Exposure Therapy using EDPCGRL Paper](https://ojs.aaai.org/index.php/AIIDE/article/view/18904).


## Features

- **Parameter Configuration**: Update `StartState` and `TargetStress` settings.
- **Multiple Optimization Methods**:
  - Random
  - Greedy
  - Genetic Algorithms (GA)
  - Q-Learning
  - Deep Reinforcement Learning (DQN, PPO, A2C)

## Requirements

- Python 3.x
- Required packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_project.git
   ```
   

2. Navigate to the project directory:

   ```bash
   cd Simulation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script with the following command:

    ```bash
    python main.py --start_state <MIN|AVG|MAX> --target_stress <1-9> --method <Random|Greedy|GA|QLearning|DQN|PPO|A2C>
    ```

## Arguments

- `--start_state`: Set the initial state. Choices: `MIN`, `AVG`, `MAX`. Default is `None`.
- `--target_stress`: Set the target stress, must be an integer between 1 and 9. Default is `None`.
- `--method`: Select the optimization method. Options:
  - `Random` (Random algorithm)
  - `Greedy` (Greedy method)
  - `GA` (Genetic Algorithm)
  - `QLearning` (Tabular QLearning - epsilon greedy or decay epsilon greedy)
  - `DQN` (Deep Q-Network)
  - `PPO` (Proximal Policy Optimization)
  - `A2C` (Advantage Actor-Critic)

## Example

To run the Genetic Algorithm with an average starting state and a target stress of 5, execute:

```bash
python main.py --start_state AVG --target_stress 5 --method GA
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request.

## Contact Information

For questions or feedback, feel free to reach out at [athar1@ualberta.ca](mailto:athar1@ualberta.ca).
