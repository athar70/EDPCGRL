import random
import numpy as np
import csv

# Constants for generating random attributes
NUM_PERSONS = 30
ATTRIBUTES = ['LocoMotion', 'Motion', 'Closeness', 'Largeness', 'Hairiness', 'Color']
MEANS = [0.9, 0.9, 0.4, 0.7, 0.6, 0.5]
STDS = [0.15, 0.15, 0.17, 0.16, 0.21, 0.20]

def generate_random_from_normal(mu, sigma):
    """Generates a random number from a normal distribution."""
    return np.random.normal(mu, sigma)

def generate_persons(num_persons):
    """Generates a list of persons with random attributes."""
    persons = []
    for _ in range(num_persons):
        attributes = [generate_random_from_normal(MEANS[i], STDS[i]) for i in range(len(ATTRIBUTES))]
        persons.append(attributes)
    return persons

def save_to_csv(file_path='./Data/test.csv', data=None):
    """Saves the provided data to a CSV file."""
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ATTRIBUTES)  # Write the header
        writer.writerows(data)         # Write the data rows

def read_from_csv(file_path='./Data/test.csv'):
    """Reads data from a CSV file and returns it as a list of persons."""
    persons = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read the header row
        for row in reader:
            # Convert each value in the row to a float and append to persons
            persons.append([float(i) for i in row])
    return persons

# Generate persons and save to CSV
if __name__ == "__main__":
    persons_data = generate_persons(NUM_PERSONS)
    save_to_csv(data=persons_data)
