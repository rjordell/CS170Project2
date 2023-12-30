# Feature Selection and Nearest Neighbor Classification

## Overview
This repository contains a Python script, `main.py`, along with two test datasets: `large-test-dataset-1.txt` and `small-test-dataset.txt`. The script implements feature selection algorithms, specifically Forward Selection and Backward Elimination, in the context of nearest neighbor classification.

## Contents
- `main.py`: The main Python script containing the implementation of feature selection algorithms and nearest neighbor classification.
- `large-test-dataset-1.txt`: A large test dataset for experimentation.
- `small-test-dataset.txt`: A small test dataset for quick testing.

## Dependencies
Make sure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`

You can install them using the following command:
```bash
pip install numpy pandas scikit-learn
```

## How to Run
1. Clone this repository to your local machine.
2. Open a terminal and navigate to the project directory.
3. Run the script by executing the following command:
```bash
python main.py
```
4. Follow the on-screen instructions to choose the dataset and the feature selection algorithm.

## Feature Selection Algorithms
- **Forward Selection:** Iteratively adds features to the set, evaluating the accuracy at each step to find the best feature subset.
- **Backward Elimination:** Iteratively removes features from the set, evaluating the accuracy at each step to find the best feature subset.

## Dataset Format
- The datasets should be in a tabular format where each row represents an instance, and the first column contains the class labels.
- The script will prompt you to enter the name of the dataset file.

## Note
- Ensure that you have the necessary permissions to execute the script.
- The script relies on standard machine learning libraries, and you might need an internet connection for the first run to download scikit-learn models.

Feel free to experiment with different datasets and explore how feature selection impacts the accuracy of the nearest neighbor classifier.

