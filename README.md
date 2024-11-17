# MNIST CNN Classifier with CI/CD

![ML Pipeline](https://github.com/Anubis997/ERA_V3_Assignment_5/actions/workflows/ml-pipeline.yml/badge.svg)

This project implements a CNN classifier for the MNIST dataset with automated testing and CI/CD pipeline using GitHub Actions.

## Project Structure
- `train.py`: Contains the model architecture and training code
- `test_model.py`: Contains test cases for model validation
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow file
- `.gitignore`: Specifies which files Git should ignore
- `augmented_samples.png`: Visualization of augmented training samples

## Model Architecture
- 2 Convolutional layers
- 2 Fully connected layers
- Less than 25,000 parameters
- Achieves >95% accuracy in 1 epoch

## Local Setup and Testing

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

