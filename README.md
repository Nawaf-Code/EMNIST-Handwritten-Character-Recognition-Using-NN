# MNIST Handwritten Digits Classifier

This project is a part of the **Deep Learning Nanodegree** from **Udacity**. The goal is to train a deep learning model to classify handwritten digits from the MNIST dataset using a neural network.

## Project Overview

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. In this project, a neural network is trained to recognize and classify these digits.

## Dependencies

To run this project, you need the following libraries:

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow / PyTorch (depending on implementation)
- Jupyter Notebook

You can install the dependencies using:

```bash
pip install numpy matplotlib tensorflow jupyter
```

or for PyTorch:

```bash
pip install numpy matplotlib torch torchvision jupyter
```

## Running the Notebook

1. Open the Jupyter Notebook environment:
   ```bash
   jupyter notebook
   ```
2. Load the `MNIST_Handwritten_Digits-STARTER.ipynb` file.
3. Run all cells to train and evaluate the model.

## Dataset

The MNIST dataset is automatically downloaded via TensorFlow/Keras or PyTorch's torchvision. No manual download is required.

## Results

After training, the model should achieve an accuracy of around **98%** on the test set.

## Future Improvements

- Experiment with different architectures (CNNs, deeper networks).
- Use data augmentation to improve generalization.
- Fine-tune hyperparameters for better performance.

---

This project is a great starting point for understanding deep learning and neural networks in image classification.
