# Churn Prediction with Artificial Neural Network (ANN)

This project focuses on predicting customer churn using an Artificial Neural Network (ANN) model. The model is built with TensorFlow and Keras libraries to classify whether a customer will churn (leave) or remain with a company. The model utilizes various activation functions and optimizers to enhance performance and training accuracy.

# Dependencies

To run this project, you will need the following libraries:

TensorFlow: for building and training the neural network.
Keras: used for neural network architecture and training.
NumPy and Pandas: for data manipulation.
Matplotlib (optional): for plotting and visualizations.

#  Model Architecture

The model is constructed using multiple layers, including:

Input Layer: The initial layer that takes input features.
Hidden Layers: Multiple dense layers with different activation functions:
LeakyReLU, ReLU, PReLU, and ELU activation functions help introduce non-linearity and allow the model to learn more complex patterns.
Dropout Layers: Used to prevent overfitting by randomly setting a fraction of input units to 0 during training.
Output Layer: The final layer that produces the prediction output.

# Early Stopping Callback

The model uses the EarlyStopping callback during training to prevent overfitting. It monitors the validation loss and stops training if it does not improve for a set number of epochs. This feature helps save valuable computation time and ensures that the model does not overfit the data.

# Performance

After training, the model achieved an accuracy score of 0.8575, indicating its ability to predict churn with a reasonable level of accuracy. Further improvements can be made by tuning hyperparameters or exploring more advanced techniques.

