# Brain-Tumer-Detection-CNN
1. Library Imports and Device Setup
The code initiates by importing essential Python libraries for deep learning and data manipulation, including torch for model construction, torchvision for pre-trained models and data transformations, and numpy for numerical operations. Additional libraries such as matplotlib and seaborn are used for visualizing results. The PIL library is employed for image processing. The device is configured to use a GPU if available, ensuring computational efficiency.

# 2. Data Transformations
Data transformations are defined for augmenting and preprocessing the dataset:

Training Transformations: Includes resizing images to 224x224 pixels, applying random horizontal flips for data augmentation, and converting images to tensor format.
Testing Transformations: Involves resizing images to 224x224 pixels and converting them to tensor format, without augmentation.
# 3. Custom Dataset Class
A custom dataset class BrainTumorDataset is defined, inheriting from torch.utils.data.Dataset. This class is designed to handle the loading and transformation of brain tumor MRI images:

Initialization: Scans the root directory for image files, organizes them by class, and stores their paths and corresponding labels.
Length Method: Returns the total number of images in the dataset.
Item Method: Loads an image from disk, applies the specified transformations, and returns the image along with its label.
# 4. Dataset and DataLoader Creation
Instances of the BrainTumorDataset are created for both training and testing datasets, specifying the root directories and transformations. DataLoader objects are then instantiated for batching, shuffling, and loading data in parallel to enhance training efficiency.

# 5. CNN Model Definition
A custom CNN architecture is implemented in the CNNModel class:

Convolutional Layers: Includes three convolutional layers with increasing depth (16, 32, and 64 filters), applying ReLU activations and max pooling to extract hierarchical features from the images.
Fully Connected Layers: Consists of a fully connected layer with 512 units followed by a final layer with 4 units corresponding to the number of output classes.
Forward Pass: Defines the data flow through the network, including convolutional operations, activation functions, pooling, and final classification.
# 6. Model Visualization
To visualize the model architecture, the torchviz library is used to generate a graphical representation of the network. A dummy input tensor is fed through the model to create a visualization of the computational graph, which is then saved and displayed.

# 7. Model Summary
Using torchinfo, a detailed summary of the model is generated, providing insights into the input and output sizes, number of parameters, kernel sizes, and multiply-accumulate operations.

# 8. Model Initialization and Compilation
The CNN model is instantiated and moved to the designated device (GPU/CPU). The loss function used is cross-entropy, suitable for classification tasks, and the Adam optimizer is chosen for its adaptive learning rate capabilities.

# 9. Model Training
A function train_model is defined to train the CNN model:

Training Loop: Iterates over the specified number of epochs, performs forward and backward passes, updates weights, and calculates both loss and accuracy.
Model Saving: Saves the model state if the current epoch achieves the best accuracy observed so far, ensuring that the best-performing model is retained.
# 10. Model Testing
The test_model function evaluates the model's performance on the test dataset:

Evaluation Mode: Sets the model to evaluation mode, performs inference on the test set, and calculates accuracy.
Prediction Collection: Gathers predicted and true labels for further analysis.
# 11. Performance Evaluation
Classification Report: Prints a detailed classification report including precision, recall, and F1-score for each class.
Confusion Matrix: Generates and plots a confusion matrix to visualize the performance of the model across different classes.
Precision, Recall, and F1-score: Computes and displays precision, recall, and F1-scores, both weighted and macro-averaged.
# 12. Model Evaluation
Finally, the saved model is reloaded, and inference is performed on a subset of randomly selected images from the test set. For each image, the actual and predicted labels are printed, demonstrating the model’s performance on individual examples.

