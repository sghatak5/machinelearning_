import os
import numpy as np
from PIL import Image

def load_images(base_folder, image_size):
    data = []
    labels = []
    label_map = {}

    for label_folder in os.listdir(base_folder):
        label_path = os.path.join(base_folder, label_folder) #Assigns full path of each sub_folder/obeject to label_path
        if os.path.isdir(label_path):
            label_index = len(label_map)
            label_map[label_folder] = label_index #Creates a dictionary of the sub_folders with corresponding index
            for image_file in os.listdir(label_path):
                try:
                    image_path = os.path.join(label_path, image_file)
                    with Image.open(image_path) as img:
                        img = img.resize(image_size)
                        img_array = np.array(img) / 255 #Normalize pixel values
                        flattened_img_array = img_array.flatten()  # Flatten the image
                        data.append(flattened_img_array)
                        labels.append(label_index)
                except Exception as e:
                    print(f"Failed to load image {image_file}: {e}")

    return np.array(data), np.array(labels), label_map

def load_test_images(test_folder, image_size):
    data = []
    labels = []
    labels_map = {}
    for image_file in os.listdir(test_folder):
        try:
            image_path = os.path.join(test_folder,image_file)
            #print(image_path)
            with Image.open(image_path) as img:
                img = img.resize(image_size)
                img_array = np.array(img)/255
                flattened_img_array = img_array.flatten()
                data.append(flattened_img_array)
        except Exception as e:
            print(f"Failed to load image {image_file}: {e}")
    return np.array(data)


train_folder = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_train'
validation_folder = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_validation'
#test_folder = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_test'
y = np.array([0, 1, 2, 3, 4, 5])
num_classes = 6
X_train, y_train, X_train_map = load_images(train_folder, image_size=(5, 5))
X_val, y_val, Y_val_map = load_images(validation_folder, image_size=(5, 5))
#print(f"y_val: {y_val.shape}")
#print(f"y_train: {y_train.shape}")
#X_test = load_test_images(test_folder, image_size =(5,5))
#label_map = X_train_map
input = X_train.shape[1]
#input_val = X_val.shape[1]
#input_test = X_test.shape[1]

#Define Layer Class
class DenseLayer:
    def __init__(self, n_input, n_neurons):
        self.output = None
        self.weight = 0.10 * np.random.randn(n_input, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        self.input = None
        self.dweight = None
        self.dbias = None

    def forward(self, input): #Input is the X for Hidden Layer 1 and Output of the Activation Layers for the rest
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias

    def backward(self, dvalue):
        self.dweight = np.dot(self.input.T, dvalue)
        self.dbias = np.sum(dvalue, axis=0, keepdims=True)
        self.dinput = np.dot(dvalue, self.weight.T)


#Relu Activation
class ActivationReLU:
    def __init__(self):
        self.output = None
        self.input = None
        self.dinput = None

    def forward(self, input):  #Input is output of the Dense Layer
        self.input = input
        self.output = np.maximum(0, input) #Output is the final output of the Hidden Layer

    def backward(self, dvalue):
        self.dinput = dvalue.copy()
        self.dinput[self.input <= 0] = 0

#Softmax Activation
class ActivationSoftMax:
    def __init__(self):
        self.output = None
        self.dinput = None

    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalue):
        self.dinput = dvalue.copy()

def OneHotEncoder(y, num_classes):
    y = np.array(y, dtype=int)
    # Create an array of zeros with shape (len(y), num_classes)
    one_hot = np.zeros((len(y), num_classes))

    # Set the appropriate element to one
    one_hot[np.arange(len(y)), y] = 1

    return one_hot

#Loss Function
def CrossEntropyLoss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred))
    dvalue = y_pred - y_true # For cross-entropy loss and softmax, the derivative simplifies to (y_pred - y_true)
    return loss, dvalue

#HyperParameters
LearningRate = 0.0001
Epochs = 300
batch_size = 16

#Define Layer1
Layer1 = DenseLayer(input, 25)
activation1 = ActivationReLU()

#Define Layer2
Layer2 = DenseLayer(25, 15)
activation2 = ActivationReLU()

#Define Layer3
Layer3 = DenseLayer(15, 6)
activation3 = ActivationSoftMax()

best_train_loss = 10
count = 0

#Training
for epoch in range(Epochs):
    # Splitting the training data into batches
    for start_idx in range(0, len(X_train), batch_size):
        end_idx = min(start_idx + batch_size, len(X_train))
        batch_X_train = X_train[start_idx:end_idx]
        batch_y_train = y_train[start_idx:end_idx]
        #print(f"batch_y_train: {batch_y_train}")
        # Forward pass on Training Data
        Layer1.forward(batch_X_train)
        activation1.forward(Layer1.output)
        Layer2.forward(activation1.output)
        activation2.forward(Layer2.output)
        Layer3.forward(activation2.output)
        activation3.forward(Layer3.output)

        output = activation3.output
        #print(f"Output shape: {output.shape}")


        # Convert y to one-hot encoding
        y_true = OneHotEncoder(batch_y_train, num_classes)
        #print(f"y_true Shape: {y_true.shape}")
        # Loss Computation for Training Data
        train_loss, dvalue = CrossEntropyLoss(output, y_true)

        # Backward Propagation
        activation3.backward(dvalue)
        Layer3.backward(activation3.dinput)
        activation2.backward(Layer3.dinput)
        Layer2.backward(activation2.dinput)
        activation1.backward(Layer2.dinput)
        Layer1.backward(activation1.dinput)

        # Updating Weights and Biases
        Layer1.weight -= LearningRate * Layer1.dweight
        Layer1.bias -= LearningRate * Layer1.dbias
        Layer2.weight -= LearningRate * Layer2.dweight
        Layer2.bias -= LearningRate * Layer2.dbias
        Layer3.weight -= LearningRate * Layer3.dweight
        Layer3.bias -= LearningRate * Layer3.dbias

    #Forward pass on Validation Set
    '''for start_idx in range(0, len(X_val), batch_size):
        end_idx = min(start_idx + batch_size, len(X_val))
        batch_X_val = X_val[start_idx:end_idx]
        batch_y_val = y_val[start_idx:end_idx]
        

        print(f"batch_X_val shape: {batch_X_val.shape}")
        print(f"batch_y_val shape: {batch_y_val.shape}")'''

    for start_idx in range(0, len(X_val), batch_size):
        end_idx = min(start_idx + batch_size, len(X_val))
        batch_X_val = X_val[start_idx:end_idx]
        batch_y_val = y_val[start_idx:end_idx]

        Layer1.forward(batch_X_val)
        activation1.forward(Layer1.output)

        Layer2.forward(activation1.output)
        activation2.forward(Layer2.output)

        Layer3.forward(activation2.output)
        activation3.forward(Layer3.output)

        output_val = activation3.output

        #Convert y to one hot encoding
        y_val_true = OneHotEncoder(batch_y_val, num_classes)

        #Loss Computation for Validating Data
        val_loss, _ = CrossEntropyLoss(output_val, y_val_true)

    # Forward pass on test data
    """Layer1.forward(X_test)
    activation1.forward(Layer1.output)

    Layer2.forward(activation1.output)
    activation2.forward(Layer2.output)

    Layer3.forward(activation2.output)
    activation3.forward(Layer3.output)

    test_output = activation3.output

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        Layer1_params = [Layer1.weight, Layer1.bias]
        Layer2_params = [Layer2.weight, Layer2.bias]"""

    #Print Loss after certain iteration iteration
    if (epoch % 10) == 0:
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        #print(f"Best_train_loss: {best_train_loss}")

#predicted_classes = np.argmax(test_output, axis=1)

# Translate numeric predictions to actual class names
#predicted_class_names = [label_map[pred] for pred in predicted_classes]

# Output predictions with class names
#for i, class_name in enumerate(predicted_class_names):
    #print(f"Image {i}: Class {class_name}")

#Testing the data
