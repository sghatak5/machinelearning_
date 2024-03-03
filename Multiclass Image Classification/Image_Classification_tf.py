import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def DataPreProcessing(train_dir, val_dir, test_dir, image_size, batch_size):
    """
    Prepares image data from directories for training, validation, and testing using the Keras ImageDataGenerator.

    This function applies rescaling to the images as a form of normalization and sets up data generators
    for the training, validation, and testing datasets. It uses the flow_from_directory method to load
    images from specified directories.

    Parameters:
    train_dir (str): Directory path containing the training dataset.
    val_dir (str): Directory path containing the validation dataset.
    test_dir (str): Directory path containing the test dataset.
    image_size (tuple): Target size of the images (width, height) to which all images found will be resized.
    batch_size (int): Size of the batches of data (number of images).

    Returns:
    tuple: Returns three ImageDataGenerator objects:
        - train_data: ImageDataGenerator for training data.
        - val_data: ImageDataGenerator for validation data.
        - test_data: ImageDataGenerator for test data.
    """
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        color_mode='rgb')

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        color_mode='rgb')

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode=None,
        color_mode='rgb',
        shuffle=False)

    return train_data, val_data, test_data


def NeuralNetworkArchitecture():
    """
    Constructs and returns a sequential neural network model.

    This function builds a neural network architecture using TensorFlow's Keras API. The model consists
    of a sequence of layers designed for multiclass classification. It starts with a flattening layer
    to convert image data into a 1D array. This is followed by two dense (fully connected) layers with
    ReLU activation, and a final dense layer with softmax activation for classification.

    The network is intended for images with dimensions 28x28x3 (height, width, color channels).

    Returns:
    tf.keras.Model: A compiled TensorFlow Keras sequential model with the defined architecture.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 3)),  # Flatten the images
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes
    ])

    return model


def TrainingErrorRate(model_history):
    """
    Calculates and prints the final training error rate based on the training history of a model.

    This function takes the history object of a trained TensorFlow Keras model and computes the error rate
    from the accuracy metric for the last epoch. The training error rate is the complement of the accuracy,
    representing the proportion of incorrect predictions made by the model on the training dataset.

    Parameters:
    model_history (tf.keras.callbacks.History): A history object returned by the fit method of a TensorFlow Keras model.
        It contains the performance metrics collected during the training process.

    Returns:
    None: This function prints the final training error rate but does not return any value.
    """
    final_train_accuracy = model_history.history['accuracy'][-1]  # For the last epoch
    final_train_error_rate = 1 - final_train_accuracy

    print(f"Final Training Error Rate: {final_train_error_rate:.2f}")


def ValidationErrorRate(model,val_data):
    """
    Evaluates the provided model on a given validation dataset and prints the loss, accuracy, and error rate.

    This function uses the 'evaluate' method of the TensorFlow Keras model to compute its performance metrics on a
    provided validation dataset. It calculates and prints the validation loss, accuracy, and error rate. The error
    rate is computed as the complement of the accuracy, representing the proportion of incorrect predictions.

    Parameters:
    model (tf.keras.Model): A trained TensorFlow Keras model that will be evaluated.
    val_data (tf.keras.preprocessing.image.DirectoryIterator): The validation dataset to evaluate the model on.
        It is typically obtained from an ImageDataGenerator's flow_from_directory method.

    Returns:
    None: The function prints the validation loss, accuracy, and error rate but does not return any values.
    """
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_data)

    # Calculate the error rate
    val_error_rate = 1 - val_accuracy

    print(f"Validation Loss: {val_loss:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Validation Error Rate: {val_error_rate:.2f}")


def ModelPrediction(model, test_data, class_indices):
    """
    Generates predictions for a given dataset using a trained model and maps these predictions to class names.

    This function takes a trained TensorFlow Keras model and a dataset, then generates predictions for each
    sample in the dataset. It then maps these predictions to their corresponding class names based on a provided
    mapping of class indices to class names.

    Parameters:
    model (tf.keras.Model): A trained TensorFlow Keras model to be used for making predictions.
    test_data (tf.keras.preprocessing.image.DirectoryIterator): A dataset containing the data for which predictions
        are to be made. This is typically the output of a Keras ImageDataGenerator's flow_from_directory method.
    class_indices (dict): A dictionary mapping class indices to class names. This is typically obtained from the
        'class_indices' attribute of the Keras DirectoryIterator used for training the model.

    Returns:
    None: This function prints the filename and its predicted class name for each sample in the test dataset but
          does not return any value.
    """
    predictions = model.predict(test_data)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    filenames = test_data.filenames
    class_indices = dict((v, k) for k, v in class_indices.items())

    for i in range(len(filenames)):
        filename = filenames[i]
        predicted_class_index = predicted_classes[i]
        predicted_class_name = class_indices[predicted_class_index]
        print(f"Filename: {filename}, Predicted Class: {predicted_class_name}")


def PlotLossvEpoch(model_history):
    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# ---------------Main-----------------#
train_dir = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_train'
val_dir = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_validation'
test_dir = 'C:/Users/sagni/PycharmProjects/pythonProject/machineleanrning/AIAS/NN_Project1/Intel_Dataset/seg_test'

train_data, val_data, test_data = DataPreProcessing(train_dir, val_dir, test_dir, image_size=(28, 28), batch_size=48)

model = NeuralNetworkArchitecture()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
)

TrainingErrorRate(model_history)
ValidationErrorRate(model, val_data)
ModelPrediction(model, test_data, train_data.class_indices)
PlotLossvEpoch(model_history)
