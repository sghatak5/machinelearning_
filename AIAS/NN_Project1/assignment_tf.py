import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def DataPreProcessing(train_dir, val_dir, test_dir, image_size, batch_size):
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 3)),  # Flatten the images
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes
    ])

    return model


def ModelPrediction(model, test_data, class_indices):
    predictions = model.predict(test_data)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    filenames = test_data.filenames
    class_indices = dict((v, k) for k, v in class_indices.items())

    for i in range(len(filenames)):
        filename = filenames[i]
        predicted_class_index = predicted_classes[i]
        predicted_class_name = class_indices[predicted_class_index]
        print(f"Filename: {filename}, Predicted Class: {predicted_class_name}")


def TrainingErrorRate(model_history):
    final_train_accuracy = model_history.history['accuracy'][-1]  # For the last epoch
    final_train_error_rate = 1 - final_train_accuracy

    print(f"Final Training Error Rate: {final_train_error_rate:.2f}")


def ValidationErrorRate(model):
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_data)

    # Calculate the error rate
    val_error_rate = 1 - val_accuracy

    print(f"Validation Loss: {val_loss:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Validation Error Rate: {val_error_rate:.2f}")


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
ValidationErrorRate(model)
ModelPrediction(model, test_data, train_data.class_indices)
PlotLossvEpoch(model_history)
