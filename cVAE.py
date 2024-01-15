import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle


def show_images(images, labels):
    """
    Display a set of images and their labels using matplotlib.
    The first column of `images` should contain the image indices,
    and the second column should contain the flattened image pixels
    reshaped into 28x28 arrays.
    """
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 28, 28)

    # Create a figure with subplots for each image
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3 * len(images))
    )

    # Loop over the images and display them with their labels
    for i in range(len(images)):
        # Display the image and its label
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title("Label: {}".format(labels[i]))

        # Remove the tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Index: {}".format(i))

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the figure
    plt.show()


class ConditionalVAE(tf.keras.Model):
    def __init__(self, num_classes):
        super(ConditionalVAE, self).__init__()
        self.num_hidden = 8

        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(784,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.num_hidden, activation='relu')
        ])

        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(self.num_hidden,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(784, activation='sigmoid')
        ])

        self.mu = layers.Dense(self.num_hidden)
        self.log_var = layers.Dense(self.num_hidden)

        self.label_projector = keras.Sequential([
            layers.Dense(self.num_hidden, activation='relu')
        ])

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y)
        return z + projected_label

    def call(self, x, y):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(self.condition_on_label(z, y))
        return encoded, decoded, mu, log_var

    def sample(self, num_samples, y):
        # Generate random noise
        z = tf.random.normal(shape=(num_samples, self.num_hidden))
        # Pass the noise through the decoder to generate samples
        samples = self.decoder(self.condition_on_label(z, y))
        return np.array(samples)


# Adjust the input shape in the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = tf.reduce_sum(keras.losses.binary_crossentropy(x, recon_x))
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return BCE + KLD


def train_cvae(X_train, y_train, num_classes, num_epochs=10, batch_size=32, learning_rate=1e-3):
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.one_hot(tf.convert_to_tensor(y_train), depth=num_classes)

    model = ConditionalVAE(num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)

        for data, labels in dataset:
            with tf.GradientTape() as tape:
                encoded, decoded, mu, log_var = model(data, labels)
                loss = loss_function(decoded, data, mu, log_var)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy() * len(data)

        epoch_loss = total_loss / len(X_train)
        print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss))

    return model


def one_hot(labels, num_classes):
    return tf.one_hot(labels, num_classes)

def split_data(data, test_size=0.2, val_size=0.1, random_state=42):

    images = data['images']
    filenames = data['filenames']

    # Split the data
    train_images, test_val_images, train_filenames, test_val_filenames = train_test_split(
        images, filenames, test_size=(test_size + val_size), random_state=random_state)

    test_images, val_images, test_filenames, val_filenames = train_test_split(
        test_val_images, test_val_filenames, test_size=(val_size / (test_size + val_size)),
        random_state=random_state)

    train_data = {'images': train_images, 'filenames': train_filenames}
    test_data = {'images': test_images, 'filenames': test_filenames}
    val_data = {'images': val_images, 'filenames': val_filenames}

    return train_data, test_data, val_data

if __name__ == '__main__':
    # Load CelebA attributes
    attr_path = 'C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset\\subset_attributes.csv'
    image_path = 'C:\\Users\\tosic\\tensorflow_datasets\\celeba_dataset\\celeba.pkl'
    labels = pd.read_csv(attr_path, delim_whitespace=True)

    data = pickle.load(open(image_path, "rb"))

    # Split data into train, test, and validation sets
    #train_data, test_data, val_data = split_data(data)
    x_train, x_test, y_train, y_test = train_test_split(data['images'], labels.iloc[:, 1:], test_size=0.2, random_state=42)

    cvae = train_cvae(x_train, y_train, num_classes=10)

    num_samples = 10
    random_labels = [8] * num_samples

    show_images(
        cvae.sample(num_samples, one_hot(tf.convert_to_tensor(random_labels), num_classes=10)),
        labels=random_labels
    )
