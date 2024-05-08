# https://www.kaggle.com/code/jackttai/dog-breed-classifier-with-pytorch-using-resnet50/notebook

import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchmetrics
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.metrics import classification_report

comp_df = pd.read_csv('./raw_dog_breed/labels.csv')
test_df = pd.read_csv('./raw_dog_breed/sample_submission.csv')

count = comp_df.breed.value_counts()  # might not need

# replace the breed column into digits and creat a breed-to-index dic
comp_df['label'] = LabelEncoder().fit_transform(comp_df.breed)

dict_df = comp_df[['label', 'breed']].copy()
dict_df.drop_duplicates(inplace=True)
dict_df.set_index('label', drop=True, inplace=True)
index_to_breed = dict_df.to_dict()['breed']

train_directory = './raw_dog_breed/train'
comp_df.id = comp_df.id.apply(lambda x: x + '.jpg')
comp_df.id = comp_df.id.apply(lambda x: train_directory + '/' + x)

comp_df.pop('breed')


class img_dataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        x = Image.open(self.dataframe.iloc[index, 0])
        if self.transform:
            x = self.transform(x)

        if self.test:
            return x
        else:
            y = self.dataframe.iloc[index, 1]
            return x, y

    def __len__(self):
        return self.dataframe.shape[0]


train_transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomRotation(15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transformer = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def print_epoch_result(train_loss, train_acc, val_loss, val_acc):
    print('loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'.format(train_loss,
                                                                                train_acc,
                                                                                val_loss,
                                                                                val_acc))


def train_model(model, cost_function, optimizer, num_epochs=5):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    train_acc_object = torchmetrics.Accuracy(task='multiclass', num_classes=120)
    val_acc_object = torchmetrics.Accuracy(task='multiclass', num_classes=120)

    for epoch in range(num_epochs):
        """
        On epoch start
        """
        print('-' * 15)
        print('Start training {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)

        # Training
        train_sub_losses = []
        model.train()

        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            # update loss sublist
            train_sub_losses.append(loss.item())
            # update accuracy object
            train_acc_object(y_hat.cpu(), y.cpu())

        # Validation
        val_sub_losses = []
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            val_sub_losses.append(loss.item())
            val_acc_object(y_hat.cpu(), y.cpu())

        """
        On epoch end
        """
        # Update the loss list
        train_losses.append(np.mean(train_sub_losses))
        val_losses.append(np.mean(val_sub_losses))

        # Update the accuracy list and reset the metrics object
        train_epoch_acc = train_acc_object.compute()
        val_epoch_acc = val_acc_object.compute()
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)
        train_acc_object.reset()
        val_acc_object.reset()

        # print the result of epoch
        print_epoch_result(np.mean(train_sub_losses), train_epoch_acc, np.mean(val_sub_losses), val_epoch_acc)

    print('Finish Training.')
    return train_losses, train_acc, val_losses, val_acc


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# Parameters for dataset
training_samples = comp_df.shape[0]  # Use small number first to test whether the model is doing well, then change
# back to full dataset
test_size = 0.05
batch_size = 64

# Reduce the number of samples
sample_df = comp_df.sample(training_samples)

# Split the comp_df into training set and validation set
x_train, x_test, y_train, y_test = train_test_split(sample_df, sample_df, test_size=test_size)
# x_train = x_train.astype('float32')
# y_train = y_train.astype('float32')

# Create dataloaders form datasets
train_set = img_dataset(x_train, transform=train_transformer)
val_set = img_dataset(x_test, transform=val_transformer)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

# How many images in training set and val set?
print('Training set: {}, Validation set: {}'.format(x_train.shape[0], x_test.shape[0]))


# Use resnet-50 as a base model
class net(torch.nn.Module):
    def __init__(self, base_model, base_out_features, num_classes):
        super(net, self).__init__()
        self.base_model = base_model
        self.linear1 = torch.nn.Linear(base_out_features, 512)
        self.output = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.base_model(x))
        x = F.relu(self.linear1(x))
        x = self.output(x)
        return x


res = torchvision.models.resnet50(pretrained=True)
for param in res.parameters():
    param.requires_grad = False

model_final = net(base_model=res, base_out_features=res.fc.out_features, num_classes=120)
model_final = model_final.to(device)

# Cost function and optimizer
cost_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([param for param in model_final.parameters() if param.requires_grad], lr=0.0003)

# Learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

# Epoch
EPOCHS = 30
# Start Training
train_losses, train_acc, val_losses, val_acc = train_model(model=model_final,
                                                           cost_function=cost_function,
                                                           optimizer=optimizer,
                                                           num_epochs=EPOCHS)


def plot_result(train_loss, val_loss, train_acc, val_acc):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(train_loss, label='loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_xticks(range(0, EPOCHS + 1))
    ax2.plot(train_acc, label='acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_xticks(range(0, EPOCHS + 1))
    plt.show()


plot_result(train_losses, val_losses, train_acc, val_acc)

# Prepare for test data dataframe
test_df = pd.read_csv('./raw_dog_breed/sample_submission.csv')
test_dir = './raw_dog_breed/test'
test_df = test_df[['id']]
test_df.id = test_df.id.apply(lambda x: x + '.jpg')
test_df.id = test_df.id.apply(lambda x: test_dir + '/' + x)
test_set = img_dataset(test_df, transform=val_transformer, test=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_final.eval()
predictions = torch.tensor([])
print('Start predicting....')
for x in test_loader:
    x = x.to(device)
    y_hat = model_final(x)
    predictions = torch.cat([predictions, y_hat.cpu()])
print('Finish prediction.')

predictions = F.softmax(predictions, dim=1).detach().numpy()

answer_id = pd.read_csv('./raw_dog_breed/sample_submission.csv').id.tolist()
predictions_df = pd.DataFrame(predictions, index=answer_id)
predictions_df.columns = predictions_df.columns.map(index_to_breed)
predictions_df.rename_axis('id', inplace=True)
predictions_df.to_csv('submission.csv')

# BELOW IS THE TRUJILLO/GONZALEZ MODEL

with tf.device('/GPU:0'):
    num_classes = 120
    learning_rate = .01
    decay = 1e-6
    momentum = .9
    epochs = 30
    batch_size = 128

    x_train1, x_test1, y_train1, y_test1 = train_test_split(sample_df, sample_df, test_size=test_size)


    class DataGenerator(Sequence):
        def __init__(self, df, batch_size=32, image_size=(224, 224)):
            super(DataGenerator, self).__init__()
            self.df = df
            self.batch_size = batch_size
            self.image_size = image_size

        def __len__(self):
            return len(self.df) // self.batch_size

        def __getitem__(self, index):
            batch_df = self.df[index * self.batch_size:(index + 1) * self.batch_size]
            x_batch = np.zeros((len(batch_df), *self.image_size, 1))  # Assuming grayscale images
            y_batch = np.zeros((len(batch_df), num_classes))  # Assuming one-hot encoded labels

            for i, (_, row) in enumerate(batch_df.iterrows()):
                file_path = row["id"]
                image = Image.open(file_path)
                image = image.resize(self.image_size)
                image = image.convert('L')
                image = np.array(image) / 255.0

                # Assuming one-hot encoded labels
                label = row["label"]
                y_batch[i, label] = 1
            return tf.convert_to_tensor(x_batch, dtype=tf.float32), tf.convert_to_tensor(y_batch, dtype=tf.float32)


    # Usage
    train_generator = DataGenerator(x_train1, batch_size=batch_size)
    validation_generator = DataGenerator(x_test1, batch_size=batch_size)

    x_test_np = x_test1.values
    y_test_np = y_test1.values
    print(y_test_np)

    NN = Sequential()

    NN.add(MaxPooling2D(pool_size=(2, 2)))
    NN.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation=LeakyReLU()))
    NN.add(MaxPooling2D(pool_size=(3, 3)))
    NN.add(Dropout(.5))
    NN.add(Conv2D(64, kernel_size=(15, 15), padding='same', activation=LeakyReLU()))
    NN.add(MaxPooling2D(pool_size=(3, 3)))
    NN.add(Flatten())
    NN.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    NN.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    fit = NN.fit(x=train_generator, validation_data=validation_generator, epochs=epochs)

    print("[INFO] evaluating network...")

    score = NN.evaluate(validation_generator, batch_size=32)
    print("Accuracy: {:.2f}%".format(score[1] * 100))
    print("Loss: ", score[0])

    predictions = NN.predict(validation_generator)

    # Get the true labels
    true_labels = y_test_np[:, 1].astype(int)
    predicted_labels = np.argmax(predictions, axis=-1)

    report = classification_report(true_labels, predicted_labels)
    print(report)
    print(fit.history)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), fit.history["accuracy"], label="train_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
