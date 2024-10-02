# Import the Library (Including Tensorflow, Numpy)
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

import os
import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR')
print("GPU Usage Status: ",tf.test.is_gpu_available())

#define parameters
BATCH_SIZE = 12 # Batch Size
IMG_SIZE = (224, 224) # Input Image Size
dp_rate = 0.3 # Dropout Layer Rate
base_learning_rate=0.001 # The learning rate of Optimizer
initial_epochs = 10 # The stage 1 for initial training. The epoch is normally 10-20
fine_tune_epochs = 20 # The stage 2 for fine-tune training. The epoch is normally equal or doulbe to the initial epoch.
fine_tune_at = 670 # The layer number to start fine tuning

#Selection Range:"resnet50v2","resnet101v2","resnet152v2","densenet121","densenet169","densenet201","mobilenetv2"
model_str="resnet152v2"

folder = model_str
if os.path.exists(folder)<=0:
	os.makedirs(folder)
model_path='./savemodel/'





def model_selection(ms):
    if ms=="resnet50v2":
        return tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="resnet101v2":
        return tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="resnet152v2":
        return tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="densenet121":
        return tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="densenet169":
        return tf.keras.applications.DenseNet169(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="densenet201":
        return tf.keras.applications.DenseNet201(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    if ms=="mobilenetv2":
        return tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')


#Set the Dataset details and paths
PATH=os.getcwd()

train_dir = os.path.join(PATH, '/home/htihe/datadisk/Data_OLD/NerveClassification/Rearrange/train')
validation_dir = os.path.join(PATH, '/home/htihe/datadisk/Data_OLD/NerveClassification/Rearrange/val')

train_dataset = image_dataset_from_directory(train_dir ,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
											 seed = 1337,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
												  seed = 1337,
                                                  image_size=IMG_SIZE)


												  
print("train dataset: {}" .format(train_dataset))
print("validation dataset: {}" .format(validation_dataset))


print(train_dataset.class_names)
print(validation_dataset.class_names)
class_names = train_dataset.class_names
				
for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

IMG_SHAPE = IMG_SIZE + (3,)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])				


# Detailed Model Setting from the Input Layer to Output Layer

preprocess_input = tf.keras.applications.densenet.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(inputs)
base_model = model_selection(model_str)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(dp_rate)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

e = open(folder+"/model.txt", 'a')
print("summary: {}" .format(base_model.summary()), file=e)
print("Depth of the model is: {}" .format(len(base_model.layers)), file=e)


# compile and train the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



print("summary: {}" .format(model.summary()), file=e)
print("Depth of the model is: {}" .format(len(model.layers)), file=e)
e.close()

history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)


# Plot the Loss value and the Accuracy into the figures
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(folder+"/acc_loss.png")




##############################################Start fine tuning#######################################
	

# Set the finetune layer and start the complie
base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

e = open(folder+"/model.txt", 'a')
print(len(model.trainable_variables))
print("Depth of the model is: {}" .format(len(model.trainable_variables)), file=e)
e.close()

# train the model

total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
	

# Start the plot of loss and accuracy

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(folder+"/acc_loss_after_fine_tuning.png")



	
########################################################SAVE fine tuned weighting#############################
tf.keras.models.save_model(model, filepath=folder+'/model_path') 




























