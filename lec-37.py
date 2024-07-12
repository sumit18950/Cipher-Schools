from tensorflow.keras.datasets import fashion_mnist

import numpy as np

#load the dataset

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

#display the shape  and data lbls

print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)
print("testing data shape: ", x_test.shape)
print("testing labels shape: ", y_test.shape)



#3 visulising the dataset
import matplotlib.pyplot as plt
def plot_initial_images(images,labels,class_names):
    fig,axes=plt.subplots(1,10,figsize=(20,3))
    for i in range(10):
        ax=axes[i]
        ax.imshow(images[i],cmap='grey')
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.show()
    

#class names
class_names=['T-shirt/top', 'Trouser/pants','Pullover shirt','Dress','Coat','Sandal','Shirt',
'Sneaker','Bag','Ankle boot']

#plot  initial imgs with labels

plot_initial_images(x_train,y_train,class_names)


#data preprocessing

#normalizing the data-> it scales pixels from 0-1

x_train=x_train/255.0
x_test=x_test/255.0

#reshape images print(X_train.shape,X_test.shape)
# X_train=X_train.reshape(60000,28,28,1)
# X_test=X_test.reshape(10000,28,28,1)
# print(X_train.shape,X_test.shape)

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

print(x_train.shape,x_test.shape)


#Extracting images
import cv2
from skimage.feature import hog

def extract_hog_features(images):
    hog_features = []
    for image in images:
        # Extract HOG features
        features = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
        hog_features.append(features)
    return np.array(hog_features)


#extract hog features from train nd testing

x_test_hog = extract_hog_features(x_test)
x_train_hog = extract_hog_features(x_train)

#display 
print(x_test_hog.shape)
print(x_train_hog.shape)

#   training clasifier
from sklearn.svm import SVC

#create svm classifier
classifier=SVC(kernel='linear',random_state=0)
#training
classifier.fit(x_train_hog,y_train)
#accuracy
train_accuracy = classifier.score(x_train_hog,y_train)
print(train_accuracy)



#evaluating model
test_accuracy = classifier.score(x_test_hog,y_test)
print(test_accuracy)


#visualise
# Get predictions on the test set
y_pred = svm.predict(x_test_hog)

# Function to plot images with true and predicted labels
def plot_output_images(images, true_labels, predicted_labels, class_names):

    fig, axes = plt.subplots(1, 10, figsize=(20, 3))

    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Plot some test images along with their true and predicted labels
plot_output_images(x_test[:10], y_test[:10], y_pred[:10], class_names)
