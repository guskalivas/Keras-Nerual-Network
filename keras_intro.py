# Name Gus Kalivas\


import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot
from keras import optimizers, layers, losses, metrics

#global list of class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
get_dataset param is training, if true it represents the training images and labels to return, otherwise
returns the test images and test labels 
'''
def get_dataset(training=True):
    #get daya from keras dataset
    fashion_mnist = keras.datasets.fashion_mnist
    #get train images, labels and test images, labels 
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    if training: # if true returns training set
         return train_images, train_labels
    return test_images, test_labels #else returns test set

'''
print stats, prints the total number of images in the given dataset, Image dimension and
the number of images corresponding to each of the class labels
'''
def print_stats(images, labels):
    tot_num = len(images) # total number of images 
    x = len(images[0]) # x dimension of image
    y = len(images[0][0]) #y dimension of image 
    print(tot_num)
    print(x,'x', y)
    # counts the number of images corresponding to each class label 
    for i in range(len(class_names)):
        count = 0
        for j in labels:
            if i == j:
                count+=1
        print(i, class_names[i], count)
   
    return None

'''
 takes a single image as an array of pixels and displays an image
'''
def view_image(image, label):
    # creats the subplot of one row and one col
    f, ax1 = pyplot.subplots(nrows = 1, ncols= 1)
    # sets the title to the label of the image
    ax1.set_title(label)
    pos = ax1.imshow(image, aspect='equal')
    # creates the colorbar
    f.colorbar(pos, ax=ax1, fraction=0.046, pad=0.04)
    #shows the image
    pyplot.show()
    return None


'''
takes no arguments and returns an untrained neural network
'''
def build_model():
    #squential object to hold the layers
    model = keras.Sequential()
    # specifies the input shape of the first input layer
    model.add(keras.layers.InputLayer(input_shape= (28,28)))
    #Flatten layer to convert the 2D pixel array to a 1D array of numbers
    model.add(keras.layers.Flatten())
    #a Dense layer with 128 nodes and a relu activation
    model.add(keras.layers.Dense(128,activation= "relu"))
    #a Dense layer with 10 nodes, that will produce a score for each of the possible class labels
    model.add(keras.layers.Dense(10))
    #compile model with optimizer 'Adam' and metrics 'Accruacy
    model.compile(loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='Adam', metrics = ['accuracy'])
    return model

'''
 takes the model produced by the previous function and the images and labels produced 
 by the first function and trains the data for T epochs
'''
def train_model(model, train_images, train_labels, T):
    #fits the model and supresses some output
    model.fit(x =train_images, y = train_labels, epochs = T ,verbose = 2)
    return None

'''
Takes the trained model produced by the previous function and the test image/labels, 
and prints the accuracy and loss
'''
def evaluate_model(model, images, labels, show_loss=True):
    # gets test loss and accuracy
    test_loss, test_accuracy = model.evaluate(images, labels ,verbose = 0)
    if show_loss: # if true show loss and accuracy
        print('Accuracy: ',round(test_accuracy*100,2) ,"%")
        print('Loss: ',round(test_loss, 2))
    else: #else show just accuracy
        print('Accuracy: ',round(test_accuracy*100,2) ,"%")
    # return rounded accracy as a percent and rounded loss to two decimal places 
    return round(test_accuracy*100, 2), round(test_loss,2)

'''
Takes the trained model and test images, and prints the top 3 most likely labels 
for the image at the given index, along with their probabilities
'''
def predict_label(model, images, index):
    # adds a softmax layer
    probability_model = keras.Sequential([model,keras.layers.Softmax()])
    # gets the probability of all images
    pred = probability_model.predict(images)
    # gets the probabilities for the class labels for this index
    a = np.array(pred[index])*100
    # sorts the index to get top three most likely labels
    res = sorted(range(len(a)), key = lambda sub: a[sub])[-3:] 
    #prints top three labels with probabilities as a percent
    print(class_names[res[-1]],':', round(a[res[-1]],2), '%')
    print(class_names[res[-2]],':', round(a[res[-2]],2), '%' )
    print(class_names[res[-3]], ':', round(a[res[-3]],2), '%' )
    
    return None

if __name__=="__main__":
    trian_images, train_labels = get_dataset(True)
    train = get_dataset(True)
    #print(trian_images, train_labels)

    print_stats(trian_images, train_labels)
    view_image(trian_images[9], class_names[train_labels[9]])
    model = build_model()
    train_model(model, train[0], train[1], 5)
    (test_images, test_labels) = get_dataset(False)
    a, l = evaluate_model(model, test_images, test_labels)
    print(a,l)

    predict_label(model, test_images, 0)





