import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plotsave_accuracy(history, fname):
    '''
    Takes in history object from tensorflow model
    and a file name
    
    Plotting the accuracy of the model over epochs
    and saves the figure in the assets folder
    '''
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('assets/' + fname, transparent = False)
    
    
def plotsave_loss(history, fname):
    '''
    Takes in history object from tensorflow model
    and a file name
    
    Plotting the loss of the model over epochs
    and saves the figure in the assets folder
    '''
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('assets/' + fname, transparent = False)

def predict_breed(img_dir, model, IMG_HEIGHT, IMG_WIDTH, class_names):
    '''
    Converts and predicts and image file with the 
    corresponding model

    '''
    img = tf.keras.utils.load_img(img_dir, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(f"This cat breed is : {class_names[np.argmax(score)]} with {np.max(score) * 100:.2f} % confidence")