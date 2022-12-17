import matplotlib.pyplot as plt

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