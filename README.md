# Information 

The goal of this project is to a CNN (convolutional neural network) that can detect different breeds of cats. CNN is a type of Deep Learning neural network specifically used for image classification. The trained model could be useful if you encounter a random cat and you would like to identify its breed. 

Presently, the model is trained on only 9 cat breeds but

The dataset used for this project is acquired by scraping from Google Images. The process involves googling the name of each cat breed and using the chrome extension [Donwload All Images](https://download-all-images.mobilefirst.me/) to download over a 100 pictures for each cats.

# Tools used for this project

Python (Tensorflow, numpy, matplotlib), Donwload All Images, Flask, HTML, Css  

# Exploratory Data Analysis

Cleaning the dataset for images that were downloaded from Google was quite tricky. Currently, Tensorflow can only decode .bmp, .gif, .png, .jpeg image files which means some images might have to be filtered out from the dataset. 

Manual cleaning for the photos was also carried out because some photos were duplicates, did not match the corresponding cat breed or were too low in resolution. Furthermore, some photos contained multiple cats which is not ideal because it might influence the neural network to label a cat breed under the condition that there is more than one cat. Essentially, the idea behind manual cleaning is that the data should also be easy for us humans to classify the cats. 

Here are the few images used for training the model: 

![Sample image of cats](assets\cat_sample.png)

# Designing the convolutional neural network

The following sequential model was used:

```
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9)
])
```

This model optimizes with Adam and calculates loss with sparse categorical crossentropy : 
```
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

# Model Evaluation

Here is the graph for both the loss and accuracy function :

![Loss over epochs](assets\cats9loss.png)

![Accuracy over epochs](assets\cats9acc.png)

The validation loss function follows a similar decreasing trend to the training loss. Same thing can be observed on the accuracy graph as the validation also follows a similar increasing trend to the training accuracy. These metrics confirms that the model does a decent job at detecting cat breeds. 

# Predictions

We can now test out the model by predicting on unknown data. 

Here are the predictions for both Abyssinian and Chartreux cat breeds:

![Abyssian predicted correclty with 75.89% confidence](assets\abys_predict_result.png)

![Abyssian predicted correclty with 59.05% confidence](assets\chtrx_predict_result.png)

It seems our model works quite well!

# Contact Me

Thames Manisy - [My Discord](https://discord.com/channels/Thames#7138) - thamesmanisy@gmail.com

Project Link: [Song Recommender](https://github.com/thames31/lyrics-emotion-detector)



