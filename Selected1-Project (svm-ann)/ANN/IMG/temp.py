#--->importing req libraries
try:
    import tensorflow as tf
    import cv2
    import os
    import pickle
    import numpy as np             
    import pandas as pd 
    import matplotlib.pyplot as plt 
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Flatten,Dense,Activation,LeakyReLU
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import accuracy_score
    from tensorflow.keras import datasets, layers, models
    from sklearn.metrics import confusion_matrix
    from keras.preprocessing.image import ImageDataGenerator
    from keras import regularizers
    from PIL import Image
    import random
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")

class MasterImage(object):

    def __init__(self,PATH='', IMAGE_SIZE = 80):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        print("Found Categories ",self.list_categories,'\n')
        return self.list_categories

    def Process_Image(self):
        try:

            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)         # Folder Path
                class_index = self.CATEGORIES.index(categories)                 # this will get index for classification

                for img in os.listdir(train_folder_path):                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)             # image 
                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE()) # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE),interpolation= cv2.INTER_CUBIC)
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass
            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])        # Get the X_Data
                self.y_data.append(x[1])        # get the label

            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data
            Y_Data = np.asarray(self.y_data)   
            # reshape x_Data
            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
            return X_Data, Y_Data
        except:
            print("Failed to run Function Process Image ")

    def pickle_image(self):

        # Call the Function and Get the Data
        X_Data,Y_Data = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('X_Data','wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('Y_Data', 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return X_Data,Y_Data

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('X_Data','rb') #rb for binary
            X_Data = pickle.load(X_Temp)

            Y_Temp = open('Y_Data','rb')
            Y_Data = pickle.load(Y_Temp)

            print('Reading Dataset from PIckle Object')
            return X_Data,Y_Data
        except:
            print('Could not Found Pickle File ')
            print('Loading File and Dataset  ..........')
            X_Data,Y_Data = self.pickle_image()
            return X_Data,Y_Data


if __name__ == "__main__":
    
    path_t = r'E:\FCAI.HU\Lev.3\Se.1\Selected-1\ANN Project\dataset1\training_set'
    train = MasterImage(PATH=path_t,IMAGE_SIZE=80)
    X_train,Y_train = train.load_dataset()
    
    print(X_train.shape)
     
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
    number_of_train = X_train.shape[0]
    number_of_test = X_test.shape[0]
    
    x_train = X_train.reshape(number_of_train,  X_train.shape[1]*X_train.shape[2]*X_test.shape[3])
    x_test = X_test .reshape(number_of_test,  X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

    ann = models.Sequential([
            layers.Flatten(input_shape=[6400]),
            layers.Dense(500, activation='relu',),
            layers.Dense(240, activation='relu',kernel_regularizer = regularizers.l1(0.00001)),  #----> lamda
            layers.Dense(2, activation='sigmoid')    
        ])

    ann.compile(optimizer='SGD',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    ann.fit(x_train,
            y_train,batch_size=50,
            epochs=50)
    
    ann.optimizer.lr=0.01 
    
    ann_fit = ann.fit(x_train, y_train,validation_split = 0.20,
                              batch_size=40,
                              epochs=50)
    
    pd.DataFrame(ann_fit.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
    predictions = ann.predict(x_test,verbose=0,batch_size=50)
    rounded_predictions = np.argmax(predictions, axis=-1)
    cm = confusion_matrix( y_test,rounded_predictions)
    y_test = np.argmax(predictions, axis=-1)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot 
    normalize=True
    if normalize:
        cm = cm.astype('int') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization') 
    print(cm)
    fpr, tpr, thresholds = roc_curve(y_test, rounded_predictions)
    auc = roc_auc_score(y_test, rounded_predictions)
    print('AUC: %.3f' % auc)    
  
    for i in range(1,10):
            plt.figure()
            if predictions[i, 0] > predictions[i, 1]:
                plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(predictions[i, 0], predictions[i, 1]))
            else:
                plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(predictions[i, 1], predictions[i, 0]))
            plt.imshow(X_test[i])
            plt.show()
   
    
       
    
           
  # def plot_confusion_matrix(cm, classes,
  #                     normalize=False,
  #                     title='Confusion matrix',
  #                     cmap=plt.cm.Blues):
  #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
  #     plt.title(title)
  #     plt.colorbar()
  #     tick_marks = np.arange(len(classes))
  #     plt.xticks(tick_marks, classes, rotation=45)
  #     plt.yticks(tick_marks, classes)  
  #     plot_confusion_matrix(cm=y_test, classes = y_test, title='Confusion Matrix')
       # if normalize:
       #     cm = cm.astype('int') / cm.sum(axis=1)[:, np.newaxis]
       #     print("Normalized confusion matrix")
       # else:
       #     print('Confusion matrix, without normalization') 
       # print(cm)

  # thresh = cm.max() / 2.
  # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  #     plt.text(j, i, cm[i, j],
  #         horizontalalignment="center",
  #         color="white" if cm[i, j] > thresh else "black")

  # plt.tight_layout()
  # plt.ylabel('True label')
  # plt.xlabel('Predicted label')
 
  # cm_plot_labels = ['cat','dog']
  










