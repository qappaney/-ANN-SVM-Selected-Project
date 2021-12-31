#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print('\n ')
print('Getting traing dataset...')

data = pd.read_csv("C:\\Users\Masria For PCs\\Downloads\\Pokemon.csv")

print('Traing data set obtained. \n')


# In[2]:


def type_numbering(string) : 
    number = 0
    if string == 'Normal' :
        number = 1
    elif string == 'Fire' :
        number = 2
    elif string == 'Fighting' :
        number = 3
    elif string == 'Water' :
        number = 4
    elif string == 'Flying' :
        number = 5
    elif string == 'Grass' :
        number = 6
    elif string == 'Poison' :
        number = 7
    elif string == 'Electric' :
        number = 8
    elif string == 'Ground' :
        number = 9
    elif string == 'Psychic' :
        number = 10
    elif string == 'Rock' :
        number = 11
    elif string == 'Ice' :
        number = 12
    elif string == 'Bug' :
        number = 13
    elif string == 'Dragon' :
        number = 14
    elif string == 'Ghost' :
        number = 15
    elif string == 'Dark' :
        number = 16
    elif string == 'Steel' :
        number = 17
    elif string == 'Fairy' :
        number = 18
    else :
        number = 0
    
    return number;


# In[3]:


def ANN_classifier(data, test_size=0.3, batch_size = 10, epochs=10):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,confusion_matrix
    
    print('Splitting data...')
    df = data
    df['Type 1'] = data['Type 1'].apply(type_numbering)
    df['Type 2'] = data['Type 2'].apply(type_numbering)
    X = df.drop('Legendary',axis=1).drop('Name', axis=1)
    y = df['Legendary']
    lenght = len(df.drop('Legendary',axis=1).drop('Name', axis=1).columns)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print('Splitting done. \n')
    
    
    
    print('Initializing classifier...')
    from keras import Sequential
    from keras.layers import Dense
    
    clf = Sequential()
    
    clf.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=lenght))
    
    clf.add(Dense(4, activation='relu', kernel_initializer='random_normal'))   
    
    clf.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

    clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    
    clf.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    print('\n')
    print(clf.summary())
    print('\n')
    history = clf.fit(X_train,y_train, batch_size=batch_size, epochs=epochs)
    
    print('Initialization done. \n')
    
    
    print('Evaluating the classifier...')
    eval_model = clf.evaluate(X_train, y_train)
    
    epoch_nums = range(1,epochs+1)
    training_loss = history.history["loss"]
    train_acc = history.history["accuracy"]
    
    plt.figure(figsize=(13,5))

    plt.subplot(1,2,1)
    plt.plot(epoch_nums, training_loss)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    

    plt.subplot(1,2,2)
    plt.plot(epoch_nums, train_acc)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    
    
    print('Accuray: ', eval_model[1])
    print('Loss: ', eval_model[0])
    print('\n ')
    
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    df_cm = pd.DataFrame(cm, index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])
    plt.figure(figsize = (7,7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.xlabel("Predicted Class", fontsize=18)
    plt.ylabel("True Class", fontsize=18)
    
    
    print('\n ')
    print('Done.')
    
    return clf
    


# In[4]:


ANN_classifier(data, batch_size=10, epochs=100)


# In[ ]:




