import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import matplotlib.pyplot as plt

from MLP import MultilayerPerceptron

from matplotlib import pyplot as plt
import seaborn as sb
from sklearn import datasets
from tqdm import tqdm
from accuracy_score import accuracy_score
from data_manipulations import normalize, to_categorical, train_test_split

def load_data(dir_name):
    """
    Function for loading MNIST data stored in comma delimited files. Labels for 
    each image are the first entry in each row.

    Parameters
    ----------
    dit_name : str
         Path to where data is contained

    Returns
    -------
    X : array_like
        A (N x p=784) matrix of samples 
    Y : array_like
        A (N x 1) matrix of labels for each sample
    """
    data = list() # init a list called `data`
    
    with open(dir_name,"r") as f: # open the directory as a read ("r"), call it `f`
        for line in f: # iterate through each `line` in `f`
            split_line = np.array(line.split(',')) # split lines by `,` - cast the resultant list into an numpy array
            split_line = split_line.astype(np.float32) # make the numpy array of str into floats
            data.append(split_line) # collect the sample into the `data` list
            
    data = np.asarray(data) # convert the `data` list into a numpy array for easier indexing
    
    # as the first number in each sample is the label (0-9), extract that from the rest and return both (X,Y)
    return data[:,1:],data[:,0]

def main():
    y_train = pd.read_csv("data/target.csv")
    x_train = pd.read_csv("data/train.csv")
    x_test = pd.read_csv("data/test.csv")
    
    print(x_train.head(10))
    print(x_train.shape,x_test.shape,y_train.shape)
    
    temp_df = y_train.groupby('radiant_won').count()
    temp_df.reset_index(inplace=True)
    temp_df.iloc[0,0]='False'
    temp_df.iloc[1,0]='True'
    sb.barplot(data = temp_df, x='radiant_won', y='fight_id')
    plt.title('Number of figth for each class')
    plt.show()
    
    
  
    
    print(x_train.shape)
    # print(x_train.corr())
    # plt.figure(figsize=(20,20))
    # sb.heatmap(x_train.iloc[:,1:x_train.shape[1]-30].corr(), annot=True)
    # plt.show()
    
    missing_values_count = x_train.isna().sum()
    print(f"{missing_values_count[0:50]} {missing_values_count[0:50]/x_train.shape[0]*100}" )
    print(f"{missing_values_count[50:92]} {missing_values_count[50:92]/x_train.shape[0]*100}" )
    
    # how many total missing values do we have?
    total_cells = np.product(x_train.shape)
    total_missing = missing_values_count.sum()

    # percent of data that is missing
    percent_missing = (total_missing/total_cells) * 100
    print(percent_missing)
    
    x_train.drop(['first_blood_time', 'first_blood_team', 'first_blood_player1','first_blood_player2',
                  'radiant_bottle_time','radiant_courier_time','dire_bottle_time','dire_courier_time','fight_id'], axis=1, inplace=True)
    
    print(x_train.shape)
    
    for i in x_train.columns[x_train.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        x_train[i].fillna(x_train[i].mean(),inplace=True)
    print(x_train.isna().sum())
    
    
    
    # data = datasets.load_digits()
    # X = normalize(data.data)
    
    # print(X)
    # y = data.target

    # print(y.shape)
    # # Convert the nominal y values to binary
    # y = to_categorical(y)
    
    # print(type(y))
    
    # print(y.shape)
    # print(y)

    X = normalize(x_train.values)
    # print(X)
    # print(y_train['radiant_won'].shape)
    y =  to_categorical(y_train['radiant_won'].astype(int)) 
    # print(type(y))
    # # print(y)
    # # print(y.shape)
    
    # print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, seed=1)

    # MLP
    clf = MultilayerPerceptron(n_hidden=64,
        n_iterations=1000,
        learning_rate=0.00001)

    clf.fit(X_train, y_train)
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=np.unique(y))
    
    

if __name__ == "__main__":
    main()