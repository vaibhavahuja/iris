import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
#loading the data set csv file using the panda library
dataset = pandas.read_csv(url,names=names)

#checking out the size of data
print(dataset.shape)

#the elements can be seen using head. as shown below
print(dataset.head(20))

#the description of the elements including the min max and count mean etc.
print(dataset.describe())

#we can also group the data set using parameters, here we group them by class
print(dataset.groupby('class').size())

#VISUALISING THE DATA IN FORM OF GRAPHS

#univariate plots, plot of each individual variable
#box and whisper plots are shown below
dataset.plot(kind='box',subplots=True,layout=(2,2))
#using the matplotlib to plot the graphs
plt.show()

#histograms are shown as follows
dataset.hist()
plt.show()

#multivariate, looking at the interactions between the variables

#scatter plot matrix
scatter_matrix(dataset)
plt.show()


#EVALUATING THE 6 ALGOS.

#creating a validation dataset
#splitting the loaded dataset into two, 80% used to train the model, 20% to hold back as validation dataset

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
#what this line does is that it separates the array according to the validation size to make another test array.
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

#BUILDING MODELS
#we evaluate 6 different algorithms
#logistic regression, linear discriminant analysis, k-nearest neighbours, classification and regression trees, gaussian naive bayes, and support vector mechanics
#to study these algorithms properly once
#LR and LDA are linear rest are non linear algorithms.
#the algorithms are build and evaluated accordingly

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name,model in models:
    #each fold is then used once as a validation while the k-1 remoaining folds form the training set
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    #evaluates a score by cross validation
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#analyze which model gives the best score.
#also can create a plot to analyse these 6 algorithms

fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#chose the best model you feel, according to the results. I am getting SVM.
#and make predictions accordingly

svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation,predictions))

