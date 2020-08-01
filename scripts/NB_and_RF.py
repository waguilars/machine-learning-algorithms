import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df=pd.read_csv('./data/Wine.csv',header=None)
df.head(2)
df.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]

df.head(2)
df.isnull().sum()

corr = df[df.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)

X= df.drop(['name','ash'], axis=1)

X.head()

Y=df.iloc[:,:1]
Y.head(2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# print(X_train.shape)
# print(X_test.shape)

models = []

models.append(("Naive Bayes:",GaussianNB()))
models.append(("Random Forest:",RandomForestClassifier(n_estimators=7)))

# print('Models appended...')

results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_result = cross_val_score(model,X_train,Y_train.values.ravel(), cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)