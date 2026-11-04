import pandas as pd
from sklearn import tree

data = pd.read_csv('titanic-test.csv')
data2 = pd.read_csv('titanic-train.csv')
data.info()
data2.info()


data = pd.get_dummies(data, columns=['Gender', 'Embarked'])
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])

data2 = pd.get_dummies(data2, columns=['Gender', 'Embarked'])
data2 = data2.drop(columns=['Name', 'Ticket', 'Cabin'])

print(data.dtypes)
print(data2.dtypes)


data['Age'] = data['Age'].fillna(data['Age'].mean())
data2['Age'] = data2['Age'].fillna(data2['Age'].mean())
print(data['Age'].isna().sum())
print(data2['Age'].isna().sum())


y_target = data['Survived'].values
columns = ['Fare', 'Pclass', 'Gender_female', 'Age', 'SibSp']
X_input = data[columns].values

y_target2 = data2['Survived'].values
columns2 = ['Fare', 'Pclass', 'Gender_female', 'Age', 'SibSp']
X_input2 = data2[columns2].values

clf_train = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf_train = clf_train.fit(X_input, y_target)

clf_train2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf_train2 = clf_train2.fit(X_input2, y_target2)

print('Pontosság:', clf_train.score(X_input, y_target) * 100, '%')
print('Pontosság2:', clf_train2.score(X_input2, y_target2) * 100, '%')


from sklearn.tree import export_graphviz  
from IPython.display import Image  
import graphviz


dot_data = export_graphviz(clf_train, out_file=None, 
                      feature_names=columns,
                      class_names=["Died", "Survived"],
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("decision_tree", format="png", cleanup=True)



dot_data = export_graphviz(clf_train2, out_file=None, 
                      feature_names=columns2,
                      class_names=["Died", "Survived"],
                      filled=True, rounded=True,  
                      special_characters=True)  
graph2 = graphviz.Source(dot_data)  
graph2.render("decision_tree2", format="png", cleanup=True)

