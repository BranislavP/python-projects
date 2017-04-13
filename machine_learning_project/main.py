import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import boxcox
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_columns', 50)


def normalization(data, shift, scale):
    return (np.array(data) - float(shift))/scale

train_data = pd.read_csv("train.csv", sep='\t')
train_data["class"] = pd.read_csv("train_target.csv", names = ["class"])

name_list = ['cli_pl_body', 'cli_cont_len', 'aggregated_sessions', 'net_samples', 'tcp_frag', 'tcp_ooo',
             'cli_tcp_ooo', 'cli_tcp_frag', 'cli_win_zero', 'cli_tcp_full', 'cli_pl_change', 'srv_tcp_ooo',
             'srv_tcp_frag', 'srv_win_zero', 'cli_tx_time', 'proxy', 'sp_healthscore', 'sp_req_duration',
             'sp_is_lat', 'sp_error']
transformed_data = pd.DataFrame()
box_param = pd.DataFrame(data=None, columns=train_data.columns,index=range(0,1))
for word in list(train_data.columns.values):
    if word == 'class':
        transformed_data['class'] = train_data['class']
    elif word not in name_list:
        transformed_data[word], box_param[word][0] = boxcox(train_data[word] + 1)

del train_data
percentile = pd.DataFrame(data=None, columns=transformed_data.columns,index=range(0,2))
for word in list(transformed_data.columns.values):
    if word not in name_list and word != 'class':
        percentile[word][0] = np.percentile(transformed_data[word], 25)
        percentile[word][1] = np.percentile(transformed_data[word], 75)
        transformed_data[word] = normalization(transformed_data[word], percentile[word][0], percentile[word][1])

for word in list(transformed_data.columns.values):
    if word != 'class':
        mean = np.mean(transformed_data[word])
        std = np.std(transformed_data[word])
        transformed_data = transformed_data[transformed_data[word] < mean + 3*std]
        transformed_data = transformed_data[transformed_data[word] > mean - 3*std]

array = []
for i in np.random.rand(len(transformed_data)):
    array.append(i > 0.9)

undersampled_data = transformed_data[(transformed_data['class'] != 0) | (array)]
del transformed_data

X, y = undersampled_data.drop("class", 1), undersampled_data["class"]
del undersampled_data

from sklearn.externals import joblib
import os.path as pth

if pth.exists('forest.pk1'):
    forest = joblib.load('forest.pk1')
    print('forest')
else:
    clf2 = RandomForestClassifier(n_estimators=50, max_depth=7, n_jobs=1, max_features=0.5)
    forest = clf2.fit(X, y)
    joblib.dump(forest, 'forest.pk1')

print('fitted forest')


if pth.exists('bayes.pk1'):
    bayes = joblib.load('bayes.pk1')
    print('bayes')
else:
    clf3 = GaussianNB()
    bayes = clf3.fit(X, y)
    joblib.dump(bayes, 'bayes.pk1')

print('fitted bayes')

if pth.exists('neighbours.pk1'):
    neigh = joblib.load('neighbours.pk1')
    print('knn')
else:
    clf4 = KNeighborsClassifier(5)
    neigh = clf4.fit(X, y)
    joblib.dump(neigh, 'neighbours.pk1')


print('fitted knn')

if pth.exists('svm.pk1'):
    svm = joblib.load('svm.pk1')
    print('svm')
else:
    svm = svm.SVC().fit(X, y)
    joblib.dump(svm, 'svm.pk1')


print('fitted svm')

if pth.exists('tree.pk1'):
    tree = joblib.load('tree.pk1')
    print('tree')
else:
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42).fit(X, y)
    joblib.dump(tree, 'tree.pk1')


print('fitted tree')


valid_data = pd.read_csv("test.csv", sep='\t')
trans_valid = pd.DataFrame()
for word in list(valid_data.columns.values):
    if word not in name_list:
        trans_valid[word] = boxcox(valid_data[word] + 1, lmbda=box_param[word][0])

del valid_data

for word in list(trans_valid.columns.values):
    if word not in name_list:
        trans_valid[word] = normalization(trans_valid[word], percentile[word][0], percentile[word][1])

#valid_test = np.genfromtxt("valid_target.csv", delimiter="\n")

print('valid data in memory')


out_valid = forest.set_params(n_jobs=1).predict(trans_valid)
print('forest: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('forest.csv', sep='\n', format="%d")



out_valid = bayes.predict(trans_valid)
print('bayes: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('bayes.csv', sep='\n', format="%d")

out_valid = neigh.predict(trans_valid)
print('knn: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('neighbour.csv', sep='\n', format="%d")

out_valid = svm.predict(trans_valid)
print('svm: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('svm.csv', sep='\n', format="%d")

out_valid = tree.predict(trans_valid)
print('tree: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('tree.csv', sep='\n', format="%d")


from sklearn.ensemble import GradientBoostingClassifier

if pth.exists('gradient_boost.pk1'):
    gradient_boost = joblib.load('gradient_boost.pk1')
    print('grad boost')
else:
    gradient_boost = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42).fit(X, y)
    joblib.dump(gradient_boost, 'gradient_boost.pk1')

print('fitted gradient boost')

out_valid = gradient_boost.predict(trans_valid)
print('gradient boost: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('gradient_boost.csv', sep='\n', format="%d")


from sklearn.ensemble import BaggingClassifier

if pth.exists('bagging.pk1'):
    bagging = joblib.load('bagging.pk1')
    print('bagging')
else:
    bagging = BaggingClassifier(KNeighborsClassifier(6), max_samples=0.5, max_features=0.5).fit(X, y)
    joblib.dump(bagging, 'bagging.pk1')

print('fitted bagging')
out_valid = bagging.predict(trans_valid)
print('bagging: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('bagging.csv', sep='\n', format="%d")


if pth.exists('bagging_forest.pk1'):
    bagging_forest = joblib.load('bagging_forest.pk1')
    print('bagging_forest')
else:
    bagging_forest = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=None,
                                                                             random_state=42),
                                       n_estimators=150, max_samples=0.5, max_features=0.5, random_state=42).fit(X, y)
    joblib.dump(bagging_forest, 'bagging_forest.pk1')

print('fitted bagging_forest')
out_valid = bagging_forest.predict(trans_valid)
#print('bagging_forest: ')
#print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('bagging_forest_only_train.csv', sep='\n', format="%d")

from sklearn.ensemble import VotingClassifier

if pth.exists('voting_soft.pk1'):
    voting_soft = joblib.load('voting_soft.pk1')
    print('voting_soft')
else:
    voting_soft=VotingClassifier(estimators=[('knn', neigh), ('gnb', bayes), ('bg', bagging_forest)], voting='soft').fit(X, y)
    #voting_soft=VotingClassifier(estimators=[('knn', neigh), ('rf', forest), ('gnb', bayes), ('bg', bagging_forest)], voting='soft').fit(X, y)
    joblib.dump(voting_soft, 'voting_soft.pk1')

print('fitted soft voting')

out_valid = voting_soft.predict(trans_valid)
print('soft voting: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('voting_soft.csv', sep='\n', format="%d")

if pth.exists('voting.pk1'):
    voting = joblib.load('voting.pk1')
    print('voting')
else:
    voting=VotingClassifier(estimators=[('knn', neigh), ('gnb', bayes), ('bg', bagging_forest)], voting='hard').fit(X, y)
    #voting=VotingClassifier(estimators=[('knn', neigh), ('rf', forest), ('gnb', bayes), ('bg', bagging_forest)], voting='hard').fit(X, y)
    joblib.dump(voting, 'voting.pk1')

print('fitted voting regular')

out_valid = voting.predict(trans_valid)
print('hard voting: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('voting.csv', sep='\n', format="%d")


from sklearn.model_selection import GridSearchCV

if pth.exists('grid.pk1'):
    grid = joblib.load('grid.pk1')
    print('grid')
else:
    voting_class = VotingClassifier(estimators=[('knn', neigh), ('rf', forest), ('gnb', bayes)], voting='soft')
    params = {'knn__n_neighbors': [3, 5], 'rf__n_estimators': [20, 50], 'rf__max_depth': [3, 7]}
    grid = GridSearchCV(estimator=voting_class, param_grid=params, cv=5)
    grid = grid.fit(X, y)
    results = grid.grid_scores_
    print('grids search results: ')
    print(results)
    joblib.dump(grid, 'grid.pk1')

print('fitted grid search')

out_valid = grid.predict(trans_valid)
print('grid search: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('grid.csv', sep='\n', format="%d")


from sklearn.ensemble import AdaBoostClassifier
if pth.exists('ada_boost.pk1'):
    ada = joblib.load('ada_boost.pk1')
    print('ada')
else:
    ada = AdaBoostClassifier(n_estimators=100).fit(X, y)
    joblib.dump(ada, 'ada_boost.pk1')

print('fitted ada boost')
out_valid = ada.predict(trans_valid)
print('ada boost: ')
print(np.count_nonzero(out_valid == valid_test)/len(valid_test) * 100)
out_valid.tofile('ada_boost.csv', sep='\n', format="%d")


del X
del y
#del valid_data
#del valid_test
del out_valid
del trans_valid
del percentile
del box_param

train_data = pd.read_csv("train.csv", sep='\t')
train_data["class"] = pd.read_csv("train_target.csv", names = ["class"])
valid_data = pd.read_csv("valid.csv", sep='\t')
valid_data["class"] = pd.read_csv("valid_target.csv", names = ["class"])

name_list = ['cli_pl_body', 'cli_cont_len', 'aggregated_sessions', 'net_samples', 'tcp_frag', 'tcp_ooo',
             'cli_tcp_ooo', 'cli_tcp_frag', 'cli_win_zero', 'cli_tcp_full', 'cli_pl_change', 'srv_tcp_ooo',
             'srv_tcp_frag', 'srv_win_zero', 'cli_tx_time', 'proxy', 'sp_healthscore', 'sp_req_duration',
             'sp_is_lat', 'sp_error']

train_data = train_data.append(valid_data, ignore_index=True)
del valid_data

print("One datafreme")

transformed_data = pd.DataFrame()
box_param = pd.DataFrame(data=None, columns=train_data.columns,index=range(0,1))
for word in list(train_data.columns.values):
    if word == 'class':
        transformed_data['class'] = train_data['class']
    elif word not in name_list:
        transformed_data[word], box_param[word][0] = boxcox(train_data[word] + 1)

del train_data
percentile = pd.DataFrame(data=None, columns=transformed_data.columns,index=range(0,2))
for word in list(transformed_data.columns.values):
    if word not in name_list and word != 'class':
        percentile[word][0] = np.percentile(transformed_data[word], 25)
        percentile[word][1] = np.percentile(transformed_data[word], 75)
        transformed_data[word] = normalization(transformed_data[word], percentile[word][0], percentile[word][1])

for word in list(transformed_data.columns.values):
    if word != 'class':
        mean = np.mean(transformed_data[word])
        std = np.std(transformed_data[word])
        transformed_data = transformed_data[transformed_data[word] < mean + 3*std]
        transformed_data = transformed_data[transformed_data[word] > mean - 3*std]

array = []
for i in np.random.rand(len(transformed_data)):
    array.append(i > 0.9)

undersampled_data = transformed_data[(transformed_data['class'] != 0) | (array)]
del transformed_data

print(undersampled_data)

X, y = undersampled_data.drop("class", 1), undersampled_data["class"]
del undersampled_data

test_data = pd.read_csv("test.csv", sep='\t')
trans_test = pd.DataFrame()
for word in list(test_data.columns.values):
    if word not in name_list:
        trans_test[word] = boxcox(test_data[word] + 1, lmbda=box_param[word][0])

del test_data

for word in list(trans_test.columns.values):
    if word not in name_list:
        trans_test[word] = normalization(trans_test[word], percentile[word][0], percentile[word][1])


if pth.exists('bagging_forest_test.pk1'):
    bagging_forest = joblib.load('bagging_forest_test.pk1')
    print('bagging_forest')
else:
    bagging_forest = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=None,
                                                                             random_state=42),
                                       n_estimators=150, max_samples=0.5, max_features=0.5, random_state=42).fit(X, y)
    joblib.dump(bagging_forest, 'bagging_forest_test.pk1')

print('fitted bagging_forest')
out_valid = bagging_forest.predict(trans_test)
out_valid.tofile('bagging_forest_final.csv', sep='\n', format="%d")
print("done")
