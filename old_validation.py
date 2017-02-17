# -*- coding: utf-8 -*-

from rdkit.Chem import PandasTools
import pandas as pd
import numpy
from utils import computeFP, topNpreds
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

print 'Data preparation'

data = pd.read_csv('out/22/chembl_1uM.csv')

print "data", data.shape

mols = data[['MOLREGNO', 'SMILES']]
mols = mols.drop_duplicates('MOLREGNO')
mols = mols.set_index('MOLREGNO')
mols = mols.sort_index()

print "mols", mols.shape

targets = data[['MOLREGNO', 'TARGET_CHEMBL_ID']]
targets = targets.sort_index(by='MOLREGNO')

targets = targets.groupby('MOLREGNO').apply(lambda x: ','.join(x.TARGET_CHEMBL_ID))
targets = targets.apply(lambda x: x.split(','))
targets = pd.DataFrame(targets, columns=['targets'])

print "targets", targets.shape

PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='SMILES')

dataset = pd.merge(mols, targets, left_index=True, right_index=True)

dataset = dataset.ix[dataset['ROMol'].notnull()]

print "dataset", dataset.shape

# Learning

dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)

# filter potentially failed fingerprint computations
dataset = dataset.ix[dataset['FP'].notnull()]

print 'fps done'

print 'validation'

X = numpy.array([f.fp for f in dataset['FP']])
y = numpy.array([c for c in dataset['targets']])

del dataset['ROMol']
del dataset['FP']

skf = StratifiedKFold(y, n_folds=5)

counter = 0
for train_ind, test_ind in skf:
    counter += 1
    print counter

    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]

    morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=4)

    print 'model building'
    morgan_bnb.fit(X_train, y_train)

    print 'model done'

    classes = list(morgan_bnb.classes_)
    pred_targ = []
    probas = []

    print 'model validation'
    for f in X_test:
        pt, prb = topNpreds(morgan_bnb, f, 10)
        pred_targ.append(pt)
        probas.append(prb)

    data_test = dataset.iloc[test_ind]
    print len(pred_targ)
    print data_test.shape
    data_test['pred_targets'] = pred_targ
    data_test['probabilities'] = probas

    print data_test.shape
    data_test.to_csv('out/22/pred_1uM_{0}.csv'.format(counter), sep='\t')

print 'done!'