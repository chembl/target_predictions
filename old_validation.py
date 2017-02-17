# -*- coding: utf-8 -*-

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import pandas as pd
from collections import OrderedDict
import numpy
from rdkit import DataStructs
from sklearn import cross_validation
import pickle

print 'Data preparation'

data = pd.read_csv('chembl_1uM.csv')

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

class FP:
    def __init__(self, fp):
        self.fp = fp

    def __str__(self):
        return self.fp.__str__()


def computeFP(x):
    # compute depth-2 morgan fingerprint hashed to 2048 bits
    fp = Chem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
    res = numpy.zeros(len(fp), numpy.int32)
    # convert the fingerprint to a numpy array and wrap it into the dummy container
    DataStructs.ConvertToNumpyArray(fp, res)
    return FP(res)


def topNpreds(fp, N=5):
    probas = list(morgan_bnb.predict_proba(fp)[0])
    d = dict(zip(classes, probas))
    scores = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))
    return scores.keys()[0:N], scores.values()[0:N]


dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)

# filter potentially failed fingerprint computations
dataset = dataset.ix[dataset['FP'].notnull()]

print 'fps done'

print 'validation'

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

X = numpy.array([f.fp for f in dataset['FP']])
y = numpy.array([c for c in dataset['targets']])

del dataset['ROMol']
del dataset['FP']

skf = cross_validation.StratifiedKFold(y, n_folds=5)

counter = 0

for train_ind, test_ind in skf:
    counter += 1
    print counter

    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]

    # morgan_bnb = OneVsRestClassifier(BernoulliNB(), n_jobs=4)
    morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=4)

    print 'model building'
    morgan_bnb.fit(X_train, y_train)

    print 'model done'

    classes = list(morgan_bnb.classes_)
    pred_targ = []
    probas = []

    print 'model validation'
    for f in X_test:
        pt, prb = topNpreds(f, 10)
        pred_targ.append(pt)
        probas.append(prb)

    data_test = dataset.iloc[test_ind]
    print len(pred_targ)
    print data_test.shape
    data_test['pred_targets'] = pred_targ
    data_test['probabilities'] = probas

    print data_test.shape
    data_test.to_csv('pred_1uM_{0}.csv'.format(counter), sep='\t')

print 'done!'

