from rdkit.Chem import PandasTools
import pandas as pd
import numpy
from utils import computeFP, topNpreds
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support


def process_data(df):

    # get molecules
    mols = df[['MOLREGNO', 'SMILES']]
    mols = mols.drop_duplicates('MOLREGNO')
    mols = mols.set_index('MOLREGNO')
    mols = mols.sort_index()

    # get targets
    targets = df[['MOLREGNO', 'TARGET_CHEMBL_ID']]
    targets = targets.sort_values(by='MOLREGNO')
    targets = targets.groupby('MOLREGNO').apply(lambda x: ','.join(x.TARGET_CHEMBL_ID))
    targets = targets.apply(lambda x: x.split(','))
    targets = pd.DataFrame(targets, columns=['targets'])

    # generate fingerprints
    PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='SMILES')
    dataset = pd.merge(mols, targets, left_index=True, right_index=True)
    dataset = dataset.ix[dataset['ROMol'].notnull()]
    dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)
    dataset = dataset.ix[dataset['FP'].notnull()]

    X = numpy.array([f.fp for f in dataset['FP']])
    yy = numpy.array([c for c in dataset['targets']])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(yy)

    del dataset['ROMol']
    del dataset['FP']
    return X, y


def validation(X, y):
    morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=12)
    scores = cross_val_score(morgan_bnb, X, y, scoring=precision_recall_fscore_support, cv=KFold(n_splits=5, shuffle=True))
    return scores


df = pd.read_csv('out/22/chembl_1uM.csv')
X, y = process_data(df)
scores = validation(X, y)
print scores
