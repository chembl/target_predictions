from rdkit.Chem import PandasTools
import pandas as pd
import numpy
from utils import computeFP
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import luigi


class ProcessValidationData(luigi.Task):

    def requires(self):
        return []

    def run(self):
        df = pd.read_csv('out/22/chembl_1uM.csv')

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
        df = pd.merge(mols, targets, left_index=True, right_index=True)
        df = df.ix[df['ROMol'].notnull()]
        df['FP'] = df.apply(lambda row: computeFP(row['ROMol']).fp, axis=1)
        del df['ROMol']
        df = df.ix[df['FP'].notnull()]
        df.to_csv(self.output().path)

    def output(self):
        return luigi.LocalTarget('processed_validation_data.csv')


class Validate(luigi.Task):
    def requires(self):
        return ProcessValidationData()

    def run(self):
        df = pd.read_csv('processed_validation_data.csv')

        X = df['FP']
        yy = df['targets']
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(yy)

        morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=12)
        sc = cross_val_score(morgan_bnb, X, y, scoring=precision_recall_fscore_support,
                                 cv=KFold(n_splits=5, shuffle=True))
        print sc
        with self.output().open('w') as output:
            output.write(sc)

    def output(self):
        return luigi.LocalTarget('validated.txt')

if __name__ == "__main__":
    luigi.run(['Validate', '--local-scheduler', '--workers', '1'])
