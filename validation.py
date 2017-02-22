import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import luigi
import ast

def make_FP(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        res = np.zeros(len(fp), np.int32)
        DataStructs.ConvertToNumpyArray(fp, res)
        res = fp.ToBitString()
    except:
        res = None
    return res

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
        df = pd.merge(mols, targets, left_index=True, right_index=True)
        df['FP'] = df.apply(lambda row: make_FP(row['SMILES']), axis=1)
        df = df.ix[df['FP'].notnull()]
        df.to_csv(self.output().path)

    def output(self):
        return luigi.LocalTarget('processed_validation_data.csv')


class Validate(luigi.Task):
    def requires(self):
        return ProcessValidationData()

    def run(self):
        df = pd.read_csv(self.input().path)

        X = np.array([[int(i) for i in x] for x in df['FP']])
        yy = [ast.literal_eval(x) for x in df['targets']]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(yy)

        morgan_bnb = OneVsRestClassifier(MultinomialNB(), n_jobs=12)
        sc = cross_val_score(morgan_bnb, X, y, scoring='f1_weighted',
                                 cv=KFold(n_splits=5, shuffle=True))
        with self.output().open('w') as output:
            output.write(str(sc))

    def output(self):
        return luigi.LocalTarget('validated.txt')

if __name__ == "__main__":
    luigi.run(['Validate', '--local-scheduler', '--workers', '1'])