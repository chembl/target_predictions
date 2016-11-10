import csv
import os
import sys
from collections import OrderedDict
import luigi
import numpy
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'deploy', 'deploy')))

import settings
from django.core.management import setup_environ
setup_environ(settings)

from chembl_migration_model.models import Activities, MoleculeDictionary, CompoundRecords, MoleculeHierarchy

class FP:
    def __init__(self, fp):
        self.fp = fp

    def __str__(self):
        return self.fp.__str__()


class GetActivities(luigi.Task):

    value = luigi.IntParameter()
    final_cols = ['MOLREGNO', 'TID', 'SMILES', 'PREF_NAME', 'CHEMBL_ID', 'TARGET_PREF_NAME', 'TARGET_CHEMBL_ID', 'TARGET_ACCESSION']

    def requires(self):
        return []

    def run(self):
        ac = Activities.objects.filter(molecule__compoundproperties__mw_freebase__range=(150, 1000),
                                       molecule__compoundproperties__aromatic_rings__lte=7,
                                       molecule__compoundproperties__rtb__lte=20,
                                       molecule__compoundproperties__heavy_atoms__range=(7, 42),
                                       molecule__compoundproperties__alogp__range=(-4, 10),
                                       molecule__compoundproperties__num_alerts__lte=4,
                                       standard_units='nM',
                                       standard_type__in=['EC50', 'IC50', 'Ki', 'Kd', 'XC50', 'AC50', 'Potency'],
                                       standard_value__lte=self.value*1000,
                                       data_validity_comment=None,
                                       standard_relation__in=['=', '<'],
                                       potential_duplicate=0,
                                       assay__confidence_score__confidence_score__gte=8,
                                       assay__target__target_type__target_type='SINGLE PROTEIN',)

        df = pd.DataFrame.from_records(ac.values('activity_id',
                                                 'assay_id',
                                                 'doc_id',
                                                 'molecule__moleculehierarchy__parent_molecule_id',
                                                 'standard_relation',
                                                 'standard_value',
                                                 'standard_units',
                                                 'molecule__compoundstructures__canonical_smiles',
                                                 'molecule__pref_name',
                                                 'molecule__chembl_id',
                                                 'assay__target__tid',
                                                 'assay__target__pref_name',
                                                 'assay__target__chembl_id',
                                                 'assay__target__component_sequences__accession'))

        df.rename(columns={'activity_id': 'ACTIVITY_ID',
                           'assay_id': 'ASSAY_ID',
                           'doc_id': 'DOC_ID',
                           'molecule__moleculehierarchy__parent_molecule_id': 'MOLREGNO',
                           'standard_relation': 'STANDARD_RELATION',
                           'standard_value': 'STANDARD_VALUE',
                           'standard_units': 'STANDARD_UNITS',
                           'molecule__compoundstructures__canonical_smiles': 'SMILES',
                           'molecule__pref_name': 'PREF_NAME',
                           'molecule__chembl_id': 'CHEMBL_ID',
                           'assay__target__tid': 'TID',
                           'assay__target__pref_name': 'TARGET_PREF_NAME',
                           'assay__target__chembl_id': 'TARGET_CHEMBL_ID',
                           'assay__target__component_sequences__accession': 'TARGET_ACCESSION'}, inplace=True)

        # Upper branch
        # IT'S ONLY A REMOVAL OF DUPLICATES
        dfu2 = df.drop_duplicates(subset=['MOLREGNO', 'TID'])

        # Middle branch
        dfm2 = df[['MOLREGNO', 'TARGET_CHEMBL_ID']].drop_duplicates()
        dfm3 = dfm2.groupby(['TARGET_CHEMBL_ID']).agg('count')
        dfm4 = dfm3[dfm3['MOLREGNO'] >= 30].reset_index()

        # Lower branch
        dfl2 = df[['DOC_ID', 'TARGET_CHEMBL_ID']].drop_duplicates()
        dfl3 = dfl2.groupby(['TARGET_CHEMBL_ID']).agg('count')
        dfl4 = dfl3[dfl3['DOC_ID'] >= 2.0].reset_index()

        # Joins
        ml_join = pd.merge(dfm4, dfl4, how='inner', on='TARGET_CHEMBL_ID')

        u_join = dfu2[dfu2['TARGET_CHEMBL_ID'].isin(ml_join['TARGET_CHEMBL_ID'])].sort_values(
            by=['MOLREGNO', 'TID', ]).reset_index(drop=True)
        u_join = u_join[self.final_cols]
        u_join.to_csv('chembl_{}uM.csv'.format(self.value), index=False, quoting=csv.QUOTE_NONNUMERIC)

    def output(self):
        return luigi.LocalTarget('chembl_{}uM.csv'.format(self.value))


class GetDrugs(luigi.Task):

    final_cols = ['PARENT_MOLREGNO', 'CHEMBL_ID', 'CANONICAL_SMILES']

    def requires(self):
        return []

    def run(self):
        crs = CompoundRecords.objects.filter(src_id__in=[8, 9, 12, 13, 36],).values_list('molecule__molregno', flat=True)
        # 8 -> clinical candidates
        # 9 -> fda orange book
        # 12 -> Manually added drugs
        # 13 -> usp dictionary of usan and international drug names
        # 36 -> withdrawn drugs

        mhs = MoleculeHierarchy.objects.filter(molecule__molecule_type='Small molecule',
                                               molecule__compoundstructures__canonical_smiles__isnull=False,
                                               molecule__compoundproperties__mw_freebase__range=(150, 1000),
                                               molecule__compoundproperties__aromatic_rings__lte=7,
                                               molecule__compoundproperties__rtb__lte=20,
                                               molecule__compoundproperties__heavy_atoms__range=(7, 42),
                                               molecule__compoundproperties__alogp__range=(-4, 10),
                                               molecule__compoundproperties__num_alerts__lte=4,
                                               molecule__molregno__in=crs).values_list('parent_molecule__molregno', flat=True).distinct()

        mols = MoleculeDictionary.objects.filter(molregno__in=mhs)

        df = pd.DataFrame.from_records(mols.values('molregno',
                                                   'chembl_id',
                                                   'compoundstructures__canonical_smiles',))

        df.rename(columns={'molregno': 'PARENT_MOLREGNO',
                           'chembl_id': 'CHEMBL_ID',
                           'compoundstructures__canonical_smiles': 'CANONICAL_SMILES'}, inplace=True)

        df2 = df[self.final_cols]
        df2.to_csv('chembl_drugs.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    def output(self):
        return luigi.LocalTarget('chembl_drugs.csv')


class MakeModel(luigi.Task):
    value = luigi.IntParameter()

    def requires(self):
        return [GetActivities(self.value)]

    def run(self):
        data = pd.read_csv('chembl_{}uM.csv'.format(self.value))

        mols = data[['MOLREGNO', 'SMILES']]
        mols = mols.drop_duplicates('MOLREGNO')
        mols = mols.set_index('MOLREGNO')
        mols = mols.sort_index()

        targets = data[['MOLREGNO', 'TARGET_CHEMBL_ID']]
        targets = targets.sort_index(by='MOLREGNO')

        targets = targets.groupby('MOLREGNO').apply(lambda x: ','.join(x.TARGET_CHEMBL_ID))
        targets = targets.apply(lambda x: x.split(','))
        targets = pd.DataFrame(targets, columns=['targets'])

        PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='SMILES')
        dataset = pd.merge(mols, targets, left_index=True, right_index=True)
        dataset = dataset.ix[dataset['ROMol'].notnull()]

        def computeFP(x):
            # compute depth-2 morgan fingerprint hashed to 2048 bits
            fp = Chem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
            res = numpy.zeros(len(fp), numpy.int32)
            # convert the fingerprint to a numpy array and wrap it into the dummy container
            DataStructs.ConvertToNumpyArray(fp, res)
            return FP(res)

        dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)

        # filter potentially failed fingerprint computations
        dataset = dataset.ix[dataset['FP'].notnull()]

        X = [f.fp for f in dataset['FP']]
        yy = [c for c in dataset['targets']]

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(yy)

        morgan_bnb = OneVsRestClassifier(MultinomialNB())
        morgan_bnb.fit(X, y)
        morgan_bnb.targets = mlb.classes_

        joblib.dump(morgan_bnb, 'models/{}uM/mNB_{}uM_all.pkl'.format(self.value, self.value))

    def output(self):
        return luigi.LocalTarget('models/{}uM/mNB_{}uM_all.pkl'.format(self.value, self.value))


class MakePredictions(luigi.Task):
    value = luigi.IntParameter()

    def requires(self):
        return [GetDrugs(), MakeModel(self.value)]

    def run(self):
        mols = pd.read_csv('chembl_drugs.csv'.format(self.value))
        morgan_bnb = joblib.load('models/{}uM/mNB_{}uM_all.pkl'.format(self.value, self.value))

        def topNpreds(m, fp, N=5):
            probas = list(morgan_bnb.predict_proba(fp)[0])
            d = dict(zip(classes, probas))
            scores = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))
            return [(m, t, s) for t, s in scores.items()[0:N]]

        classes = list(morgan_bnb.targets)

        PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='CANONICAL_SMILES')
        mols = mols.ix[mols['ROMol'].notnull()]

        def computeFP(x):
            # compute depth-2 morgan fingerprint hashed to 1024 bits
            fp = Chem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
            res = numpy.zeros(len(fp), numpy.int32)
            # convert the fingerprint to a numpy array and wrap it into the dummy container
            DataStructs.ConvertToNumpyArray(fp, res)
            return FP(res.reshape(1, -1))

        mols['FP'] = mols.apply(lambda row: computeFP(row['ROMol']), axis=1)

        # filter potentially failed fingerprint computations
        mols = mols.ix[mols['FP'].notnull()]
        fps = [f.fp for f in mols['FP']]
        molregnos = mols['PARENT_MOLREGNO']

        ll = []
        for m, f in zip(molregnos, fps):
            ll.extend(topNpreds(m, f, 50))

        preds = pd.DataFrame(ll, columns=['molregno', 'target_chembl_id', 'proba'])
        preds.to_csv('drug_predictions_{}uM.csv'.format(self.value))

    def output(self):
        return luigi.LocalTarget('drug_predictions_{}uM.csv'.format(self.value))


class FinalTask(luigi.Task):
    value = luigi.IntParameter()

    def requires(self):
        return [GetActivities(self.value), GetDrugs(), MakePredictions(self.value)]

    def run(self):
        ac = pd.read_csv('chembl_{}uM.csv'.format(self.value))
        dr = pd.read_csv('chembl_drugs.csv'.format(self.value))

        # "groupby" targets
        df2 = ac.drop_duplicates('TID')[['TID', 'TARGET_PREF_NAME', 'TARGET_CHEMBL_ID', 'TARGET_ACCESSION']]
        df3 = df2.sort_values(by='TID').reset_index(drop=True)

        # add column (java snippet)
        ac['exists'] = 'YES'

        preds = pd.read_csv('drug_predictions_{}uM.csv'.format(self.value))

        # no rename
        del preds['Unnamed: 0']

        d_join = pd.merge(dr, preds, how='inner', left_on='PARENT_MOLREGNO', right_on='molregno')

        # should be right join to match number of rows but it forces pandas to convert TID to float
        d_2_join = pd.merge(df3, d_join, how='right', left_on='TARGET_CHEMBL_ID', right_on='target_chembl_id')
        d_sort = d_2_join.sort_values(by=['PARENT_MOLREGNO', 'proba', ], ascending=[1, 0])

        # final join
        last_join = pd.merge(ac[['exists', 'MOLREGNO', 'TARGET_CHEMBL_ID']], d_sort, how='right',
                             left_on=['MOLREGNO', 'TARGET_CHEMBL_ID'], right_on=['PARENT_MOLREGNO', 'TARGET_CHEMBL_ID'])
        last_join_sort = last_join.sort_values(by=['PARENT_MOLREGNO', 'proba', ], ascending=[1, 0])

        # add column (java snippet)
        last_join_sort['PRED_ID'] = range(1, len(last_join_sort) + 1)

        final_columns = ['PRED_ID', 'PARENT_MOLREGNO', 'CHEMBL_ID', 'TID', 'TARGET_CHEMBL_ID', 'TARGET_ACCESSION',
                         'proba', 'exists']

        final = last_join_sort[final_columns]
        final.rename(columns={'proba': 'PROBABILITY', 'exists': 'IN_TRAINING'}, inplace=True)

        final.to_csv('final_result_{}uM.csv'.format(self.value))

    def output(self):
        return luigi.LocalTarget('final_result_{}uM.csv'.format(self.value))


class MergeTables(luigi.Task):
    def requires(self):
        return [FinalTask(1), FinalTask(10)]

    def run(self):
        ten = pd.read_csv('final_result_10uM.csv')
        one = pd.read_csv('final_result_1uM.csv')

        one['VALUE'] = 1
        ten['VALUE'] = 10

        result = pd.concat([one, ten])
        result.to_csv('merged_tables.csv')

    def output(self):
        return luigi.LocalTarget('merged_tables.csv')


if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/10uM'):
        os.makedirs('models/10uM')
    if not os.path.exists('models/1uM'):
        os.makedirs('models/1uM')
    # luigi.run(['GetActivities', '--local-scheduler', '--value', '10', '--workers', '2'])
    luigi.run(['MergeTables', '--local-scheduler', '--workers', '2'])