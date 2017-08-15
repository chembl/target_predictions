import os
import csv
import luigi
import logging
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from chembl_release_utils.orm import create_engine_base
from chembl_release_utils.settings import DATABASES
from utils import computeFP, topNpreds
from sqlalchemy import and_
from sqlalchemy.orm import Session

# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger('target_predictions')
logger.setLevel(logging.DEBUG)

# create a file handler
handler = logging.FileHandler('target_predictions.log')
handler.setLevel(logging.DEBUG)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)

# ----------------------------------------------------------------------------------------------------------------------

CHUNK_SIZE = 10000
MAIN_DIR = 'chembl_{}/'

# ----------------------------------------------------------------------------------------------------------------------


class GetActivities(luigi.Task):

    db = luigi.Parameter()
    value = luigi.IntParameter()
    chembl_version = luigi.Parameter()
    final_cols = ['molregno', 'tid', 'canonical_smiles', 'pref_name', 'chembl_id', 'target_pref_name', 'target_chembl_id',
                  'target_accession', 'standard_value']

    def requires(self):
        return []

    def run(self):
        logger.info('GetActivities {}nM: Init'.format(self.value))
        # create dirs
        if not os.path.exists(MAIN_DIR.format(self.chembl_version)+'models/{}uM'.format(self.value)):
            os.makedirs(MAIN_DIR.format(self.chembl_version)+'models/{}uM'.format(self.value))
        engine, Base = create_engine_base(DATABASES[self.db])
        MoleculeDictionary = Base.classes.MoleculeDictionary
        CompoundStructures = Base.classes.CompoundStructures
        MoleculeHierarchy = Base.classes.MoleculeHierarchy
        TargetDictionary = Base.classes.TargetDictionary
        TargetComponents = Base.classes.TargetComponents
        CompoundProperties = Base.classes.CompoundProperties
        Assays = Base.classes.Assays
        Activities = Base.classes.Activities
        ComponentSequences = Base.classes.ComponentSequences

        s = Session(engine)
        q = s.query(Activities)\
            .join(CompoundProperties, and_(Activities.molregno == CompoundProperties.molregno))\
            .join(Assays, and_(Activities.assay_id == Assays.assay_id))\
            .join(TargetDictionary, and_(Assays.tid == TargetDictionary.tid))\
            .join(TargetComponents, and_(TargetDictionary.tid == TargetComponents.tid))\
            .join(ComponentSequences, and_(TargetComponents.component_id == ComponentSequences.component_id))\
            .join(MoleculeDictionary, and_(Activities.molregno == MoleculeDictionary.molregno))\
            .join(MoleculeHierarchy, and_(MoleculeDictionary.molregno == MoleculeHierarchy.molregno))\
            .join(CompoundStructures, and_(MoleculeHierarchy.parent_molregno == CompoundStructures.molregno))\
            .filter(CompoundProperties.mw_freebase >= 150,
                    CompoundProperties.mw_freebase <= 1000,
                    CompoundProperties.aromatic_rings <= 7,
                    CompoundProperties.rtb <= 20,
                    CompoundProperties.heavy_atoms >= 7,
                    CompoundProperties.heavy_atoms <= 42,
                    CompoundProperties.alogp >= -4,
                    CompoundProperties.alogp <= 10,
                    Activities.standard_units == 'nM',
                    Activities.standard_type.in_(['EC50', 'IC50', 'Ki', 'Kd', 'XC50', 'AC50', 'Potency']),
                    Activities.standard_value <= self.value * 1000,
                    Activities.data_validity_comment == None,
                    Activities.standard_relation.in_(['=', '<']),
                    Activities.potential_duplicate == 0,
                    Assays.confidence_score >= 8,
                    TargetDictionary.target_type == 'SINGLE PROTEIN')\
            .with_entities(Activities.doc_id,
                           Activities.standard_value,
                           MoleculeHierarchy.parent_molregno.label('molregno'),
                           CompoundStructures.canonical_smiles,
                           MoleculeDictionary.pref_name,
                           MoleculeDictionary.chembl_id,
                           TargetDictionary.tid,
                           TargetDictionary.pref_name.label('target_pref_name'),
                           TargetDictionary.chembl_id.label('target_chembl_id'),
                           ComponentSequences.accession.label('target_accession'))

        # read to pandas and convert NaN to None
        df = pd.read_sql(q.statement, q.session.bind)
        df = df.where((pd.notnull(df)), None)
        logger.info('GetActivities {}nM: Query finished'.format(self.value))

        # Upper branch
        dfu2 = df.drop_duplicates(subset=['molregno', 'tid'])

        # Get targets with at least 30 different active molecules
        dfm = df[['molregno', 'target_chembl_id']].drop_duplicates()
        dfm2 = dfm.groupby(['target_chembl_id']).agg('count')
        dfm3 = dfm2[dfm2['molregno'] >= 30].reset_index()

        # Get targets mentioned in at least two docs
        dfd = df[['doc_id', 'target_chembl_id']].drop_duplicates()
        dfd2 = dfd.groupby(['target_chembl_id']).agg('count')
        dfd3 = dfd2[dfd2['doc_id'] >= 2.0].reset_index()

        # Joins
        ml_join = pd.merge(dfm3, dfd3, how='inner', on='target_chembl_id')

        u_join = dfu2[dfu2['target_chembl_id'].isin(ml_join['target_chembl_id'])].sort_values(
            by=['molregno', 'tid', ]).reset_index(drop=True)
        u_join = u_join[self.final_cols]
        u_join.to_csv(self.output().path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info('GetActivities {}nM: Done'.format(self.value))

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'chembl_{}uM.csv'.format(self.value))

# ----------------------------------------------------------------------------------------------------------------------


class GetDrugs(luigi.Task):

    db = luigi.Parameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return []

    def run(self):
        logger.info('GetDrugs: Init')
        if not os.path.exists(MAIN_DIR.format(self.chembl_version)):
            os.makedirs(MAIN_DIR.format(self.chembl_version))
        engine, Base = create_engine_base(DATABASES[self.db])
        MoleculeDictionary = Base.classes.MoleculeDictionary
        CompoundStructures = Base.classes.CompoundStructures
        MoleculeHierarchy = Base.classes.MoleculeHierarchy
        CompoundProperties = Base.classes.CompoundProperties
        CompoundRecords = Base.classes.CompoundRecords

        s = Session(engine)
        # 8 -> clinical candidates
        # 9 -> fda orange book
        # 12 -> Manually added drugs
        # 13 -> usp dictionary of usan and international drug names
        # 36 -> withdrawn drugs
        # 41 -> WHO Anatomical Therapeutic Chemical Classification
        # 42 -> British National Formulary
        sq = s.query(MoleculeHierarchy) \
            .join(CompoundRecords, and_(MoleculeHierarchy.molregno == CompoundRecords.molregno)) \
            .filter(CompoundRecords.src_id.in_([8, 9, 12, 13, 36, 41, 42]))\
            .filter(CompoundRecords.removed == 0) \
            .with_entities(MoleculeHierarchy.parent_molregno) \
            .distinct(MoleculeHierarchy.parent_molregno).subquery()
        q = s.query(MoleculeDictionary)\
            .join(CompoundProperties, and_(MoleculeDictionary.molregno == CompoundProperties.molregno))\
            .join(CompoundStructures, and_(MoleculeDictionary.molregno == CompoundStructures.molregno))\
            .filter(MoleculeDictionary.molregno.in_(sq))\
            .filter(CompoundProperties.mw_freebase >= 150,
                    CompoundProperties.mw_freebase <= 1000,
                    CompoundProperties.aromatic_rings <= 7,
                    CompoundProperties.rtb <= 20,
                    CompoundProperties.heavy_atoms >= 7,
                    CompoundProperties.heavy_atoms <= 42,
                    CompoundProperties.alogp >= -4,
                    CompoundProperties.alogp <= 10,
                    )\
            .with_entities(MoleculeDictionary.molregno.label('parent_molregno'),
                           MoleculeDictionary.chembl_id,
                           CompoundStructures.canonical_smiles)

        # read to pandas and convert NaN to None
        df = pd.read_sql(q.statement, q.session.bind)
        df = df.where((pd.notnull(df)), None)
        df.to_csv(self.output().path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info('GetDrugs: Done')

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'chembl_drugs.csv')

# ----------------------------------------------------------------------------------------------------------------------


class MakeModel(luigi.Task):

    db = luigi.Parameter()
    value = luigi.IntParameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return GetActivities(value=self.value, chembl_version=self.chembl_version, db=self.db)

    def run(self):
        logger.info('MakeModel {}nM: Init'.format(self.value))
        data = pd.read_csv(self.input().path)
        # get unique molecules and its smiles
        mols = data[['molregno', 'canonical_smiles']]
        mols = mols.drop_duplicates('molregno')
        mols = mols.set_index('molregno')
        mols = mols.sort_index()

        # group targets by molregno
        targets = data[['molregno', 'target_chembl_id']]
        #targets = targets.sort_index(by='molregno')
        targets = targets.sort_values(by='molregno')
        targets = targets.groupby('molregno').apply(lambda x: ','.join(x.target_chembl_id))
        targets = targets.apply(lambda x: x.split(','))
        targets = pd.DataFrame(targets, columns=['targets'])

        # merge it
        PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='canonical_smiles')
        dataset = pd.merge(mols, targets, left_index=True, right_index=True)
        dataset = dataset.ix[dataset['ROMol'].notnull()]

        # generate fingerprints
        dataset['FP'] = dataset.apply(lambda row: computeFP(row['ROMol']), axis=1)
        dataset = dataset.ix[dataset['FP'].notnull()]
        logger.info('MakeModel {}nM: Data ready'.format(self.value))

        # generate models training data
        X = [f.fp for f in dataset['FP']]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(dataset['targets'])

        # train the model
        morgan_bnb = OneVsRestClassifier(MultinomialNB())
        morgan_bnb.fit(X, y)
        morgan_bnb.targets = mlb.classes_

        # save the model
        joblib.dump(morgan_bnb, self.output().path)
        logger.info('MakeModel {}nM: Done'.format(self.value))

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'models/{}uM/mNB_{}uM_all.pkl'.format(self.value, self.value))

# ----------------------------------------------------------------------------------------------------------------------


class MakePredictions(luigi.Task):

    db = luigi.Parameter()
    value = luigi.IntParameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return [GetDrugs(chembl_version=self.chembl_version, db=self.db),
                MakeModel(value=self.value, chembl_version=self.chembl_version, db=self.db)]

    def run(self):
        logger.info('MakePredictions {}nM: Init'.format(self.value))
        mols = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'chembl_drugs.csv'.format(self.value))
        morgan_bnb = joblib.load(MAIN_DIR.format(self.chembl_version)+'models/{}uM/mNB_{}uM_all.pkl'.format(self.value, self.value))

        classes = list(morgan_bnb.targets)

        PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='canonical_smiles')
        mols = mols.ix[mols['ROMol'].notnull()]

        mols['FP'] = mols.apply(lambda row: computeFP(row['ROMol']), axis=1)

        # filter potentially failed fingerprint computations
        mols = mols.ix[mols['FP'].notnull()]
        fps = [f.fp for f in mols['FP']]
        molregnos = mols['parent_molregno']

        ll = []
        for m, f in zip(molregnos, fps):
            ll.extend(topNpreds(morgan_bnb, classes, m, f, 50))

        preds = pd.DataFrame(ll, columns=['molregno', 'target_chembl_id', 'proba'])
        preds.to_csv(self.output().path)
        logger.info('MakePredictions {}nM: Done'.format(self.value))

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'drug_predictions_{}uM.csv'.format(self.value))

# ----------------------------------------------------------------------------------------------------------------------


class GenerateValueTableData(luigi.Task):

    db = luigi.Parameter()
    value = luigi.IntParameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return [GetActivities(value=self.value, chembl_version=self.chembl_version, db=self.db),
                GetDrugs(chembl_version=self.chembl_version, db=self.db),
                MakePredictions(value=self.value, chembl_version=self.chembl_version, db=self.db)]

    def run(self):
        logger.info('GenerateValueTableData {}nM: Init'.format(self.value))
        ac = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'chembl_{}uM.csv'.format(self.value))
        dr = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'chembl_drugs.csv'.format(self.value))

        # drop dupicate targets (check if is really needed)
        df2 = ac.drop_duplicates('tid')[['tid', 'target_pref_name', 'target_chembl_id', 'target_accession']]
        df3 = df2.sort_values(by='tid').reset_index(drop=True)

        ac['exists'] = 1
        preds = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'drug_predictions_{}uM.csv'.format(self.value))

        # no rename
        del preds['Unnamed: 0']
        d_join = pd.merge(dr, preds, how='right', left_on='parent_molregno', right_on='molregno')

        d_2_join = pd.merge(df3, d_join, how='right', left_on='target_chembl_id', right_on='target_chembl_id')
        d_sort = d_2_join.sort_values(by=['parent_molregno', 'proba', ], ascending=[1, 0])

        # final join
        last_join = pd.merge(ac[['exists', 'molregno', 'target_chembl_id']], d_sort, how='right',
                             left_on=['molregno', 'target_chembl_id'], right_on=['parent_molregno', 'target_chembl_id'])
        last_join_sort = last_join.sort_values(by=['parent_molregno', 'proba', ], ascending=[1, 0])

        final_columns = ['parent_molregno', 'chembl_id', 'tid', 'target_chembl_id', 'target_accession',
                         'proba', 'exists']
        final = last_join_sort[final_columns]
        final['exists'].fillna(0, inplace=True)

        final.rename(columns={'proba': 'probability', 'exists': 'in_training'}, inplace=True)
        final.to_csv(self.output().path, index=False)
        logger.info('GenerateValueTableData {}nM: Done'.format(self.value))

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'final_result_{}uM.csv'.format(self.value))

# ----------------------------------------------------------------------------------------------------------------------


class MergeTables(luigi.Task):

    db = luigi.Parameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return [GenerateValueTableData(value=1, chembl_version=self.chembl_version, db=self.db),
                GenerateValueTableData(value=10, chembl_version=self.chembl_version, db=self.db)]

    def run(self):
        logger.info('MergeTables: Init')
        ten = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'final_result_10uM.csv')
        one = pd.read_csv(MAIN_DIR.format(self.chembl_version)+'final_result_1uM.csv')

        one['value'] = 1
        ten['value'] = 10
        result = pd.concat([one, ten])

        result['pred_id'] = range(1, len(result) + 1)
        result.to_csv(self.output().path, index=False)
        logger.info('MergeTables: Done')

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'merged_tables.csv')

# ----------------------------------------------------------------------------------------------------------------------


class InsertDb(luigi.Task):

    db = luigi.Parameter()
    chembl_version = luigi.Parameter()

    def requires(self):
        return MergeTables(chembl_version=self.chembl_version, db=self.db)

    def run(self):
        logger.info('InsertDb: Init')
        engine, Base = create_engine_base(DATABASES[self.db])
        TargetPredictions = Base.classes.TargetPredictions
        s = Session(engine)
        num_rows_deleted = s.query(TargetPredictions).delete()
        s.commit()
        s.close()

        predictions = pd.read_csv(self.input().path)
        predictions = predictions.where((pd.notnull(predictions)), None)

        with engine.begin() as conn:
            chunks_ini = [x for x in range(0, predictions.shape[0], CHUNK_SIZE)]
            for ini in chunks_ini:
                # bulk insert
                conn.execute(
                    TargetPredictions.__table__.insert(),
                    predictions[ini:ini + CHUNK_SIZE].to_dict(orient='records')
                )

        with self.output().open('w') as output:
            output.write('done')
        logger.info('InsertDb: Done')

    def output(self):
        return luigi.LocalTarget(MAIN_DIR.format(self.chembl_version)+'inserts.done')

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    luigi.run(['InsertDb', '--local-scheduler', '--chembl-version', '23', '--db', 'chembl', '--workers', '3'])
