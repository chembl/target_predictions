import datetime
from django.db import models
from chembl_core_model.models import *
from chembl_core_db.db.customFields import BlobField, ChemblCharField
from chembl_core_db.db.customManagers import CompoundMolsManager
from chembl_core_db.db.models.abstractModel import ChemblCoreAbstractModel
from chembl_core_db.db.models.abstractModel import ChemblModelMetaClass
from django.utils import six

# Create your models here.

class TargetPredictions(six.with_metaclass(ChemblModelMetaClass, ChemblCoreAbstractModel)):

    pred_id = ChemblPositiveIntegerField(length=11, db_index=True, blank=False, null=False, help_text=u'Prediction unique ID')
    parent_molrgeno =  ChemblPositiveIntegerField(length=11, null=False, help_text=u'')
    chembl_id = ChemblCharField(max_length=20, null=False, help_text=u'ChEMBL identifier for this compound (for use on web interface etc)')
    tid = ChemblPositiveIntegerField(length=11, null=False, help_text=u'Unique ID for the target')
    target_chembl_id = ChemblCharField(max_length=20, help_text=u'ChEMBL identifier for this target (for use on web interface etc)')
    target_accession = ChemblCharField(max_length=20, help_text=u'Accession for the sequence in the source database from which it was taken (e.g., UniProt accession for proteins).')
    probability = models.DecimalField(max_digits=20, decimal_places=18, help_text=u'Probability of binding for this molecule-target pair')
    in_training = ChemblCharField(max_length=20, help_text=u'Mark if molecule is in training')
    value = ChemblPositiveIntegerField(length=5, null=False, help_text=u'1uM or 10uM')

    class Meta:
        db_table = 'target_predictions'
