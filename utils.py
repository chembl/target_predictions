from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
from collections import OrderedDict
import numpy

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

def topNpreds(m, fp, N=5):
    probas = list(morgan_bnb.predict_proba(fp)[0])
    d = dict(zip(classes, probas))
    scores = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))
    return [(m, t, s) for t, s in scores.items()[0:N]]
