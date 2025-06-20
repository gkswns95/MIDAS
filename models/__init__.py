from models.brits.brits import BRITS
from models.midas.midas import MIDAS
from models.midas.dbhp import DBHP
from models.graph_imputer.gi import GraphImputer
from models.ImputeFormer.imputeformer import IMPUTEFORMER
from models.naomi.naomi import NAOMI
from models.nrtsi.nrtsi import NRTSI
from models.csdi.csdi import CSDI

def load_model(model_name, params, parser=None, device=None):
    model_name = model_name.lower()

    if model_name == "midas":
        return MIDAS(params, parser)
    elif model_name == "dbhp":
        return DBHP(params, parser)
    elif model_name == "brits":
        return BRITS(params, parser)
    elif model_name == "naomi":
        return NAOMI(params, parser)
    elif model_name == "nrtsi":
        return NRTSI(params, parser)
    elif model_name == "graph_imputer":
        return GraphImputer(params, parser)
    elif model_name == "csdi":
        return CSDI(params, parser, device)
    elif model_name == "imputeformer":
        return IMPUTEFORMER(params, parser)
    else:
        raise NotImplementedError