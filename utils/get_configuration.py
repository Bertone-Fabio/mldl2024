from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import numpy as np
import torch.nn as nn
from . import aggregators

def get_loss(loss_name):
    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None

def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'avgpool' in agg_arch.lower():
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )