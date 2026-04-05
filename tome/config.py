# config.py
class ToMeConfig:
    feature_choice = 'K'            # Options: 'Xpre', 'X', 'Q', 'K', 'V'
    distance_func = 'cosine'        # Options: 'eucl', 'cosine', 'dot', 'softmax'
    head_aggregation = 'mean'       # Options: 'mean', 'concat'
    combine_method = 'weighted avg' # Options: 'keep one', 'max pool', 'avg pool', 'weighted avg'
    partition_style = 'alternating' # Options: 'alternating', 'sequential', 'random'
