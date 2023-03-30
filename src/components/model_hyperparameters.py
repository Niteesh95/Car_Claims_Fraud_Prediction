parameters = {
    "decision_tree":{
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_depth' : [3, 5, 7, 10],
        # 'min_samples_split' : range(2, 10, 1),
        # 'min_samples_leaf' : range(2, 10, 1),
        'max_features' : range(2,11,1),
        'random_state': range(0,50,5)},

    "random_forest": {
        'criterion' : ['gini', "entropy", "log_loss"], 
        'max_depth' : range(2,11,1),
        'n_estimators' : range(2,11,1),
        'n_jobs': [2,3,4,5]},

    "xgboost":{
        'learning_rate':[.1,.01,.05,.001],
        'n_estimators': [8,16,32,64,128,256],
        'criterion': ['gbtree']}
}