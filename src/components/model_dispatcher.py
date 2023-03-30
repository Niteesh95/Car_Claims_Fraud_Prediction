# model_dispatcher.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

specific_models = {
            "random_forest_gini": RandomForestClassifier(criterion='gini', 
                                                         max_depth=5,  
                                                         n_estimators=6, 
                                                         n_jobs=3),

            "random_forest_entropy": RandomForestClassifier(criterion='entropy', 
                                                         max_depth=5,  
                                                         n_estimators=6, 
                                                         n_jobs=3),
                
            "random_forest_logloss": RandomForestClassifier(criterion='log_loss', 
                                                         max_depth=5,  
                                                         n_estimators=6, 
                                                         n_jobs=3),

            "xgboost_gbtree": XGBClassifier(learning_rate=0.1, 
                                            max_depth=4, 
                                            booster='gbtree'),

            "decision_tree_gini": DecisionTreeClassifier(criterion='gini', 
                                                            max_depth=5, 
                                                            max_features=6, 
                                                            random_state=15),

            "decision_tree_entropy": DecisionTreeClassifier(criterion='entropy', 
                                                            max_depth=5, 
                                                            max_features=6, 
                                                            random_state=15),

            "decision_tree_logloss": DecisionTreeClassifier(criterion='log_loss', 
                                                            max_depth=5, 
                                                            max_features=6, 
                                                            random_state=15)                                                
            }

models = {
    "decision_tree": DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'xgboost':XGBClassifier()

}