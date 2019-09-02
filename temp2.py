import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.3659082704541167
exported_pipeline = make_pipeline(
    Normalizer(norm="l2"),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=1, max_features=0.2, min_samples_leaf=19, min_samples_split=10, n_estimators=100, subsample=1.0)),
    Normalizer(norm="l1"),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.05, min_samples_leaf=17, min_samples_split=20, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
