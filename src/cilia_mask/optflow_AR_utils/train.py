import argparse
import joblib as jb
import numpy as np
import os.path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '#4: Train',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python train.py <args>')
    parser.add_argument('-i', '--inputdir', required = True,
        help = "Input directory where X and y are found.")
    parser.add_argument('-f', '--flow_type', choices = ['nvidia', 'opencv'], 
        required = True, help = "Type of optical flow features to use.")
    parser.add_argument('-o', '--output', required = True,
        help = "Output directory.")
    
    parser.add_argument("--random_state", required = False, default = 424242,
        type = int, help = "Set the random state of the estimator so things are repeatable.")
    
    args = vars(parser.parse_args())

    ft = args['flow_type']
    X = np.load(os.path.join(args['inputdir'], f"X_{ft}_train.npy"))
    y = np.load(os.path.join(args['inputdir'], f"y_{ft}_train.npy"))

    # Let's try fitting the model!
    # h = HistGradientBoostingRegressor(loss = "absolute_error", 
    #                                   verbose = 30,
    #                                   random_state = args['random_state'])
    h = HistGradientBoostingClassifier(verbose = 30,
                                       class_weight = 'balanced',
                                       random_state = args['random_state'])
    h.fit(X, y)
    print(h.score(X, y))

    # Serialize the model.
    # https://scikit-learn.org/stable/model_persistence.html
    #jb.dump(h, os.path.join(args['output'], f"hgbr_{ft}.joblib"))
    jb.dump(h, os.path.join(args['output'], f"hgbc_{ft}.joblib"))