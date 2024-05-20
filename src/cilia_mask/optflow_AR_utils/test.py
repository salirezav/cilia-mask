import argparse
import joblib as jb
from joblib import Parallel, delayed
import json
import numpy as np
import os.path
import sklearn.metrics as skm

import matplotlib
matplotlib.use("Agg")

def predict(h, prefix, Xp, yp, outdir):
    """
    Run a model prediction on a subset of the data.

    Parameters
    ----------
    h : estimator
        The trained estimator.
    prefix : string
        The prefix representing a single video instance.
    Xp : array, shape (patches, features)
    yp : array, shape (patches,)
        The data relevant to the specific video.
    outdir : string
        Output directory for individual video reconstructions.
    
    Returns
    -------
    prefix : string
        The video prefix, so we can parallelize this.
    y_hat : array, shape (patches,)
        Predicted data from the trained model.
    """
    yhat = h.predict(Xp)
    return (prefix, yhat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '#5: Test',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python test.py <args>')
    parser.add_argument('-i', '--inputdir', required = True,
        help = "Input directory where X and y are found.")
    parser.add_argument('-f', '--flow_type', choices = ['nvidia', 'opencv'], 
        required = True, help = "Type of optical flow features to use.")
    parser.add_argument('-m', '--model', required = True,
        help = "Path to a trained model to use for running inference.")
    parser.add_argument('-o', '--output', required = True,
        help = "Path where output reconstructions will happen for each video.")
    
    parser.add_argument("--n_jobs", required = False, default = 1, type = int,
        help = "Number of parallel jobs to run. -1 uses all cores. [DEFAULT: 1]")
    
    args = vars(parser.parse_args())

    # Load the trained model.
    h = jb.load(args['model'])

    # Load the testing dataset.
    ft = args['flow_type']
    with open(os.path.join(args['inputdir'], f"i_{ft}_test.json"), "r") as fp:
        mappings = json.loads(fp.read())
    X = np.load(os.path.join(args['inputdir'], f"X_{ft}_test.npy"))
    y = np.load(os.path.join(args['inputdir'], f"y_{ft}_test.npy"))
    
    # Test the model!
    p = Parallel(n_jobs = args['n_jobs'], verbose = 30)
    out = p(delayed(predict)(
        h, prefix, X[indices[0]:(indices[0] + indices[1])], 
        y[indices[0]:(indices[0] + indices[1])], args['output']
    ) for (prefix, indices) in mappings.items())
    
    # Let's look at some global metrics.
    #yhat = h.predict(X)
    print(f"Score: {h.score(X, y)} (higher is better)")
    #print(f"Explained var: {skm.explained_variance_score(y, yhat)} (higher is better)")
    #print(f"Max error: {skm.max_error(y, yhat)} (lower is better)")
    #print(f"Mean absolute error: {skm.mean_absolute_error(y, yhat)} (lower is better)")
    y_score = h.decision_function(X)
    p, r, _ = skm.precision_recall_curve(y, y_score)
    display = skm.PrecisionRecallDisplay(p, r)
    display.plot()
    display.figure_.savefig(f"precision_recall_{args['flow_type']}.png")
    
    fpr, tpr, _ = skm.roc_curve(y, y_score)
    auc = skm.roc_auc_score(y, y_score)
    display = skm.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = auc)
    display.plot()
    display.figure_.savefig(f"roc_auc_{args['flow_type']}.png")

    # Let's look at individual videos.
