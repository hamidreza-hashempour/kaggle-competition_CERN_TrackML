import numpy as np
import pandas as pd
import timeit
import multiprocessing
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from trackml.dataset import load_event, load_dataset
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs


def find_labels(params):
    hits, dh = params
    a = hits['phi'].values
    z = hits['z'].values
    zr = hits['zr'].values
    dis = a + np.sign(z) * dh * z

    f0 = np.cos(dis)
    f1 = np.sin(dis)
    f2 = zr
    X = StandardScaler().fit_transform(np.column_stack([f0, f1, f2]))

    _, l = dbscan(X, eps=0.0045, min_samples=1, n_jobs=4)
    return l + 1

def add_count(l):
    unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
    c = count[reverse]
    c[np.where(l == 0)] = 0
    c[np.where(c > 20)] = 0
    return (l, c)

def do_dbscan_predict(hits):
    start_time = timeit.default_timer()

    hits['r'] = np.sqrt(hits['x'] ** 2 + hits['y'] ** 2)
    hits['zr'] = hits['z'] / hits['r']
    hits['phi'] = np.arctan2(hits['y'], hits['x'])

    params = []
    for j in range(0, 20):
        dd = j * 0.0001
        params.append((hits, dd))
        if j > 0:
             params.append((hits, -dd))
    for j in range(20, 60):
        dd = j * 0.0001
        if j % 2 == 0:
            params.append((hits, dd))
        else:
             params.append((hits, -dd))
             
    pool = Pool(processes=4)
    LFAS = pool.map(find_labels, params)
    res = [add_count(q) for q in LFAS]
    pool.close()

    labels, cnt = res[0]
    for j in range(1, len(res)):
        qq, zz = res[j]
        idx = np.where((zz - cnt > 0))[0]
        labels[idx] = qq[idx] + labels.max()
        cnt[idx] = zz[idx]



    return labels

def DBScan():
    data_dir = 'path'

    idd = ['***']
    sum = 0
    sum_score = 0
    for i, eve_id in enumerate(idd):
        hits, cells, particles, truth = load_event(data_dir + '/event' + eve_id)
        labels = do_dbscan_predict(hits)
        submission = create_one_event_submission(0, hits['hit_id'].values, labels)
        score = score_event(truth, submission)
        print('[%2d] score : %0.8f' % (i, score))
        sum_score += score
        sum += 1
