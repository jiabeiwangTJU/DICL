# Written by Yixiao Ge

import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from .finch import FINCH
from .compute_dist import build_dist
from time import time
from numba import njit, prange

__all__ = ["label_generator_dbscan_context_single", "label_generator_context_dbscan"]

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

##########cluster using finch############
@torch.no_grad()
def label_generator_dbscan_context_single_finch(cfg, features, all_inds, **kwargs):#

    use_outliers = cfg.PSEUDO_LABELS.use_outliers 
    thres_dist = cfg.PSEUDO_LABELS.thres_dist  

    c, num_clust, req_c = FINCH(features.numpy(), all_inds, thres_dist, distance='euclidean') 
    labels = c
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters


##########cluster using DBSCAN############
@torch.no_grad()
def label_generator_dbscan_context_single(cfg, features, dist, eps, **kwargs):
    # assert isinstance(dist, np.ndarray)

    min_samples = cfg.PSEUDO_LABELS.min_samples
    use_outliers = cfg.PSEUDO_LABELS.use_outliers


    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters


##########Enlarge the similarity of the persons in the same image#########
#########Experiments show it helps the finch cluster but is helpless to dbscan cluster.######  
@njit(parallel=True)
def compute_new_dist(dist, all_inds, thres_dist):
    for uid in prange(11205):  #go through all images
        for i in prange(uid + 1, 11206):
            tmp_id = (all_inds == uid).nonzero()[0] 
            next_id = (all_inds == i).nonzero()[0] 
            temp_dist = np.empty((tmp_id.shape[0], next_id.shape[0]))
            for m in prange(tmp_id.shape[0]): 
                for n in prange(next_id.shape[0]):
                    temp_dist[m, n] = dist[tmp_id[m], next_id[n]]
            for q in prange(tmp_id.shape[0]): 
                for k in prange(next_id.shape[0]):
                    dist[tmp_id[q], next_id[k]] = dist[tmp_id[q], next_id[k]] + thres_dist * temp_dist.max()
                    dist[next_id[k], tmp_id[q]] = dist[tmp_id[q], next_id[k]]
    return dist

def list_duplicates(seq):
    tally = collections.defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    dups = [(key,locs) for key,locs in tally.items() if len(locs)>1]
    return dups

@torch.no_grad()
def process_label_with_context(labels, centers, features, inds, num_classes):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    N_c = centers.shape[0]
    print('centers_before', N_c)
    assert num_classes == N_c
    assert N_p == labels.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    #print(unique_inds)
    #print(inds)
    for uid in unique_inds:
        b = inds == uid
        tmp_id = b.nonzero()
        tmp_labels = labels[tmp_id]
        dups = list_duplicates(list(tmp_labels.squeeze(1).cpu().numpy()))  ###(key, locs)
        if len(dups) > 0:
            for dup in dups:

                tmp_center = centers[dup[0]].cpu().numpy()
                tmp_features = features[tmp_id[dup[1]].squeeze(1)].cpu().numpy()
                # print(tmp_id[dup[1]].squeeze(1), tmp_features.shape)
                sim = np.dot(tmp_center, tmp_features.transpose())
                #print(sim)
                idx = np.argmax(sim)
                # print(idx, len(sim), sim.shape)
                for i in range(len(sim)):
                    if i != idx:
                        labels[tmp_id[dup[1][i]]] = num_classes
                        centers = torch.cat((centers, features[tmp_id[dup[1][i]]]))
                        num_classes += 1
    assert num_classes == centers.shape[0]
    print('centers_after', num_classes)
    return labels, centers, num_classes

@torch.no_grad()
def label_generator_dbscan_context(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):
    assert cfg.PSEUDO_LABELS.cluster == "dbscan_context"

    if not cuda:
        cfg.PSEUDO_LABELS.search_type = 3


    # clustering
    eps = cfg.PSEUDO_LABELS.eps


    if len(eps) == 1:

        features = features.cpu()
        labels, centers, num_classes = label_generator_dbscan_context_single_finch(
            cfg, features, all_inds)  #

        if all_inds is not None:
            labels, centers, num_classes = process_label_with_context(labels, centers, features, all_inds, num_classes)
        return labels, centers, num_classes, indep_thres

    else:

        # compute distance matrix by features
        dist = build_dist(cfg.PSEUDO_LABELS, features, verbose=True)

        # begin_time = time()
        # dist = torch.from_numpy(dist).cuda()
        # torch.cuda.empty_cache()
        # thres_dist = cfg.PSEUDO_LABELS.thres_dist
        # dist = compute_new_dist(dist, all_inds.cpu().numpy(), thres_dist=thres_dist)
        # # dist = dist.cpu().numpy()
        # end_time = time()
        # print('time', (end_time - begin_time) / 3600)

        features = features.cpu()

        assert (
            len(eps) == 3
        ), "three eps values are required for the clustering reliability criterion"

        print("adopt the reliability criterion for filtering clusters")
        eps = sorted(eps)
        labels_tight, centers_tight, num_classes_tight = label_generator_dbscan_context_single(cfg, features, dist, eps[0])
        labels_normal, centers_normal, num_classes = label_generator_dbscan_context_single(
            cfg, features, dist, eps[1]
        )
        labels_loose, centers_loose, num_classes_loose = label_generator_dbscan_context_single(cfg, features, dist, eps[2])
        #print("num_classes", num_classes_tight, num_classes, num_classes_loose)
        #print(labels_tight.max(), labels_normal.max(), labels_loose.max())

        labels_tight, _, num_classes_tight = process_label_with_context(labels_tight, centers_tight, features, all_inds, num_classes_tight)
        labels_normal, _, num_classes = process_label_with_context(labels_normal, centers_normal, features, all_inds, num_classes)
        labels_loose, _, num_classes_loose = process_label_with_context(labels_loose, centers_loose, features, all_inds, num_classes_loose)
        #print("num_classes", num_classes_tight, num_classes, num_classes_loose)
        #print(labels_tight.max(), labels_normal.max(), labels_loose.max())
        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = (
            labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float()
        )
        label_sim_tight = (
            labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float()
        )
        label_sim_loose = (
            labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float()
        )

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
            label_sim, label_sim_tight
        ).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(
            label_sim, label_sim_loose
        ).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        cluster_img_num = collections.defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [
            min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
        ]
        cluster_R_indep_noins = [
            iou
            for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
            if cluster_img_num[num] > 1
        ]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(
                    len(cluster_R_indep_noins) - 1,
                    np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                )
            ]

        labels_num = collections.defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = collections.defaultdict(list)
        outliers = 0
        #print(cluster_R_indep)
        print(len(cluster_R_indep), num_classes)  ###48858 49310
        
        for i, label in enumerate(labels_normal):
            label = label.item()
            #print(label)
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not cfg.PSEUDO_LABELS.use_outliers, "exists a bug"
                continue
            if (indep_score > indep_thres) or (
                comp_score.item() > cluster_R_comp[label]
            ):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres