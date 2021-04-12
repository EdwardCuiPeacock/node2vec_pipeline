#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:40:11 2021

@author: edwardcui
"""
import gc
#from line_profiler import LineProfiler
#from memory_profiler import profile as mem_profile
#import tracemalloc
#from collections import OrderedDict
#snapshots = OrderedDict()
#import time

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#import networkx  # simulation only


def tf_sparse_multiply(a: tf.SparseTensor, b: tf.SparseTensor):
    a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        a.indices, a.values, a.dense_shape)

    b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        b.indices, b.values, b.dense_shape)

    c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
        a=a_sm, b=b_sm, type="float64")

    c = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
        c_sm, "float64")

    return tf.SparseTensor(
        c.indices, c.values, dense_shape=c.dense_shape)

    
def preproc_W_tf(W, symmetrify=True):
    W = tf.cast(W, "float64")
    if symmetrify:
        W = tf.sparse.maximum(W, tf.sparse.transpose(W)) # symmetrify
    
    if not bool(tf.reduce_all(tf.sparse.reduce_max(W, axis=1) > 0)):
        indices = tf.sparse.to_dense(
            tf.sets.difference(
                [tf.range(W.shape[0], dtype="int64")], [tf.unique(W.indices[:, 0]).y]
            )
        )
        indices = tf.transpose(tf.concat([indices, indices], axis=0))
        terms = tf.sparse.SparseTensor(
            indices, tf.ones(indices.shape[0], dtype="float64"), dense_shape=W.shape
        )
        W = tf.sparse.add(W, terms)
    return W


def sample_from_sparse_tf(W_sample, seed=None, DEBUG=True):
    """Take a sample given unnormalized weight matrix."""
    # Make sure the minimum value of the sampling weight matrix is not zero
    epsilon=1E-7
    W_sample = tf.sparse.SparseTensor(W_sample.indices, 
                                      tf.clip_by_value(W_sample.values, 
                                                       epsilon, 
                                                       tf.float32.max),
                                      W_sample.shape)
    
    if DEBUG:
        check = bool(tf.reduce_min(W_sample.values) > 0)
        logging.info(f"All W_sample values are positive before normalization: {check}")
    
    # Normalize each row
    row_sum = tf.sparse.reduce_sum(W_sample, axis=1, keepdims=False)
    normalized_values = tf.divide(W_sample.values, tf.gather(row_sum, W_sample.indices[:, 0]))
    W_sample = tf.sparse.SparseTensor(W_sample.indices, normalized_values, W_sample.shape)
    W_sample = tf.sparse.reorder(W_sample) # Make sure the indices are sorted
    
    if DEBUG:
        check = bool(tf.reduce_min(W_sample.values) > 0)
        logging.info(f"All W_sample values are positive after normalization: {check}")
        
        check = float(tf.reduce_min(tf.sparse.reduce_sum(W_sample, axis=1)))
        logging.info(f"Min value of row sum after normalization (expected to be 1) {check}")
            
    # Compute the CDF row-wise
    sample_values = tf.cumsum(W_sample.values) - tf.cast(W_sample.indices[:, 0], "float64")
    cdf = tf.sparse.SparseTensor(W_sample.indices, sample_values, dense_shape=W_sample.shape)
    cdf = tf.sparse.reorder(cdf)
    
    if DEBUG:
        check = bool(tf.reduce_all(tf.sparse.reduce_max(cdf, axis=1) > 0.999))
        logging.info(f"All cdf rows cumulative of 1: {check}")
    
    # Use Inverse Trasnform sampling on the sparse matrix
    values_random = tf.random.uniform((cdf.shape[0], ), minval=0, maxval=0.999, dtype="float64", seed=seed)
    sample_values = cdf.values - tf.gather(values_random, cdf.indices[:, 0])
    cdf = tf.sparse.SparseTensor(cdf.indices, sample_values, dense_shape=cdf.shape)
    cdf = tf.sparse.reorder(cdf)
    
    if DEBUG:
        missing_rows = tf.sparse.to_dense(
                tf.sets.difference(
                    [tf.range(W_sample.shape[0], dtype="int64")], [tf.unique(cdf.indices[:, 0]).y]
                )
            ).numpy().tolist()
        
        logging.info(f"random df missing rows: {missing_rows}")
        
        check = bool(tf.reduce_all(tf.sparse.reduce_max(cdf, axis=1) > 0))
        logging.info(f"All random cdf rows have some positive values: {check}")
    
    # Remove negative values
    is_pos = tf.greater_equal(cdf.values, -epsilon)
    cdf_sample = tf.sparse.retain(cdf, is_pos)
    cdf_sample = tf.sparse.reorder(cdf_sample)
    
    if DEBUG:
        missing_rows = tf.sparse.to_dense(
               tf.sets.difference(
                   [tf.range(W_sample.shape[0], dtype="int64")], [tf.unique(cdf_sample.indices[:, 0]).y]
               )
           ).numpy().tolist()
         
        logging.info(f"cdf_sample missing rows: {missing_rows}")
    
    # Materialize the samples: Take the first nonzero col of each row
    # s_next = tf.constant([list(item)[0][1] for _, item in \
    #    itertools.groupby(cdf_sample.indices.numpy(), lambda x: x[0])])
    # Casting to csr matrix
    index = cdf_sample.indices # assuming sorted already
    indices = tf.concat([tf.constant([1], dtype="int64"), 
                         index[1:, 0] - index[:-1, 0]], axis=0)
    s_next = index[:, 1][tf.greater(indices, 0)]
    
    logging.info(f"s_size={len(s_next)} vs. W_size={W_sample.shape[0]}")
    
    return W_sample, cdf, cdf_sample, s_next

def random_walk_sampling_step_tf(W, s0, s1, p, q, seed=None):
    # Get dimension
    num_nodes = W.shape[0]

    # alpha_1 / P
    P = tf.sparse.SparseTensor(tf.cast(tf.stack([tf.range(num_nodes, dtype="int64"), s0], axis=1), dtype="int64"), 
                               tf.ones(num_nodes, dtype="float64"), 
                               dense_shape=(num_nodes, num_nodes))
    # alpha_2 / R
    A_0 = tf.sparse.SparseTensor(W.indices, tf.ones_like(W.values, dtype="float64"), dense_shape=W.shape)
    A_i_1 = tf_sparse_multiply(P, A_0)

    I = tf.sparse.SparseTensor(tf.cast(tf.stack([tf.range(num_nodes, dtype="int64"), s1], axis=1), "int64"), 
                               tf.ones(num_nodes, dtype="float64"), 
                               dense_shape=(num_nodes, num_nodes)) # permutation matrix
    A_i = tf_sparse_multiply(I, A_0)

    ## intersection
    R = tf.sparse.minimum(A_i_1, A_i)
    is_nonzero = tf.not_equal(R.values, 0)
    R = tf.sparse.retain(R, is_nonzero)

    # alpha3 / Q
    Q = tf.sparse.add(A_i, P.__mul__(tf.constant([-1], dtype="float64")))
    Q = tf.sparse.add(Q, R.__mul__(tf.constant([-1], dtype="float64")))
    is_nonzero = tf.not_equal(Q.values, 0)
    Q = tf.sparse.retain(Q, is_nonzero)

    # Combine to get the final weight
    W_sample = tf.sparse.add(P.__mul__(tf.constant([1/p], dtype="float64")), R)
    W_sample = tf.sparse.add(W_sample, Q.__mul__(tf.constant([1/q], dtype="float64")))
    is_nonzero = tf.not_equal(W_sample.values, 0)
    W_sample = tf.sparse.retain(W_sample, is_nonzero)
    W_sample = tf.sparse.reorder(W_sample)

    # Make sure the orders of indices are the same
    W_new = tf_sparse_multiply(I, tf.cast(tf.sparse.reorder(W), dtype="float64"))
    W_new = tf.sparse.reorder(W_new)

    # Multiply the weights by creating a new sparse matrix
    W_sample = tf.sparse.SparseTensor(W_sample.indices, 
                                      W_sample.values * W_new.values,
                                      dense_shape=W_sample.shape)

    # Taking samples from the sparse matrix
    W_sample, cdf, cdf_sample, s_next = sample_from_sparse_tf(W_sample, seed=seed)
    
    return W_sample, cdf, cdf_sample, s_next


def sample_1_iteration_tf(W, p, q, walk_length=80, symmetrify=True, seed=None):
    W = preproc_W_tf(W, symmetrify)
    
    checks = bool(tf.reduce_all(tf.sparse.reduce_max(W, axis=1) > 0))
    print(f"All rows have something: {checks}")
    
    # make sure the indices are sorted
    W = tf.sparse.reorder(W)

    # First step
    s0 = tf.range(W.shape[0], dtype="int64")
    _, _, _, s1 = sample_from_sparse_tf(W, seed=seed)
    S = [s0, s1]
                  
    #print(f"check length: s0={len(s0)}, s1={len(s1)}")

    for i in range(walk_length - 1):
        _, _, _, s_next = random_walk_sampling_step_tf(
            W, S[-2], S[-1], p, q, seed=(seed + i + 1) if seed is not None else None
        )
        S.append(s_next)

    # for ii, ss in enumerate(S):  # verbose print
    #     logging.info(f"s{ii}: {ss}")

    return S



def test_random_walk_sampling_step_tf():
    """Test tensorflow.sparse implementation of random walk."""
    p = 0.2
    q = 0.8
    df = pd.DataFrame({"content":[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], 
                       "cast":[1, 2, 3, 4, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3], 
                       "weight":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    W = tf.sparse.SparseTensor(df[["content", "cast"]].values,
                               df["weight"].values, dense_shape=(5,5))
    W = tf.cast(W, "float64")
    s0 = tf.constant([3, 5, 2, 1, 4], dtype="int64")-1
    s1 = tf.constant([2, 4, 1, 5, 3], dtype="int64")-1  
    W_sample, cdf, cdf_sample, s_next = random_walk_sampling_step_tf(W, s0, s1, p, q)
    
    #print(tf.sparse.to_dense(W_sample))
    #print(s_next)
    
    assert W_sample.shape[0] == 5
    assert len(s_next) == 5
    assert np.all(np.array([np.where(x)[0][0] for x in \
                            tf.sparse.to_dense(cdf_sample).numpy()]) == s_next.numpy())
        


def test_sample_1_iteration_tf():
    """Test numpy / scipy.sparse implementation of sample_1_iteration."""
    #df = pd.DataFrame({"content":[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], 
    #                "cast":[0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3], 
    #                "weight":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    
    num_nodes = int(1E3)
    num_entry = int(1E4)
    walk_length = 3
    rs = np.random.RandomState(0)
    df = pd.DataFrame({"content": rs.randint(0, num_nodes-1, size=num_entry),
                       "cast": rs.randint(0, num_nodes-1, size=num_entry),
                       "weight": rs.rand(num_entry)})
    W_sample= tf.sparse.SparseTensor(df[["content", "cast"]].values,
                               df["weight"].values, dense_shape=(num_nodes, num_nodes))
    W_sample = preproc_W_tf(W_sample)
    
    S = sample_1_iteration_tf(W_sample, p=0.2, q=0.8, walk_length=walk_length, seed=0)
    
    assert len(S) - 1 == walk_length
    assert np.all(np.array(list(map(len, S))) == num_nodes)

# %%
def numpy_slice_sparse_matrix(A, s, default_value=True, default_dtype=bool):
    """
    A: csr sparse matrix
    s: row indices
    default_value: default the new sparse matrix with this single value
    """
    #snapshot.append(tracemalloc.take_snapshot()) # at start
    A = A.tocsr()
    start_index = np.take(A.indptr, s)
    end_index = np.take(A.indptr, s+1)
    new_indptr = np.concatenate([[0], np.cumsum(end_index - start_index)])
    #snapshot.append(tracemalloc.take_snapshot()) # after finding out start end indices
    # This could still be efficient if the slice length is not very long
    # Hence efficient for sparse matrices
    new_indices = np.concatenate([A.indices[i:j] for (i, j) in zip(start_index, end_index)])
    #snapshot.append(tracemalloc.take_snapshot()) # after finding new indices
    if default_value is None:
        # Use original value
        new_values = np.concatenate([A.data[i:j] for (i, j) in zip(start_index, end_index)])
    else:
        new_values = default_value * np.ones_like(new_indices, dtype=default_dtype)
    #snapshot.append(tracemalloc.take_snapshot()) # after finding new values

    # free some memory
    del start_index
    del end_index 
    new_A = csr_matrix((new_values, new_indices, new_indptr))
    #snapshot.append(tracemalloc.take_snapshot()) # after making the new sparse matrix
    return new_A




def sample_from_sparse_numpy(W_sample, seed=None, DEBUG=False):
    epsilon = np.finfo(np.float32).eps
    W_sample.data = np.clip(W_sample.data, epsilon, np.finfo(np.float32).max)
    
    num_nodes = W_sample.shape[0]
    # Normalize for each row
    row_sum = np.asarray(W_sample.sum(axis=1)).ravel() # dense
    W_sample = W_sample.tocoo()
    W_sample.data /= np.take(row_sum, W_sample.row)
    
    if DEBUG:
        check = np.min(W_sample.sum(axis=1))
        logging.info(f"Sum of row is around 1: {check}")
    
    # Compute cdf cumsum with csr matrix
    cdf = W_sample.copy().tocsr() # ordered by row
    cdf.data = np.cumsum(cdf.data)
    
    cdf = cdf.tocoo()
    # Subtract each row by broadcasting
    cdf.data = cdf.data - cdf.row
    
    if DEBUG:
        check = np.min(cdf.max(axis=1).todense())
        logging.info(f"cdf each row has greatest value around 1: {check}")
    
    # Take the sample
    rs = np.random.RandomState(seed)
    uniform_sample = rs.uniform(low=0.0, high=0.999, size=num_nodes)  # [0, 1)
    cdf.data -= np.take(uniform_sample, cdf.row)
    # remove any negative
    samp_ind = cdf.data >= -epsilon
    cdf.data = cdf.data[samp_ind]
    cdf.row = cdf.row[samp_ind]
    cdf.col = cdf.col[samp_ind]
    
    if DEBUG:
        check = np.all(cdf.getnnz(axis=1) > 0)
        logging.info(f"random cdf has at least 1 entry: {check}")
    
    # Slice out the column indices: starting index of each row
    cdf = cdf.tocsr()
    s_next = cdf.indices[cdf.indptr[:-1]]
    
    logging.info(f"s_size={len(s_next)} vs. W_size={W_sample.shape[0]}")
    
    return W_sample, cdf, s_next
    
#@mem_profile
def random_walk_sampling_step_numpy(W, s0, s1, p, q, seed=None, DEBUG=False):
    """Take 1 step of the random walk, with numpy / scipy.sparse."""
    W.data = W.data.astype(np.float32)
    if DEBUG:
        logging.info("Start random walk")
        logging.info(f"s0={s0.shape}")
        logging.info(f"s1={s1.shape}")
    num_nodes = W.shape[0]
    # alpha_1 / P
    #tracemalloc.start()
    #snapshots["Begin"] = tracemalloc.take_snapshot()
    P = coo_matrix((np.ones(num_nodes, dtype=bool), 
                    (np.arange(num_nodes), s0)), 
                   shape=(num_nodes, num_nodes),
                  ).tocsr()
    #snapshots["After creating P"] = tracemalloc.take_snapshot()

    # alpha_2 / R
    A_0 = W.copy().tocsr()
    A_0.data = np.ones_like(A_0.data, dtype=bool)
    #snapshots["After creating A_0"] = tracemalloc.take_snapshot()
    
    A_i = A_0[s1, :]
    #snapshots["After creating A_i"] = tracemalloc.take_snapshot()

    A_i_1 = A_0[s0, :]
    #snapshots["After creating A_i_1"] = tracemalloc.take_snapshot()
    R = A_i.multiply(A_i_1) # elementwise multiply
    #snapshots["After creating R"] = tracemalloc.take_snapshot()
    
    if DEBUG:
        logging.info(f"Shape: A_i={A_i.shape}")
        logging.info(f"Shape: P={P.shape}")
        logging.info(f"Shape: R={R.shape}")
    
    # alpha_3 / Q
    Q = A_i - P - R
    #snapshots["After creating Q"] = tracemalloc.take_snapshot()
    del A_i_1 # free some memory
    del A_i # free some memory
    del A_0 # free some memory
    if DEBUG:
        logging.info(f"Shape: Q={Q.shape}")
    #snapshots["After deleting A_i, A_0"] = tracemalloc.take_snapshot()
    # Combine to get the final weight
    W_i = W.tocsr()[s1, :]
    #snapshots["After creating W_i"] = tracemalloc.take_snapshot()
    W_sample = ((1/p) * P + R + (1/q) * Q).multiply(W_i)
    #snapshots["After create W_sample"] = tracemalloc.take_snapshot()
    #print(W_sample.toarray())
    del P
    del Q
    del R
    #snapshots["After deleting P, Q, R"] = tracemalloc.take_snapshot()
    # Make sure the rows are sorted
    W_sample, cdf, s_next = sample_from_sparse_numpy(W_sample, seed=None)
    #snapshots["After taking sample"] = tracemalloc.take_snapshot()
    
    return W_sample, cdf, s_next

def preproc_W_numpy(W, symmetrify=True):
    W = W.astype("float64")
    if symmetrify:
        W = W.maximum(W.transpose()).tocoo()
    # Make sure each row has at least 1 entry
    indices = W.getnnz(axis=1) < 1
    if np.sum(indices) > 0:
        indices = np.where(indices)[0]
        W.row = np.concatenate([W.row, indices], axis=0)
        W.col = np.concatenate([W.col, indices], axis=0)
        W.data = np.concatenate([W.data, np.ones_like(indices)], axis=0)
        
    return W

def sample_1_iteration_numpy(W, p, q, walk_length=80, symmetrify=True, seed=None):
    W = preproc_W_numpy(W, symmetrify)

    # First step
    s0 = np.arange(W.shape[0])
    W_sample_1, cdf_1, s1 = sample_from_sparse_numpy(W, seed=seed)
    gc.collect()
    S = [s0, s1]

    for i in range(walk_length - 1):
        _, _, s_next = random_walk_sampling_step_numpy(
            W, S[-2], S[-1], p, q, seed=(seed + 1 + i) if seed is not None else None
        )
        S.append(s_next)
        gc.collect()

    # for ii, ss in enumerate(S):  # verbose print
    #     logging.info(f"s{ii}: {ss}")

    return S


def test_random_walk_sampling_step_numpy():
    """Test numpy / scipy.sparse implementation of random walk."""
    df = pd.DataFrame({"content":[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], 
                    "cast":[1, 2, 3, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3], 
                    "weight":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    W = coo_matrix((df["weight"].values, (df["content"].values, 
                                          df["cast"].values)), 
                   dtype=np.float32, shape=(5,5))
    s0 = np.array([3, 5, 2, 1, 4])-1
    s1 = np.array([2, 4, 1, 5, 3])-1
    W_sample, cdf , s_next= random_walk_sampling_step_numpy(W, s0, s1, p=0.2, q=0.8)
    
    # print(cdf.toarray())
    # print(s_next)
    assert W_sample.shape[0] == 5
    assert len(s_next) == 5
    assert np.all(np.array([np.where(x)[0][0] for x in cdf.toarray()]) == s_next)
    
    
def test_sample_1_iteration_numpy():
    """Test numpy / scipy.sparse implementation of sample_1_iteration."""
    num_nodes = int(1E3)
    num_entry = int(1E4)
    walk_length = 30
    rs = np.random.RandomState(None)
    df = pd.DataFrame({"content": rs.randint(0, num_nodes-1, size=num_entry),
                       "cast": rs.randint(0, num_nodes-1, size=num_entry),
                       "weight": rs.rand(num_entry)})
    
    W = coo_matrix((df["weight"].values, (df["content"].values, 
                                          df["cast"].values)), 
                   dtype=np.float64, shape=(num_nodes, num_nodes))
    W = W.tocsc().tocoo()
    W = preproc_W_numpy(W)
    
    S = sample_1_iteration_numpy(W, p=0.2, q=0.8, walk_length=walk_length, seed=None)
    #print(S)
    
    
def test_permutation_matrix():
    """Verify repeated row slice can be achieved with permutation matrix."""
    A = np.array([[1, 2, 3, 4, 5]]).T - 1
    I = np.eye(5)
    I = I[[0, 1, 1, 4, 3], :]
   
    print(I @ A)
    
    
if __name__ == '__main__':
    #test_sample_1_iteration_tf()
    #test_sample_1_iteration_numpy()
    import json
    df = json.load(open("/Users/edwardcui/Downloads/output_pandas_20210410.json", "r"))
    df = pd.DataFrame(df)
    df["InSeasonSeries_Id"] = df["InSeasonSeries_Id"].astype(int)
    df["token"] = df["token"].astype(int)
    df = df.loc[(df["InSeasonSeries_Id"] > -0.5) & (df["token"] >-0.5), :]
    
    num_nodes = int(np.max(df.values) + 1)
    walk_length = 3
    
    W = coo_matrix((df["weight"].values, (df["InSeasonSeries_Id"].values, 
                                           df["token"].values)), 
                    dtype=np.float64, shape=(num_nodes, num_nodes))
    W = W.tocsc().tocoo()
    W = preproc_W_numpy(W)
    
    
    s0 = np.arange(W.shape[0])
    W_sample_1, cdf_1, s1 = sample_from_sparse_numpy(W, seed=42)
    S = [s0, s1]
    
    random_walk_sampling_step_numpy(W, s0, s1, p=0.2, q=0.8, seed=42)
    
    # lp = LineProfiler()
    # lp_wrapper = lp(random_walk_sampling_step_numpy)
    # lp_wrapper(W, s0, s1, p=0.2, q=0.8, seed=42)
    # lp.print_stats()
    
    
    S = sample_1_iteration_numpy(W, p=0.2, q=0.8, walk_length=walk_length, seed=42)
    #%%
    # j, i = 9, 8
    # top_stats = list(snapshots.values())[j].compare_to(list(snapshots.values())[i], 'lineno')
    # key_j, key_i = list(snapshots.keys())[j], list(snapshots.keys())[i]
    # print(f"Between {key_j} vs. {key_i}")
    # print("Top 10 diferences")
    # for stat in top_stats[:10]:
    #     print(stat) 
   

    
    