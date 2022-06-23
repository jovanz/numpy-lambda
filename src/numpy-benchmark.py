#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Roughly based on: http://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration

import numpy as np
from time import time
from datetime import datetime

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

def handler(event, context):
  # Matrix multiplication
  N = 20
  start_time = datetime.now()
  for i in range(N):
      np.dot(A, B)
  time_elapsed = datetime.now() - start_time
  print(f'Dotted two {size} x {size} matrices in (h:mm:ss.ms) {time_elapsed}')
  del A, B

  # Vector multiplication
  N = 5000
  start_time = datetime.now()
  for i in range(N):
      np.dot(C, D)
  time_elapsed = datetime.now() - start_time
  print(f'Dotted two vectors of length {size * 128} in (h:mm:ss.ms) {time_elapsed}')
  del C, D

  # Singular Value Decomposition (SVD)
  N = 3
  start_time = datetime.now()
  for i in range(N):
      np.linalg.svd(E, full_matrices = False)
  time_elapsed = datetime.now() - start_time
  print(f'SVD of a {size / 2} x {size / 4} matrix in (h:mm:ss.ms) {time_elapsed}')
  del E

  # Cholesky Decomposition
  N = 3
  start_time = datetime.now()
  for i in range(N):
      np.linalg.cholesky(F)
  time_elapsed = datetime.now() - start_time
  print(f'Cholesky decomposition of a {size / 2} x {size / 2} matrix in (h:mm:ss.ms) {time_elapsed}')

  # Eigendecomposition
  start_time = datetime.now()
  for i in range(N):
      np.linalg.eig(G)
  time_elapsed = datetime.now() - start_time
  print(f'Eigendecomposition of a {size / 2} x {size / 2} matrix in (h:mm:ss.ms) {time_elapsed}')

  print('')
  print('This was obtained using the following Numpy configuration:')
  np.__config__.show()
