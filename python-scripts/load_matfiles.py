#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:35:40 2021

@author: gustaf
"""

import scipy.io as IO
A = IO.loadmat('/home/gustaf/nsm_data/mixed data benchmarks barbora/collection_D11.mat')

A['collection']['N'][0][0][0]
A['collection']['D_eff'][0][0][0]