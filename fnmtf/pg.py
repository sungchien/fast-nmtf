#!/usr/bin/env python

import numpy as np
import scipy.linalg as la

from common import *

@tri_factorization
def nmtf_pg(engine, X, Xt, U, S, V, TrX, k=20, k2=20, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    
    for it in range(max_iter):
        timer.split('XV')
        XV = bigdot(X, V)
        timer.split('error')
        NK15 = dot(XV, S.T)
        KK16 = dot(NK15.T, U)
        tr2 = trace(KK16)
        KM19 = dot(S, V.T)
        KK16 = dot(KM19, KM19.T)
        NK20 = dot(U, KK16)
        KK16 = dot(U.T, NK20)
        tr3 = trace(KK16)
        E = 1 - (2*tr2 - tr3)/TrX
        err_history.append(E)
        
        if verbose:
            print("Error", E)
        
        timer.split('U')
        NK22 = divide(U, NK20)
        NK23 = multiply(NK22, NK15)
        Pu = sub(U, NK23)
        
        NK22 = sub(NK20, NK15)
        NK20 = multiply(Pu, NK22)
        AA18 = norm1(NK20)
        KL24 = dot(KM19, V)
        KK16 = dot(Pu.T, Pu)
        LK25 = dot(S.T, KK16)
        KK16 = dot(KL24, LK25)
        AA17 = trace(KK16)
        nu = divide(AA18, AA17)
        
        NK23 = multiply(Pu, nu)
        NK20 = sub(U, NK23)
        U = project(NK20)
        
        timer.split('Xt')
        MK28 = bigdot(Xt, U)
        timer.split('V')
        NL14 = dot(U, S)
        KL24 = dot(U.T, NL14)
        ML26 = dot(KM19.T, KL24)
        ML27 = divide(V, ML26)
        ML29 = dot(MK28, S)
        ML30 = multiply(ML27, ML29)
        Pv = sub(V, ML30)
        
        ML30 = sub(ML26, ML29)
        ML26 = multiply(Pv, ML30)
        AA17 = norm1(ML26)
        LL31 = dot(Pv.T, Pv)
        
        KL32 = dot(S, LL31)
        KK16 = dot(KL32, KL24.T)
        AA18 = trace(KK16)
        nv = divide(AA17, AA18)
        
        ML29 = multiply(Pv, nv)
        ML26 = sub(V, ML29)
        V = project(ML26)
        
        timer.split('S')
        KK16 = dot(U.T, U)
        LL31 = dot(V.T, V)
        KL32 = dot(KK16, S)
        KL24 = dot(KL32, LL31)
        KL32 = divide(S, KL24)
        KL33 = dot(MK28.T, V)
        KL34 = multiply(KL32, KL33)
        Ps = sub(S, KL34)
        
        KL32 = sub(KL24, KL33)
        KL33 = multiply(Ps, KL32)
        AA17 = norm1(KL33)
        KL33 = dot(KK16, Ps)
        LK25 = dot(LL31, Ps.T)
        KK16 = dot(KL33, LK25)
        AA18 = trace(KK16)
        ns = divide(AA17, AA18)
        
        KL34 = multiply(Ps, ns)
        KL33 = sub(S, KL34)
        S = project(KL33)
    
    print("Timer", str(timer))
    factors = U, S, V
    return factors, err_history