# Simple class for learning an alignment of strings, MED-style.
# Weights are learned by a Chinese Restaurant Process sampler
# that weights single alignments x:y in proportion to how many times
# such an alignment has been seen elsewhere out of all possible alignments.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Usage:
# Align(wordpairs) <= wordpairs is an iterable of 2-tuples containing strings as iterables
# The resulting Align.alignedpairs is a list of aligned 2-tuples

# Relies on C-code in libalign.so built from align.c through ctypes.
# Author: Mans Hulden
# MH20151102
# Last modified 20200422

import itertools
import os
from ctypes import *

_LIBALIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libalign.so')
libalign = cdll.LoadLibrary(_LIBALIGN_PATH)

libalign_add_int_pair = libalign.add_int_pair
libalign_add_int_pair.argtypes = [POINTER(c_int), POINTER(c_int)]
libalign_add_int_pair.restype = None
libalign_clear_counts = libalign.clear_counts
libalign_clear_counts.argtypes = []
libalign_clear_counts.restype = None
libalign_initial_align = libalign.initial_align
libalign_initial_align.argtypes = []
libalign_initial_align.restype = None
libalign_crp_train = libalign.crp_train
libalign_crp_train.argtypes = [c_int, c_int, c_int]
libalign_crp_train.restype = None
libalign_crp_align = libalign.crp_align
libalign_crp_align.argtypes = []
libalign_crp_align.restype = None
libalign_med_align = libalign.med_align
libalign_med_align.argtypes = []
libalign_med_align.restype = None

libalign_getpairs_init = libalign.getpairs_init
libalign_getpairs_init.argtypes = []
libalign_getpairs_init.restype = c_void_p
libalign_getpairs_in = libalign.getpairs_in
libalign_getpairs_in.argtypes = [c_void_p]
libalign_getpairs_in.restype = POINTER(c_int)
libalign_getpairs_out = libalign.getpairs_out
libalign_getpairs_out.argtypes = [c_void_p]
libalign_getpairs_out.restype = POINTER(c_int)
libalign_getpairs_advance = libalign.getpairs_advance
libalign_getpairs_advance.argtypes = [c_void_p]
libalign_getpairs_advance.restype = c_void_p
libalign_align_init = libalign.align_init
libalign_align_init.argtypes = []
libalign_align_init.restype = None

class Aligner:

    def __init__(self, wordpairs, align_symbol = ' ', iterations = 10, burnin = 1, lag = 1, mode = 'crp'):
        wordpairs = list(wordpairs)
        if mode not in ('crp', 'med'):
            raise ValueError("mode must be 'crp' or 'med'")

        s = set()
        for wl, wr in wordpairs:
            s |= {a for a in wl}
            s |= {a for a in wr}
        if len(s) > 255:
            raise ValueError("Too many symbols for C backend (max 255 non-epsilon symbols).")

        symbols = sorted(s)
        self.symboltoint = dict(zip(symbols, range(1, len(symbols) + 1)))
        self.inttosymbol = {v:k for k, v in self.symboltoint.items()}
        self.inttosymbol[0] = align_symbol
        ## Map stringpairs to -1 terminated integer sequences ##
        intpairs = []
        for i, o in wordpairs:
            if len(i) > 255 or len(o) > 255:
                raise ValueError("Input strings are too long for C backend (max 255 symbols per side).")
            intin = [self.symboltoint[x] for x in i] + [-1]
            intout = [self.symboltoint[x] for x in o] + [-1]
            #intin = map(lambda x: self.symboltoint[x], i) + [-1]
            #intout = map(lambda x: self.symboltoint[x], o) + [-1]
            intpairs.append((intin, intout))

        libalign_align_init()
        for i, o in intpairs:
            icint = (c_int * len(i))(*i)
            ocint = (c_int * len(o))(*o)
            libalign_add_int_pair(icint, ocint)
            
        # Run CRP align
        if mode == 'crp':
            libalign_clear_counts()
            libalign_initial_align()
            libalign_crp_train(c_int(iterations), c_int(burnin), c_int(lag))
            libalign_crp_align()
        else:
            libalign_clear_counts()
            libalign_initial_align()
            libalign_med_align()
        
        # Reconvert to output
        self.alignedpairs = []
        stringpairptr = libalign_getpairs_init()
        while stringpairptr is not None:
            sp = c_void_p(stringpairptr)
            inints = libalign_getpairs_in(sp)
            outints = libalign_getpairs_out(sp)
            instr = []
            outstr = []
            for j in itertools.count():
                if inints[j] == -1:
                    break
                instr.append(self.inttosymbol[inints[j]])
            for j in itertools.count():
                if outints[j] == -1:
                    break
                outstr.append(self.inttosymbol[outints[j]])
            self.alignedpairs.append((instr, outstr))
            stringpairptr = libalign_getpairs_advance(sp)
