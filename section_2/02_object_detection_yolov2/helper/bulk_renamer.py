import os
import glob
import random
import string

"""
    All files under `path` are renamed with `N` random  characters
"""
path = "../data/images"
N = 8

files = glob.glob(os.path.join(path, '*.*'))

for fn in files:
    rand_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
    _, ext = os.path.splitext(os.path.basename(fn))
    os.rename(fn, os.path.join(path, rand_string+ext))
