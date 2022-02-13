import numpy as np
from scipy import stats

bm25 = np.loadtxt('bm25.avg_p.txt', dtype=np.float)
print(len(bm25))
inl2 = np.loadtxt('inl2.avg_p.txt', dtype=np.float)
print(len(inl2))

rel = stats.ttest_rel(bm25, inl2)
print(rel)
