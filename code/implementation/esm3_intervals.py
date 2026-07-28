import numpy as np

def chop(L, min_overlap=511, max_len=1022):
    return L[max_len-min_overlap:-max_len+min_overlap]

def intervals(L, min_overlap=511, max_len=1022, parts=None):
    if parts is None: parts = []
    if len(L) <= max_len:
        if parts[-2][-1] - parts[-1][0] < min_overlap:
            return parts + [np.arange(L[int(len(L)/2)] - int(max_len/2), L[int(len(L)/2)] + int(max_len/2))]
        else:
            return parts
    else:
        parts += [L[:max_len], L[-max_len:]]
        L = chop(L, min_overlap, max_len)
        return intervals(L, min_overlap, max_len, parts=parts)

def get_intervals_and_weights(seq_len, min_overlap=511, max_len=1022, s=16):
    ints = intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]

    a = int(np.round(min_overlap/2))
    t = np.arange(max_len)

    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a/2) / s))
    f[max_len-a:] = 1 / (1 + np.exp((t[:a] - a/2) / s))

    f0 = np.ones(max_len)
    f0[max_len-a:] = 1 / (1 + np.exp((t[:a] - a/2) / s))

    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a/2) / s))

    filt = [f0] + [f for i in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k]
        M_sum = M.sum(0)
        M_norm = np.divide(M, M_sum, out=np.zeros_like(M), where=M_sum != 0)
    return (ints, M, M_norm) 