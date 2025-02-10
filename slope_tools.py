import numpy as np

def minmod(a,b):
    mm               = np.zeros_like(a)
    absa             = np.abs(a)
    absb             = np.abs(b)
    ab               = a*b
    mask             = np.logical_and(absa<absb,ab>0)
    mm[mask]         = a[mask]
    mask             = np.logical_and(absa>absb,ab>0)
    mm[mask]         = b[mask]
    return mm

def maxmod(a,b):
    mm               = np.zeros_like(a)
    absa             = np.abs(a)
    absb             = np.abs(b)
    ab               = a*b
    mask             = np.logical_and(absa<absb,ab>0)
    mm[mask]         = b[mask]
    mask             = np.logical_and(absa>absb,ab>0)
    mm[mask]         = a[mask]
    return mm

def minmodslope(q,dx):
    sl = np.zeros_like(q)[1:-1]
    qc               = q[1:-1]
    ql               = q[:-2]
    qr               = q[2:]
    dql              = (qc-ql)/dx
    dqr              = (qr-qc)/dx
    sl[:]            = minmod(dql,dqr)
    return sl
    
def superbeeslope(q,dx):
    sl = np.zeros_like(q)[1:-1]
    qc               = q[1:-1]
    ql               = q[:-2]
    qr               = q[2:]
    dql              = (qc-ql)/dx
    dqr              = (qr-qc)/dx
    sl1              = minmod(2*dql,dqr)
    sl2              = minmod(dql,2*dqr)
    sl[:]            = maxmod(sl1,sl2)
    return sl
