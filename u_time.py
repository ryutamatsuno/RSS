

import time

from typing import Callable


__timekeys = {}

def start(key = ''):

    __timekeys[key] = time.time()


def pause(key = ''):
    start('__pause::' + key)

def resume(key = ''):
    x = stop('__pause::' + key)
    __timekeys[key] += x


def stop(key=''):
    t = time.time() - __timekeys[key]
    del __timekeys[key]
    return t

def time2str(sec:int):
    #x = '%.4f[s]'%sec
    txt = None
    if sec > 0:
        s = sec % 60
        sec = sec // 60
        txt = '%5.2f[s]'%(s)
    if sec > 0:
        m = sec % 60
        sec = sec // 60
        txt = '%2d[m]'%(m) + txt
    if sec > 0:
        h = sec % 24
        sec = sec // 24
        txt = '%2d[h]'%(h) + txt
    if sec > 0:
        d = sec
        txt = '%2d[d]'%(d) + txt
    #txt += '  = ' + x
    return txt