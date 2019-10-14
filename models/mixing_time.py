import math
from sampling_util import ln, binom


def t_k(n, k, e, delta, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 2 * k * delta
    tau = rho * (ln(binom(n, k)) + ln(k) + ln(delta) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t


def t_k2(n, k, e, delta, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 2 * k * delta
    tau = rho * (ln(binom(n, k)) + 3 * ln(k) + ln(delta) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t


def tMCMC_k(n, k, e, delta, dia, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 1/2 * math.factorial(k) * delta ** k * (dia + k - 1) * n
    tau = rho * (ln(binom(n, k)) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t

def tPSRW_k(n, k, e, delta, dia, mixing_time_ratio=1.0):
    if mixing_time_ratio == 0:
        return 0
    rho = 1 / 2 * math.factorial(k - 1) * (k - 1) * delta ** k * (dia + k - 2) * n
    tau = rho * (ln(k - 1) + ln(delta) + ln(binom(n, k - 1)) + ln(1 / e))
    t = int(math.ceil(tau * mixing_time_ratio))
    if t == 0:
        t = 1
    return t