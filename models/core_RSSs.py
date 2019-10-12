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
