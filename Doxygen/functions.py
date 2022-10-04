import numpy as np

"""
Contains all the different functions required for the application
to apply the functions on the data uploaded.
"""


def derivative(t, x):
    """
    Derivation of a signal.

    :param t: Time array
    :param x: Signal array
    :return: Array containing the derivative of the signal 'x'.
    """
    h = t[1] - t[0]
    y = np.zeros(len(t))
    i = 1
    while i <= len(x) - 1:
        y[i] = (x[i] - x[i - 1]) / h
        i = i + 1
    return y


def integration(t, x):
    """
    Integration of a signal.

    :param t: Time array
    :param x: Signal array
    :return: Array containing the integration of the signal 'x'.
    """
    h = t[1] - t[0]
    y = np.zeros(len(t))
    i = 1
    while i <= len(x) - 1:
        y[i] = y[i - 1] + h * (x[i - 1])
        i = i + 1
    return y


def myhighpass(t, u, tc):
    """
    First order high pass filter.

    :param t: Time array
    :param u: Signal array
    :param tc: Cut-off frequency for the filter
    :return: Array showing the high-pass output of signal 'x' based on cutoff frequency 'tc'.
    """
    x = [u[0]]
    y = np.zeros(len(t))
    h = t[1] - t[0]
    for i in range(len(t) - 1):
        y1 = (x[i] + (h / tc * u[i + 1])) / (1 + h / tc)
        x.append(y1)
    for k in range(len(t)):
        y[k] = -x[k] + u[k]
    return y


def mylowpass(t, u, tc):
    """
    First order low pass filter.

    :param t: Time array
    :param u: Signal array
    :param tc: Cut-off frequency for the filter
    :return: Array showing the low-pass output of signal 'x' based on cutoff frequency 'tc'.
    """
    x = [u[0]]
    y = np.zeros(len(t))
    h = t[1] - t[0]
    for i in range(len(t) - 1):
        y1 = (x[i] + (h / tc * u[i + 1])) / (1 + h / tc)
        x.append(y1)
    for k in range(len(t)):
        y[k] = x[k]
    return y


def mw_dft(data, t, omega):
    """
    Function required for calculating windowed phasor of a signal.

    :param data: Signal array
    :param t: Time array
    :param omega: Frequency
    :return: A complex number representing phasor quantity.
    """
    X_data = 0
    win_len = len(t)
    for k in range(len(t)):
        X_data = X_data + data[k] * np.exp((-omega * t[k] * 1j))
    X_data = np.sqrt(2) / win_len * X_data
    return X_data


# Take frequency as input as well
def window_phasor_mag(t, x, sr, cycles, dom_freq=50):
    """
    Magnitude of moving discrete fourier transform.

    :param t: Time array
    :param x: Signal array
    :param sr: Down-sampling factor
    :param cycles: Window size as per number of cycles
    :param dom_freq: Fundamental frequency
    :return: A complex number array of calculated fundamental phasor.
    """
    x = list(x)
    t = list(t)
    va = x[0::sr]
    t = t[0::sr]
    tnew = t
    h = tnew[1] - tnew[0]

    va_mw = np.zeros(len(va), dtype='complex_')
    period = round(cycles / (dom_freq * h))

    for i in range(period, len(t)):
        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
    return [abs(va_mw), tnew]


def window_phasor_angle(t, x, sr, cycles, dom_freq=50):
    """
    Angle of moving discrete fourier transform.

    :param t: Time array
    :param x: Signal array
    :param sr: Down-sampling factor
    :param cycles: Window size as per number of cycles
    :param dom_freq: Fundamental frequency
    :return: A complex number array of calculated fundamental phasor.
    """
    x = list(x)
    t = list(t)
    va = x[0::sr]
    t = t[0::sr]
    tnew = t
    h = tnew[1] - tnew[0]

    va_mw = np.zeros(len(va), dtype='complex_')
    period = round(cycles / (dom_freq * h))

    for i in range(period, len(t)):
        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
    return [np.angle(va_mw, deg=True), tnew]


def trendfilter(t, x, lamda1):
    """
    Returns a smoothened version of the input signal, depending on parameter 'lambda',
    considered as a median filter, also known as 'Hodrick Prescott' filter.

    :param t: Time array
    :param x: Signal array
    :param lamda1: Hyperparameter
    :return: An array of smooth version of signal 'x', depending on parameter 'lambda'
    """
    if len(t) > 10000:
        sr = int(np.ceil(len(t) / 10000))
        t = t[::sr]
        x = x[::sr]

    n = len(t)
    I = np.eye(n)
    D = np.zeros((n - 1, n))

    pp = 0
    for kk in range(n - 2):
        D[kk, 0 + pp:3 + pp] = [1, -2, 1]
        pp += 1
    y = np.linalg.inv(I + 2 * lamda1 * np.transpose(D) @ D) @ x[0:n]

    return y


def avgMovWin(t, v, t_win):
    """
    Moving window average of the signal.

    :param t: Time array
    :param v: Signal array
    :param t_win: Window length in seconds
    :return: Array representing the average of signal using a moving window.
    """
    N = len(t)
    h = t[1] - t[0]
    tw = t_win / h
    avg = np.zeros(N)

    for i in range(int(tw), N):
        sum1 = 0
        for j in range(int(i - tw), i):
            sum1 += v[j]
            avg[i] = sum1 / tw
    return avg


def rmsMovWin(t, v, t_win):
    """
    Moving window RMS of the signal.

    :param t: Time array
    :param v: Signal array
    :param t_win: Window length in seconds
    :return: Array representing the RMS of signal using a moving window.
    """
    N = len(t)
    h = t[1] - t[0]
    tw = t_win / h
    rms = np.zeros(N)

    for i in range(int(tw), N):
        sum1 = 0
        for j in range(int(i - tw), i):
            sum1 += v[j] ** 2
        rms[i] = np.sqrt(sum1 / tw)
    return rms


def clarkestranform(t, va, vb, vc):
    """
    Transforms 3 phase to alpha, beta, zero components.

    :param t: Time array
    :param va: a phase voltage array
    :param vb: b phase voltage array
    :param vc: c phase voltage array
    :return: 3 arrays corresponding to alpha, beta, zero component.
    """
    mat = [[1, 0, np.sqrt(0.5)],
           [-0.5, -np.sqrt(3) / 2, np.sqrt(0.5)],
           [-0.5, np.sqrt(3) / 2, np.sqrt(0.5)]]
    mat = np.array(mat)
    B = np.linalg.inv(np.sqrt(2 / 3) * mat)
    fabg = np.zeros([3, len(t)])
    for i in range(len(t)):
        for k in range(np.shape(fabg)[0]):
            fabg[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fabg[0], fabg[1], fabg[2]]


def inv_clarkestransform(t, va, vb, vc):
    """
    Inverse of Clarke's transform, converts alpha, beta, zero component to a,b,c component.

    :param t: Time array
    :param va: alpha component array
    :param vb: beta component array
    :param vc: zero component array
    :return: 3 arrays corresponding to a,b,c component.
    """
    mat = [[1, -0.5, -0.5],
           [0, -np.sqrt(3) / 2, np.sqrt(3) / 2],
           [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]
    mat = np.array(mat)
    B = np.linalg.inv(np.sqrt(2 / 3) * mat)
    fabg = np.zeros([3, len(t)])
    for i in range(len(t)):
        for k in range(np.shape(fabg)[0]):
            fabg[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fabg[0], fabg[1], fabg[2]]


def parkstransform(t, va, vb, vc, w, gamma):
    """
    Transforms 3 phase to D,Q,0 components.

    :param t: Time array
    :param va: a component array
    :param vb: b component array
    :param vc: c component array
    :param w: Frequency (in Hertz)
    :param gamma: Phase angle (in Radian)
    :return: 3 arrays corresponding to D,Q,0 component respectively.
    """
    fdqo = np.zeros([3, len(t)])
    for i in range(len(t)):
        a = (w * t[i]) + gamma
        mat = [[np.cos(a), np.cos(a - (2 * np.pi / 3)), np.cos(a - (4 * np.pi / 3))],
               [np.sin(a), np.sin(a - (2 * np.pi / 3)), np.sin(a - (4 * np.pi / 3))],
               [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]
        mat = np.array(mat)
        B = np.sqrt(2 / 3) * mat
        for k in range(np.shape(fdqo)[0]):
            fdqo[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fdqo[0], fdqo[1], fdqo[2]]


def inv_parkstransform(t, va, vb, vc, w, gamma):
    """
    Inverse of Park's transform, transforms D,Q,0 phase to a,b,c components.

    :param t: Time array
    :param va: D component array
    :param vb: Q component array
    :param vc: 0 component array
    :param w: Frequency (in Hertz)
    :param gamma: Phase angle (in Radian)
    :return: 3 arrays corresponding to a,b,c component respectively.
    """
    w1 = 2 * np.pi * w
    fdqo = np.zeros([3, len(t)])
    for i in range(len(t)):
        a = (w1 * t[i]) + gamma
        mat = [[np.cos(a), np.sin(a), np.sqrt(0.5)],
               [np.cos(a - (2 * np.pi / 3)), np.sin(a - (2 * np.pi / 3)), np.sqrt(0.5)],
               [np.cos(a - (4 * np.pi / 3)), np.sin(a - (4 * np.pi / 3)), np.sqrt(0.5)]]
        mat = np.array(mat)
        B = np.sqrt(2 / 3) * mat
        for k in range(np.shape(fdqo)[0]):
            fdqo[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fdqo[0], fdqo[1], fdqo[2]]


def sequencetransform(t, va, vb, vc):
    """

    :param t: Time array
    :param va: a component array
    :param vb: b component array
    :param vc: c component array
    :return:
    """
    a = np.exp(2 * np.pi * 1j / 3)
    mat = [[1, 1, 1],
           [a ** 2, a, 1],
           [a, a ** 2, 1]]
    mat = np.array(mat)
    B = np.linalg.inv(mat)
    fpno = np.zeros([3, len(t)], dtype='complex_')
    for i in range(len(t)):
        for k in range(np.shape(fpno)[0]):
            fpno[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [np.real(fpno[0]), np.real(fpno[1]), np.real(fpno[2])]


def instaLL_RMSVoltage(t, va, vb, vc):
    """
    Instantaneous line to line RMS voltage.

    :param t: Time array
    :param va: a phase voltage
    :param vb: b phase voltage
    :param vc: c phase voltage
    :return: Array corresponding to the instantaneous RMS voltage
    """
    v_rms = np.zeros(len(t))
    for i in range(len(t)):
        v_rms[i] = np.sqrt(va[i] ** 2 + vb[i] ** 2 + vc[i] ** 2)
    return v_rms


def insta_RMSCurrent(t, ia, ib, ic):
    """
    Instantaneous line current.

    :param t: Time array
    :param ia: a phase current
    :param ib: b phase current
    :param ic: c phase current
    :return: Array corresponding to the instantaneous line current.
    """
    i_rms = np.zeros(len(t))
    for i in range(len(t)):
        i_rms[i] = (1 / np.sqrt(3)) * np.sqrt(ia[i] ** 2 + ib[i] ** 2 + ic[i] ** 2)
    return i_rms
