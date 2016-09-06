# coding: utf-8
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
y = np.sin(2*np.pi*f*t)+0.5*np.random.randn(t.size)

# パラメータ設定
fp = 2                          # 通過域端周波数[Hz]
fs = 3                          # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "low")
y1 = signal.filtfilt(b1, a1, y)

# 第一種チェビシェフフィルタ
N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
b2, a2 = signal.cheby1(N, gpass, Wn, "low")
y2 = signal.filtfilt(b2, a2, y)

# 第二種チェビシェフフィルタ
N, Wn = signal.cheb2ord(Wp, Ws, gpass, gstop)
b3, a3 = signal.cheby2(N, gstop, Wn, "low")
y3 = signal.filtfilt(b3, a3, y)

# 楕円フィルタ
N, Wn = signal.ellipord(Wp, Ws, gpass, gstop)
b4, a4 = signal.ellip(N, gpass, gstop, Wn, "low")
y4 = signal.filtfilt(b4, a4, y)

# ベッセルフィルタ
N = 4
b5, a5 = signal.bessel(N, Ws, "low")
y5 = signal.filtfilt(b5, a5, y)

# FIR フィルタ
a6 = 1
numtaps = n
b6 = signal.firwin(numtaps, Wp, window="hann")
y6 = signal.lfilter(b6, a6, y)
delay = (numtaps-1)/2*dt

# プロット
plt.figure()
plt.plot(t, y, "b")
plt.plot(t, y1, "r", linewidth=2, label="butter")
plt.plot(t, y2, "g", linewidth=2, label="cheby1")
plt.plot(t, y3, "c", linewidth=2, label="cheby2")
plt.plot(t, y4, "m", linewidth=2, label="ellip")
plt.plot(t, y5, "k", linewidth=2, label="bessel")
plt.plot(t-delay, y6, "y", linewidth=2, label="fir")
plt.xlim(0, 4)
plt.legend(loc="upper right")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
#plt.savefig('../files/150610Filter.svg')
plt.show()
