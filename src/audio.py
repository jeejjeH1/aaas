"""Synthesize the promo soundtrack (120 BPM: one beat = 0.5 s) + SFX.

Timeline must match promo.html (see CUES there):
  0-4   intro: dark pad, riser, hits on each slammed word
  4     impact (brand reveal), 4-6 build with kicks + snare roll
  6     DROP (cat lands)            6-15.5 full groove
  8-9   glitch + slot-machine ticks, 9 impact ($5)
  11.5, 13 card whooshes + hits
  15.5-17 point counter ticks, 17 impact (+7 days)
  19-21 build/snare roll, 21 final impact, outro fade to 25
"""
import numpy as np
import wave

SR = 44100
DUR = 25.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(3)
L = np.zeros(N)
R = np.zeros(N)


def add(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N:
        return
    sig = sig[: N - i]
    L[i:i + len(sig)] += sig * gain * np.sqrt(0.5 * (1 - pan))
    R[i:i + len(sig)] += sig * gain * np.sqrt(0.5 * (1 + pan))


def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)


def lp_fast(x, cut):
    """Vectorised-ish fixed-cut lowpass via FFT."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / cut) ** 4)
    return np.fft.irfft(X, len(x))


def hp_fast(x, cut):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (cut / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, len(x))


def bp_fast(x, lo, hi):
    return hp_fast(lp_fast(x, hi), lo)


def saw(freq, n, detune=(0,)):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in detune:
        f = freq * 2 ** (d / 1200)
        ph = rng.random()
        out += 2 * ((t * f + ph) % 1) - 1
    return out / len(detune)


# ---------------- drums ----------------
def kick(n=int(0.45 * SR), punch=1.0):
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.22)
    click = rng.standard_normal(n) * np.exp(-t / 0.004) * 0.4
    return np.tanh((body + click) * 2.2 * punch) * 0.9


def snare(n=int(0.3 * SR)):
    t = np.arange(n) / SR
    noise = bp_fast(rng.standard_normal(n), 1200, 9000) * np.exp(-t / 0.09)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return np.tanh((noise * 1.4 + tone * 0.8) * 1.3) * 0.6


def clap(n=int(0.35 * SR)):
    t = np.arange(n) / SR
    e = np.zeros(n)
    for k, off in enumerate([0, 0.011, 0.022, 0.034]):
        e += np.where(t >= off, np.exp(-(t - off) / (0.012 if k < 3 else 0.12)), 0)
    return bp_fast(rng.standard_normal(n), 900, 6000) * e * 0.55


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    return hp_fast(rng.standard_normal(n), 7000) * np.exp(-t / (0.09 if open_ else 0.018)) * 0.35


def impact(big=1.0):
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    f = 30 + 90 * np.exp(-t / 0.08)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
    crash = hp_fast(rng.standard_normal(n), 3000) * np.exp(-t / 0.9) * 0.35
    noise = lp_fast(rng.standard_normal(n), 900) * np.exp(-t / 0.25) * 0.8
    return np.tanh((boom * 1.6 + crash + noise) * big) * 0.95


def riser(dur, f0=200, f1=3000):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    noise = rng.standard_normal(n)
    # bandpass sweep approximated by crossfading filtered copies
    out = np.zeros(n)
    bands = np.geomspace(f0, f1, 8)
    for i, fc in enumerate(bands):
        w = np.clip(1 - abs(k * (len(bands) - 1) - i), 0, 1)
        out += bp_fast(noise, fc * 0.7, fc * 1.4) * w
    sweep = np.sin(2 * np.pi * np.cumsum(f0 + (f1 - f0) * k ** 2) / SR * 0.5) * 0.25
    return (out * 1.2 + sweep) * k ** 1.6 * 0.6


def whoosh(dur=0.6, rev=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = np.sin(np.pi * t / dur) ** 2
    if rev:
        e = (t / dur) ** 3
    sig = bp_fast(rng.standard_normal(n), 400, 5000) * e
    return sig * 0.5


def tick(freq):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2.01 * t)) * np.exp(-t / 0.03) * 0.25


def coin():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    f = np.where(t < 0.07, 988, 1319)
    return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.18) * 0.12


def glitch(dur=0.12):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = rng.choice([220, 330, 880, 1760, 2637])
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + rng.standard_normal(n) * 0.3
    s = np.round(s * 4) / 4  # bitcrush
    return s * 0.25 * np.exp(-t / dur)


def pluck(freq, dur=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = saw(freq, n, (-7, 7))
    return lp_fast(s * np.exp(-t / 0.08), 3500) * 0.22


# ---------------- harmony: Am - F - C - G (one chord per bar of 4 beats) ----------------
def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
roots = [45, 41, 48, 43]


def bar_of(t):
    return int(t / (4 * BEAT)) % 4


# sidechain envelope (pumps on every kick)
sc = np.ones(N)
kick_times = []

# ---------------- arrangement ----------------
# intro (0-4): slams on words
for t in [0.5, 1.0, 2.0]:
    add(kick(punch=1.3), t, 0.9)
    add(impact(0.6), t, 0.35)
add(riser(3.5, 150, 2500), 0.5, 0.45)
add(whoosh(0.5, rev=True), 3.5, 0.9)
add(impact(1.2), 4.0, 0.9)

# 4-6 build: kicks each beat, snare roll accelerating in 5-6
for b in range(4, 11):
    t = b * BEAT + 2.0  # 4.0 .. 5.0
    if t < 5.5:
        kick_times.append(t)
roll = []
t = 5.0
step = 0.125
while t < 5.95:
    roll.append(t)
    t += step
    step = max(0.04, step * 0.82)
for i, t in enumerate(roll):
    add(snare(), t, 0.25 + 0.5 * i / len(roll))
add(riser(1.5, 300, 6000), 4.5, 0.5)

# 6-15.5 drop: four on the floor + claps + hats
for b in range(int(6 / BEAT), int(15.5 / BEAT)):
    t = b * BEAT
    kick_times.append(t)
    if b % 2 == 1:
        add(clap(), t, 0.8)
    add(hat(), t + 0.25, 0.8, pan=0.2)
    add(hat(), t + 0.125, 0.35, pan=-0.2)
    add(hat(), t + 0.375, 0.35, pan=-0.2)
add(impact(1.4), 6.0, 1.0)
# glitch section + slot ticks + $5 hit
for i in range(14):
    add(glitch(), 8.0 + i * 0.07 + rng.random() * 0.02, 0.9, pan=rng.uniform(-.7, .7))
for i in range(16):
    add(tick(900 + i * 60), 8.2 + i * 0.05, 0.9)
add(impact(1.1), 9.0, 0.85)
add(coin(), 9.05, 1.0)
# card whooshes
for t in [11.0, 12.85, 11.4, 13.0]:
    add(whoosh(0.5), t - 0.35, 0.9)
for t in [11.5, 13.0]:
    add(impact(0.8), t, 0.5)

# 15.5-19 reward: half-time kicks, counter ticks, impact at 17
add(whoosh(0.6, rev=True), 14.9, 0.9)
add(impact(1.2), 15.5, 0.8)
for b in range(int(15.5 / BEAT), int(19 / BEAT)):
    t = b * BEAT
    if b % 2 == 1 or t >= 17:
        kick_times.append(t)
    add(hat(open_=(b % 2 == 1)), t + 0.25, 0.6)
for i in range(25):
    add(tick(700 + i * 40), 15.6 + i * 0.04, 0.8)
add(coin(), 16.6, 1.0)
add(impact(1.2), 17.0, 0.8)
add(coin(), 17.05, 0.8)

# 19-21 build with snare roll
for b in range(int(19 / BEAT), int(21 / BEAT)):
    kick_times.append(b * BEAT)
t, step, roll = 20.0, 0.125, []
while t < 20.95:
    roll.append(t)
    t += step
    step = max(0.035, step * 0.84)
for i, t in enumerate(roll):
    add(snare(), t, 0.25 + 0.55 * i / len(roll))
add(riser(2.0, 300, 7000), 19.0, 0.6)

# 21 final impact + outro groove
add(impact(1.5), 21.0, 1.0)
for b in range(int(21 / BEAT), int(24 / BEAT)):
    t = b * BEAT
    kick_times.append(t)
    if b % 2 == 1:
        add(clap(), t, 0.7)
    add(hat(), t + 0.25, 0.7)

for t in kick_times:
    add(kick(), t, 0.95)
    i = int(t * SR)
    n = int(0.3 * SR)
    seg = 1 - 0.75 * np.exp(-np.arange(n) / SR / 0.09)
    j = min(N, i + n)
    sc[i:j] = np.minimum(sc[i:j], seg[: j - i])

# ---------------- music bed: bass + pad + arp ----------------
music_L = np.zeros(N)
music_R = np.zeros(N)
bar = 4 * BEAT
for k in range(int(DUR / bar) + 1):
    t0 = k * bar
    c = chords[k % 4]
    n = int(bar * SR)
    if t0 >= DUR:
        break
    # pad: detuned saw chords
    pad = sum(saw(midi(m + 12), n, (-12, -4, 5, 13)) for m in c) / 3
    pad = lp_fast(pad, 1800) * 0.16
    a = np.minimum(1, np.arange(n) / (0.05 * SR))
    i0 = int(t0 * SR)
    j0 = min(N, i0 + n)
    music_L[i0:j0] += (pad * a)[: j0 - i0]
    music_R[i0:j0] += (lp_fast(sum(saw(midi(m + 12), n, (-10, 3, 11)) for m in c) / 3, 1800) * 0.16 * a)[: j0 - i0]
    # bass: offbeat 8ths, active from 4s
    if t0 >= 4.0:
        for e in range(8):
            tt = t0 + e * BEAT / 2
            if 5.5 <= tt < 6.0:
                continue
            nb = int(0.24 * SR)
            tb = np.arange(nb) / SR
            f = midi(roots[k % 4] - 12 + (12 if e % 2 else 0))
            s = (saw(f, nb, (-5, 5)) * 0.6 + np.sin(2 * np.pi * f * tb) * 0.8) * np.minimum(1, tb / 0.005) * np.exp(-tb / 0.18)
            s = lp_fast(s, 700) * 0.42
            ii = int(tt * SR)
            if ii >= N:
                continue
            jj = min(N, ii + nb)
            music_L[ii:jj] += s[: jj - ii]
            music_R[ii:jj] += s[: jj - ii]
    # arp plucks in the drop / outro
    if 6.0 <= t0 < 15.5 or t0 >= 21.0:
        notes = [c[0] + 24, c[1] + 24, c[2] + 24, c[1] + 36]
        for s16 in range(16):
            tt = t0 + s16 * BEAT / 4
            p = pluck(midi(notes[s16 % 4]))
            ii = int(tt * SR)
            if ii >= N:
                continue
            jj = min(N, ii + len(p))
            pan = 0.35 if s16 % 2 else -0.35
            music_L[ii:jj] += p[: jj - ii] * (1 - pan) * 0.7
            music_R[ii:jj] += p[: jj - ii] * (1 + pan) * 0.7

# intro: filter the music bed (dark, muffled) then open up at 4s
t_axis = np.arange(N) / SR
open_ = np.clip((t_axis - 3.0) / 1.0, 0, 1)
muff_L, muff_R = lp_fast(music_L, 450), lp_fast(music_R, 450)
music_L = muff_L * (1 - open_) + music_L * open_
music_R = muff_R * (1 - open_) + music_R * open_
# drop gap: silence the bed just before the drop
gap = 1 - 0.85 * ((t_axis > 5.6) & (t_axis < 6.0))
gap2 = 1 - 0.7 * ((t_axis > 20.6) & (t_axis < 21.0))
music_L *= sc * gap * gap2
music_R *= sc * gap * gap2

L += music_L
R += music_R

# master: fade-in click guard, outro fade, soft clip, normalise
fade = np.clip((DUR - t_axis) / 2.5, 0, 1) * np.clip(t_axis / 0.02, 0, 1)
L *= fade
R *= fade
mix = np.stack([L, R], 1)
mix = np.tanh(mix * 1.1)
mix /= np.abs(mix).max() / 0.95
pcm = (mix * 32767).astype(np.int16)
with wave.open('assets/soundtrack.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote assets/soundtrack.wav', DUR, 's')
