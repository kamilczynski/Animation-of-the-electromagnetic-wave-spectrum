import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.family'] = 'Orbitron'
# ---------------------------------------------------------
# Definicje segmentów widma EM (x_start, x_end, etykieta, dł. fali):
# ---------------------------------------------------------
sections = [
    (0, 1,   'Radio waves',       1.0),
    (1, 2,   'Microwaves',   0.5),
    (2, 3,   'Infrared rays',    0.3),
    (3, 4,   'Visible light',     0.2),
    (4, 5,   'Ultraviolet light',          0.1),
    (5, 6,   'X-rays',       0.05),
    (6, 7,   'Gamma rays',       0.01),
]

# ---------------------------------------------------------
# (1) Obliczamy k_i = 2π / λ_i
# (2) Wyznaczamy fazę początkową każdego segmentu (ciągłość fali)
# ---------------------------------------------------------
k_vals = []
phase_offsets = [0.0]  # faza początkowa w segmencie 0

for i, (x_start, x_end, _, lam) in enumerate(sections):
    k_i = 2*np.pi / lam
    k_vals.append(k_i)

    if i < len(sections) - 1:
        theta_end = k_i * x_end + phase_offsets[i]
        k_next = 2*np.pi / sections[i+1][3]
        phi_next = theta_end - k_next * x_end
        phase_offsets.append(phi_next)

k_vals = np.array(k_vals)
phase_offsets = np.array(phase_offsets)

# ---------------------------------------------------------
# Funkcja: dla punktu x_val i numeru klatki frame
#          oblicza y = sin(k*x + faza - omega*t),
#          zależnie od segmentu (radio, micro, itp.).
# ---------------------------------------------------------
def em_wave(x_val, frame):
    shift_speed = 0.5
    time_phase = shift_speed * frame
    for i, (x_start, x_end, _, _) in enumerate(sections):
        if x_start <= x_val < x_end:
            return np.sin(k_vals[i]*x_val + phase_offsets[i] - time_phase)
    if x_val == 7:  # ostatni punkt
        i = len(sections) - 1
        return np.sin(k_vals[i]*x_val + phase_offsets[i] - time_phase)
    return 0.0

# ---------------------------------------------------------
# Funkcja do ciągłego mapowania długości fali (380..780 nm) na kolor RGB
# (przydatne do kolorowego scatter w obszarze widzialnym).
# ---------------------------------------------------------
def wavelength_to_rgb(wavelength_nm, gamma=0.8):
    w = float(wavelength_nm)
    if w < 380: w = 380
    if w > 780: w = 780

    # Odcinki: fiolet-niebieski, niebieski-zielony, zielony-żółty, itd.
    if 380 <= w < 440:
        R = -(w - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= w < 490:
        R = 0.0
        G = (w - 440) / (490 - 440)
        B = 1.0
    elif 490 <= w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        R = (w - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= w < 645:
        R = 1.0
        G = -(w - 645) / (645 - 580)
        B = 0.0
    else:
        # 645..780 => czerwony
        R = 1.0
        G = 0.0
        B = 0.0

    R = R**gamma
    G = G**gamma
    B = B**gamma
    return (R, G, B)

# ---------------------------------------------------------
# Tu tworzymy TŁO w obszarze [3..4] w 6 pasmach (od czerwieni przy x=3 do fioletu przy x=4).
# Zadane proporcje: czerwony 70, pomarańczowy 30, żółty 20, zielony 75, niebieski 45, fioletowy 70.
# Suma = 310.
# ---------------------------------------------------------
color_bands = [
    ("czerwony",       (1.0,  0.0,  0.0),   70),
    ("pomarańczowy",   (1.0,  0.65, 0.0),   30),
    ("żółty",          (1.0,  1.0,  0.0),   20),
    ("zielony",        (0.0,  1.0,  0.0),   75),
    ("niebieski",      (0.0,  0.0,  1.0),   45),
    ("fioletowy",      (0.54, 0.17, 0.89),  70),
]
total_weight = sum(b[2] for b in color_bands)  # 310

n_points = 400  # liczba próbek w poziomie
rainbow_array = np.zeros((1, n_points, 3), dtype=float)

idx_start = 0
for band_name, (r, g, b), wgt in color_bands:
    length = int(round(wgt / total_weight * n_points))
    idx_end = idx_start + length
    rainbow_array[0, idx_start:idx_end, 0] = r
    rainbow_array[0, idx_start:idx_end, 1] = g
    rainbow_array[0, idx_start:idx_end, 2] = b
    idx_start = idx_end

# Jeśli wskutek zaokrągleń mamy jeszcze niewypełnione piksele:
if idx_start < n_points:
    rainbow_array[0, idx_start:, :] = rainbow_array[0, idx_start-1, :]

# ---------------------------------------------------------
# Siatka X do rysowania fali
# ---------------------------------------------------------
x = np.linspace(0, 7, 2000)

# ---------------------------------------------------------
# Przygotowanie fig/ax
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3))
fig.set_facecolor('black')
ax.set_facecolor('black')

ax.set_xlim(0, 7)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# Podpisy każdego segmentu EM
for (x_start, x_end, label, _) in sections:
    mid = 0.5*(x_start + x_end)
    ax.text(mid, 1.1, label, color='white', ha='center', va='bottom',
            fontsize=8, alpha=0.9)

# Pasek w tle: [3..4] w 6 pasmach kolorów (czerwony->fiolet)
img_rainbow = ax.imshow(
    rainbow_array,
    extent=(3, 4, -1.2, 1.2),
    aspect='auto',
    alpha=0.3,
    zorder=0
)

# Linie: lewa, środek, prawa (biała/czysta + scatter w środku)
line_left,   = ax.plot([], [], color='white', lw=2, zorder=1)
line_middle, = ax.plot([], [], lw=2, zorder=2)
line_right,  = ax.plot([], [], color='white', lw=2, zorder=1)

scatter_mid = None

# ---------------------------------------------------------
# Funkcja update animacji
# ---------------------------------------------------------
def update(frame):
    global scatter_mid
    if scatter_mid is not None:
        scatter_mid.remove()

    # Obliczamy falę y w każdym x
    y_vals = np.array([em_wave(xx, frame) for xx in x])

    # Maski: lewa <3, środek 3..4, prawa >4
    mask_left   = (x < 3)
    mask_middle = (x >= 3) & (x <= 4)
    mask_right  = (x > 4)

    line_left.set_data(x[mask_left], y_vals[mask_left])
    line_right.set_data(x[mask_right], y_vals[mask_right])

    # Czyszczenie linii środkowej (i tak używamy scatter)
    line_middle.set_data([], [])

    # Rozkład barw na fali: x=3 => 780 nm (czerwony), x=4 => 380 nm (fiolet)
    x_mid = x[mask_middle]
    y_mid = y_vals[mask_middle]

    if len(x_mid) > 0:
        wls = 780 + (x_mid - 3)*(380 - 780)/(4 - 3)  # 780→380
        colors = [wavelength_to_rgb(wl) for wl in wls]
        scatter_mid = ax.scatter(
            x_mid, y_mid,
            c=colors,
            s=1,
            zorder=2
        )
    else:
        scatter_mid = ax.scatter([], [], s=0)

    return (line_left, line_middle, line_right, scatter_mid)

# ---------------------------------------------------------
# Uruchomienie animacji
# ---------------------------------------------------------
frames = 200
ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
ani.save(
    r'C:\Users\topgu\PycharmProjects\obrazowanie\media\videos\emwave.mp4',
    writer='ffmpeg',
    fps=20,
    dpi=150
)

plt.tight_layout()
plt.show()
