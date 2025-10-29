import matplotlib.pyplot as plt
import numpy as np

# ======== DENAH DARI GAMBAR (TAB DIGANTI SPASI) ========
asciimap = [
"        x          x          x       x                               ",
"   .................................................                  ",
"               .............D            ...........                  ",
"               ........                 X...........   x       x      ",
"               ........                  ...........X..............   ",
"                      .                  D             x       x      ",
"                      .                                               ",
"                      .                                               ",
"                 ......                                               ",
"         ..............x                                              ",
"                 ......                                               ",
"                 .                                                    ",
"       Z..........                                                    ",
"       Z..........                                                    ",
"       Z..........                                                    ",
"       Z..........                                                    ",
"                 .                                                    ",
"       RM        .L4                                                  ",
"       .         .                                                    ",
"       .    M    .                                                    ",
"       ...........                                                    ",
"       .         .                                                    ",
"       .         .                                                    ",
"       RL        .                                                    ",
"                 .                                                    ",
"       *............................                                  ",
"              L1     L2      L3    .                                  ",
"                                   .                                  ",
"                                   .                                  ",
"                                   .                                  ",
]

# Ubah jadi numpy grid
rows = len(asciimap)
cols = max(len(r) for r in asciimap)
grid = np.full((rows, cols), ' ', dtype='<U2')
for i, row in enumerate(asciimap):
    for j, ch in enumerate(row):
        grid[i, j] = ch

# Cari posisi start (*)
start_pos = np.argwhere(grid == '*')[0]
player_pos = [start_pos[0], start_pos[1]]
player_dir = '>'  # awal hadap kanan

# ======== FUNGSI UTIL ========
def get_orientation_label(dir_symbol):
    if dir_symbol == '>': return "front"
    if dir_symbol == '<': return "back"
    if dir_symbol == '^': return "left"
    if dir_symbol == 'v': return "right"

def get_xy_label():
    x = player_pos[1]
    y = rows - player_pos[0]
    return f"x{int(x):02d}_y{int(y):02d}_{get_orientation_label(player_dir)}"

# ======== RENDERING ========
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
plt.axis("off")

def render():
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')

    # Gambar peta
    for i in range(rows):
        for j in range(cols):
            ch = grid[i, j]
            if ch.strip():
                color = 'lightgray' if ch == '.' else 'yellow'
                ax.text(j, i, ch, ha='center', va='center', fontsize=8, color=color)

    # Player (ikon arah)
    ax.text(player_pos[1], player_pos[0], player_dir, color='lime',
            fontsize=14, ha='center', va='center', fontweight='bold')

    # Label koordinat
    ax.text(0, -1, f"Pos: {get_xy_label()}",
            fontsize=11, color='white', ha='left', va='top', fontweight='bold')

    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -2)
    fig.canvas.draw_idle()

# ======== KONTROL GERAK ========
def move(dx, dy, new_dir):
    global player_dir
    new_r = player_pos[0] + dy
    new_c = player_pos[1] + dx
    player_dir = new_dir

    if 0 <= new_r < rows and 0 <= new_c < cols:
        if grid[new_r, new_c] == '.' or grid[new_r, new_c] == '*':
            player_pos[0] = new_r
            player_pos[1] = new_c
    render()

def on_key(event):
    key = event.key.lower()
    if key in ['w', 'up']:
        move(0, -1, '^')
    elif key in ['s', 'down']:
        move(0, 1, 'v')
    elif key in ['a', 'left']:
        move(-1, 0, '<')
    elif key in ['d', 'right']:
        move(1, 0, '>')

fig.canvas.mpl_connect('key_press_event', on_key)
render()
plt.show()
