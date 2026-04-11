import os
import sys

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

from serial_emg import open_emg_serial

ser, port = open_emg_serial()
if not ser:
    print("Arduino not found. Is it plugged in?")
    exit()

print(f"Found Arduino on {port}")

# --- Rolling buffer: last 200 samples (~2 seconds) ---
data = deque([0] * 200, maxlen=200)

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(list(data), color='cyan', linewidth=1.5)
ax.set_ylim(0, 1023)
ax.set_xlim(0, 200)
ax.set_title('Live EMG Signal — Flex your forearm!')
ax.set_ylabel('Raw ADC value (0–1023)')
ax.set_xlabel('Time →')
ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='threshold (adjust later)')
ax.legend()
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.tick_params(colors='white')
ax.title.set_color('white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')

def update(frame):
    try:
        raw = ser.readline().decode('utf-8').strip()
        if raw.isdigit():
            data.append(int(raw))
            line.set_ydata(list(data))
    except:
        pass
    return line,

ani = animation.FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()

ser.close()