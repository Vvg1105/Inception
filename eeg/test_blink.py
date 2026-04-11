"""
test_blink.py — Manual sanity-check for blink detection.

Opens the g.tec TimeSeriesScope visualizer showing all 8 EEG channels,
and prints a timestamped line to the terminal every time a blink is detected.

Run:
  conda run -n base python eeg/test_blink.py

Blink deliberately and confirm the terminal output and on-screen spikes line up.
The detection threshold is fixed inside eeg_stream.py (DEFAULT_BLINK_THRESHOLD).
"""

import os
import sys
import time
import threading

import gpype as gp

sys.path.insert(0, os.path.dirname(__file__))
from eeg_stream import EEGBuffer, check_blink_state

POLL_EVERY = 0.1     # seconds between blink checks in the background thread

def _blink_monitor():
    """Background thread: prints to terminal whenever a blink fires."""
    blink_was_active = False
    while True:
        active = check_blink_state()
        if active and not blink_was_active:
            print(f"[{time.strftime('%H:%M:%S')}]  BLINK detected", flush=True)
        blink_was_active = active
        time.sleep(POLL_EVERY)

def main():
    # ── Pipeline ──────────────────────────────────────────────────────────────
    p        = gp.Pipeline()
    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
    buf = EEGBuffer()

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  buf)    # shared buffer — feeds check_blink_state()

    # ── App — must exist before any QWidget is constructed ───────────────────
    app   = gp.MainApp(caption="Blink Detection Test")
    scope = gp.TimeSeriesScope(time_window=5, amplitude_limit=200)
    app.add_widget(scope)
    p.connect(notch60, scope)   # visualizer

    p.start()

    # Poll for blinks in the background while the Qt window is open
    t = threading.Thread(target=_blink_monitor, daemon=True)
    t.start()

    print("close the window to stop\n")
    app.run()   # blocks until window is closed

    p.stop()
    print("Done.")

if __name__ == "__main__":
    main()
