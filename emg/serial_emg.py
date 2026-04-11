"""Shared serial helpers for EMG scripts."""

from __future__ import annotations

import serial
import serial.tools.list_ports


def find_serial_port(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    ports = serial.tools.list_ports.comports()
    for p in ports:
        dev = (p.device or "").lower()
        if "usbmodem" in dev or "usbserial" in dev:
            return p.device
        desc = (p.description or "").lower()
        if "arduino" in desc:
            return p.device
    return None


def open_emg_serial(port: str | None = None, baud: int = 9600, timeout: float = 1.0):
    p = find_serial_port(port)
    if not p:
        return None, None
    ser = serial.Serial(p, baud, timeout=timeout)
    return ser, p
