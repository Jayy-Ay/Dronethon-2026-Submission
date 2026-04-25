# Setup Guide

## Prerequisites

- Python 3.10+
- Pi and PC connected to the same network (for example the drone hotspot)
- Open UDP port `9000` on the PC firewall (or choose another port)

## Installation Steps

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pi to PC Telemetry (UDP)

1. Start the receiver on your PC from the project root:

```bash
python scripts/pc_receive_telemetry.py --port 9000
```

2. On the Raspberry Pi, run the sender and point it at your PC IP:

```bash
python scripts/pi_send_telemetry.py --pc-ip <PC_IP> --port 9000 --rate-hz 5
```

3. You should see packet sequence numbers and JSON payloads printed on the PC.

If you do not receive data:
- Confirm both devices are on the same network.
- Confirm the PC firewall allows inbound UDP on your chosen port.
- Confirm the sender uses the correct PC IPv4 address.

## Configuration

- Change the receive port with `--port` on both scripts.
- Change send frequency with `--rate-hz` on the Pi sender.
- Replace demo payload values in `scripts/pi_send_telemetry.py` with real sensor values.
