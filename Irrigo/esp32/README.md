# Irrigo — ESP32 Hardware Integration Guide

This guide details the hardware layer connecting your physical sensor array to the **7-Feature Physics-Informed ET Model** backend.

---

## 1. System Architecture

```
[ Capacitive Soil Moisture (GPIO 34) ] ─┐
[ DS18B20 Soil Temp Probe  (GPIO 5)  ] ─┼─→ [ ESP32 Microcontroller ]
[ DHT22 / BME280 Sensor    (GPIO 4)  ] ─┘          │ (Local Wi-Fi HTTP POST)
                                                   ▼
                                         FastAPI Backend (:8000)
                                          ├─ SQLite DB (irrigo.db)
                                          ├─ 6h Rolling Solar Mean Calc
                                          ├─ 7-Feature Ridge ML Model (24h Delta)
                                          └─ FAO-56 Hybrid Decision
                                                   │
                                                   ▼
                                        HTTP 200 JSON Response
                                          ├─ recommend_irrigation (bool)
                                          ├─ predicted_moisture_24h (float)
                                          └─ confidence & reasons
                                                   │
                                                   ▼
                                      [ 5V Relay Pump Actuator (GPIO 26) ]
```

---

## 2. Hardware Components & Pinout

| Component | Model / Type | ESP32 GPIO | Notes |
|---|---|---|---|
| **Microcontroller** | ESP32-WROOM-32 (30/38 pin) | — | Powered via micro-USB / 5V |
| **Soil Moisture** | Capacitive Soil Moisture v1.2/v2.0 | **GPIO 34 (ADC1)** | Analog input; corrosion-resistant |
| **Soil Temperature** | DS18B20 Waterproof Probe | **GPIO 5** | OneWire bus (requires **4.7 kΩ pull-up** between VCC and DATA) |
| **Air Temp & Humidity**| DHT22 (AM2302) or BME280 | **GPIO 4** | Digital pin |
| **Relay Module** | 5V Single Channel Relay | **GPIO 26** | Drives 12V submersible DC pump |
| **Status Indicator** | Built-in Blue LED | **GPIO 2** | Lights up during pump actuation / status |

---

## 3. Sensor Calibration (Soil Moisture)

Different soil types and capacitive sensor clones have varying raw ADC output ranges (ESP32 ADC is 12-bit: `0 – 4095`).

1. Open `irrigo_esp32.ino` in Arduino IDE.
2. **Dry Air Calibration:** Hold sensor in dry air; observe raw ADC in Serial Monitor (typically `~3000 – 3500`). Set `AIR_VALUE = <observed_val>`.
3. **Wet Calibration:** Submerge the sensor blade in a cup of water up to the max line; observe raw ADC (typically `~1200 – 1600`). Set `WATER_VALUE = <observed_val>`.

---

## 4. Software Setup & Flashing

### A. Install Arduino IDE & Libraries
In the Arduino IDE Library Manager (`Ctrl + Shift + I`), install:
* **ArduinoJson** (by Benoit Blanchon, v6.x or v7.x)
* **DHT sensor library** (by Adafruit)
* **OneWire** (by Paul Stoffregen)
* **DallasTemperature** (by Miles Burton)

### B. Configure Wi-Fi & Backend Address
In [irrigo_esp32.ino](file:///d:/SEMESTER%203/EDI%20PROJECT/Irrigo/esp32/irrigo_esp32.ino):
```cpp
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://192.168.x.x:8000/sensor-push"; // Replace with your Laptop's IP
```

*To find your laptop's local IP on Windows:* open PowerShell and run `ipconfig`. Look for **IPv4 Address** under your Wi-Fi adapter (e.g., `192.168.1.100` or `192.168.43.x`).

### C. Flash to ESP32
1. Select Board: **ESP32 Dev Module**.
2. Select COM Port.
3. Click **Upload** (`Ctrl + U`).
4. Open **Serial Monitor** at baud rate **115200**.

---

## 5. End-to-End Operation

1. Every 5 minutes (`PUSH_INTERVAL_MS = 300000`), the ESP32 samples all sensors.
2. It constructs the JSON payload and sends an `HTTP POST /sensor-push` to the backend.
3. The backend calculates `weather_radiation_source_value_roll_mean_6h` from SQLite history, runs the **7-feature Ridge model** to predict 24h soil moisture, runs FAO-56 deficit analysis, and responds in $<50$ ms.
4. If `recommend_irrigation` is `true` and data confidence is safe, the ESP32 automatically activates `RELAY_PIN` for 15 seconds to water the crops.

---

## Safety Rules

> **NEVER** actuate the pump if `requires_manual_review` is `true` in the API response.

> **NEVER** leave the pump running unattended without a physical cutoff timer.

> **DO NOT** connect the relay to mains AC. Use 12V DC pump only.

> Test the full software stack with **simulated** readings before connecting any hardware.

---

## Sensor Calibration (Capacitive Soil Moisture)

1. Read ADC value with sensor in **dry air** → record as `DRY_VALUE`
2. Read ADC value with sensor in **water** → record as `WET_VALUE`
3. Use `map(rawADC, DRY_VALUE, WET_VALUE, 0, 100)` in firmware

Typical values:
- Dry: ~3300–4095 (12-bit ADC)
- Wet: ~1200–1800

---

## Network Requirements

- ESP32 and backend must be on the same LAN, or backend must be accessible via public IP/DDNS.
- Default backend port: `8000`
- Endpoint: `POST http://<backend-ip>:8000/sensor-push`
- See `sensor_push_example.json` for exact JSON schema.

---

## Future: Open-Meteo Forecast Integration

The ESP32 can optionally fetch a 24h weather forecast from Open-Meteo before posting:

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude=19.2059&longitude=73.8694
    &hourly=temperature_2m,precipitation,windspeed_10m,shortwave_radiation
    &forecast_days=1
```

Parse the first 24h aggregates and include them in the `/sensor-push` body.

---

## Deployment Checklist (Software First)

- [x] Stage 1 — ML model trained and validated
- [x] Stage 2 — FastAPI backend running
- [ ] Stage 3 — Dashboard built and tested
- [ ] Stage 4a — Simulate ESP32 POST with curl / Postman
- [ ] Stage 4b — Wire sensors, calibrate, flash firmware
- [ ] Stage 4c — Validate relay logic with manual override enabled
- [ ] Stage 4d — Enable auto-actuation only after ≥ 7 days of supervised operation
