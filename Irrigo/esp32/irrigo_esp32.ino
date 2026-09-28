/*
  Irrigo — ESP32 Smart Irrigation Client
  =======================================
  Hardware:
    - ESP32 Development Board (ESP32-WROOM-32)
    - Capacitive Soil Moisture Sensor v1.2 / v2.0 (Analog ADC on GPIO 34)
    - DS18B20 Waterproof Temperature Probe (OneWire on GPIO 5)
    - DHT22 (or DHT11 / BME280) Temperature & Humidity Sensor (GPIO 4)
    - 5V Relay Module (GPIO 26, active LOW or HIGH configurable)

  Required Arduino Libraries:
    - ArduinoJson (by Benoit Blanchon)
    - DHT sensor library (by Adafruit)
    - OneWire (by Paul Stoffregen)
    - DallasTemperature (by Miles Burton)
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ─── Network Configuration ───────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";         // Wi-Fi or Mobile Hotspot SSID
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";     // Wi-Fi Password
const char* SERVER_URL    = "http://192.168.1.100:8000/sensor-push"; // Backend IP & Port

// ─── Device & Zone Identifiers ───────────────────────────────────────────────
const char* DEVICE_ID     = "esp32-zone1-001";
const int   ZONE_ID       = 1;
const char* CROP_TYPE     = "tomato";
const char* GROWTH_STAGE  = "mid_season";             // "initial", "development", "mid_season", "late_season"

// ─── Pin Definitions ────────────────────────────────────────────────────────
const int SOIL_PIN        = 34;   // Analog ADC1 pin for capacitive moisture sensor
const int DS18B20_PIN     = 5;    // Digital pin with 4.7k pullup for DS18B20 OneWire
const int DHT_PIN         = 4;    // Digital pin for DHT22
const int RELAY_PIN       = 26;   // Digital pin for pump relay
const int LED_STATUS_PIN  = 2;    // Built-in LED for status indication

#define DHTTYPE           DHT22   // Change to DHT11 if using DHT11
#define RELAY_ACTIVE_LOW  true    // Set true if relay turns ON with LOW signal

// ─── Calibration Constants (Soil Moisture) ──────────────────────────────────
// Calibration procedure:
// 1. Read sensor in dry air -> set AIR_VALUE
// 2. Submerge in cup of water (up to line) -> set WATER_VALUE
const int AIR_VALUE       = 3200; // Sensor in dry air (raw 12-bit ADC value ~3000-3500)
const int WATER_VALUE     = 1400; // Sensor fully submerged in water (~1200-1600)

// ─── Timing Parameters ──────────────────────────────────────────────────────
const unsigned long PUSH_INTERVAL_MS = 300000; // 5 minutes between pushes (300,000 ms)
const unsigned long PUMP_RUN_TIME_MS = 15000;  // Run pump for 15 seconds if triggered

// ─── Sensor Objects ─────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHTTYPE);
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

// ─── Helper: Relay Control ──────────────────────────────────────────────────
void setRelay(bool state) {
  if (RELAY_ACTIVE_LOW) {
    digitalWrite(RELAY_PIN, state ? LOW : HIGH);
  } else {
    digitalWrite(RELAY_PIN, state ? HIGH : LOW);
  }
}

// ─── Setup ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=================================");
  Serial.println("  Irrigo ESP32 Client Starting  ");
  Serial.println("=================================");

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);
  setRelay(false); // Ensure pump is OFF initially

  // Initialize sensors
  dht.begin();
  ds18b20.begin();
  analogReadResolution(12); // 12-bit ADC (0 - 4095)

  // Connect to Wi-Fi
  connectWiFi();
}

// ─── Main Loop ──────────────────────────────────────────────────────────────
void loop() {
  // Ensure Wi-Fi connection is healthy
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[Wi-Fi] Disconnected. Attempting reconnection ...");
    connectWiFi();
  }

  // Read sensors
  float soilMoisturePct = readSoilMoisture();
  float soilTempC       = readSoilTemperature();
  float airTempC        = dht.readTemperature();
  float humidityPct     = dht.readHumidity();

  // Validate sensor readings
  bool airValid  = !isnan(airTempC) && !isnan(humidityPct) && airTempC >= -20.0 && airTempC <= 60.0;
  bool soilValid = soilMoisturePct >= 0.0 && soilMoisturePct <= 100.0;
  bool tempValid = soilTempC > -50.0 && soilTempC < 85.0; // DS18B20 error returns -127

  Serial.println("\n--- Live Sensor Readings ---");
  Serial.printf("Soil Moisture : %.1f %%\n", soilMoisturePct);
  if (tempValid) Serial.printf("Soil Temp     : %.1f °C\n", soilTempC);
  else Serial.println("Soil Temp     : [Sensor Error / Disconnected]");
  if (airValid) {
    Serial.printf("Air Temp      : %.1f °C\n", airTempC);
    Serial.printf("Humidity      : %.1f %%\n", humidityPct);
  } else {
    Serial.println("Air Sensor    : [DHT Read Error / Disconnected]");
  }

  // Build JSON Payload
  StaticJsonDocument<512> doc;
  doc["soil_moisture_pct"] = soilMoisturePct;
  if (tempValid) doc["soil_temperature_c"] = soilTempC;
  if (airValid) {
    doc["air_temperature_c"] = airTempC;
    doc["humidity_pct"]      = humidityPct;
  }
  // If optional ambient sensors or weather forecast are added:
  doc["forecast_rainfall_mm"]   = 0.0;
  doc["wind_speed_mps"]         = 1.5;
  doc["solar_radiation"]        = 450.0; // Default estimate or from local pyranometer
  doc["water_volume_past_4h_l"] = 0.0;
  doc["crop_type"]              = CROP_TYPE;
  doc["growth_stage"]           = GROWTH_STAGE;
  doc["device_id"]              = DEVICE_ID;
  doc["zone_id"]                = ZONE_ID;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Send HTTP POST to FastAPI Backend
  sendSensorPush(jsonPayload);

  // Wait for next reading cycle
  Serial.printf("\nWaiting %lu seconds until next push ...\n", PUSH_INTERVAL_MS / 1000);
  delay(PUSH_INTERVAL_MS);
}

// ─── Wi-Fi Connection Routine ───────────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[Wi-Fi] Connecting to '%s' ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[Wi-Fi] Connected successfully!");
    Serial.printf("[Wi-Fi] ESP32 Local IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[Wi-Fi] Connection failed. Will retry next cycle.");
  }
}

// ─── Read Soil Moisture (%) ─────────────────────────────────────────────────
float readSoilMoisture() {
  // Multi-sample averaging for noise reduction
  long total = 0;
  const int samples = 10;
  for (int i = 0; i < samples; i++) {
    total += analogRead(SOIL_PIN);
    delay(10);
  }
  int rawADC = total / samples;

  // Constrain within calibration bounds
  rawADC = constrain(rawADC, min(AIR_VALUE, WATER_VALUE), max(AIR_VALUE, WATER_VALUE));

  // Map raw ADC to percentage (higher ADC = drier soil, lower ADC = wetter soil)
  float moisture = map(rawADC, AIR_VALUE, WATER_VALUE, 0, 100);
  return constrain(moisture, 0.0, 100.0);
}

// ─── Read Soil Temperature (°C) ─────────────────────────────────────────────
float readSoilTemperature() {
  ds18b20.requestTemperatures();
  float tempC = ds18b20.getTempCByIndex(0);
  return tempC;
}

// ─── Send HTTP POST /sensor-push ────────────────────────────────────────────
void sendSensorPush(const String& payload) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] Cannot send: Wi-Fi disconnected.");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000); // 10s timeout

  Serial.println("[HTTP] Sending POST to: " + String(SERVER_URL));
  int httpResponseCode = http.POST(payload);

  if (httpResponseCode == 200) {
    String responseBody = http.getString();
    Serial.println("[HTTP] Success (200 OK)");

    // Parse Decision
    StaticJsonDocument<1024> responseDoc;
    DeserializationError err = deserializeJson(responseDoc, responseBody);
    if (!err) {
      bool recommendIrrigation = responseDoc["recommend_irrigation"] | false;
      float predicted24h       = responseDoc["predicted_moisture_24h"] | -1.0;
      float confidence         = responseDoc["confidence"] | 0.0;
      bool manualReview        = responseDoc["requires_manual_review"] | false;

      Serial.println("=================================");
      Serial.printf("Predicted Moisture 24h: %.1f %%\n", predicted24h);
      Serial.printf("Decision Confidence   : %.0f %%\n", confidence * 100);
      Serial.printf("Irrigation Recommended: %s\n", recommendIrrigation ? "YES (IRRIGATE)" : "NO");
      if (manualReview) {
        Serial.println("WARNING: Manual Review Flagged by Backend!");
      }
      Serial.println("=================================");

      // Actuate Relay if irrigation is recommended and safe
      if (recommendIrrigation && !manualReview) {
        Serial.printf("[Actuator] Starting pump for %lu seconds ...\n", PUMP_RUN_TIME_MS / 1000);
        digitalWrite(LED_STATUS_PIN, HIGH);
        setRelay(true);
        delay(PUMP_RUN_TIME_MS);
        setRelay(false);
        digitalWrite(LED_STATUS_PIN, LOW);
        Serial.println("[Actuator] Pump stopped.");
      }
    } else {
      Serial.println("[HTTP] Error parsing backend JSON response.");
    }
  } else if (httpResponseCode > 0) {
    Serial.printf("[HTTP] Failed with HTTP Error Code: %d\n", httpResponseCode);
    Serial.println(http.getString());
  } else {
    Serial.printf("[HTTP] Connection failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end();
}
