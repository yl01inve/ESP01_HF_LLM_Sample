#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>

// ================= Configuration Area =================
const char* ssid = "sun";      // Please modify here
const char* password = "20001010";  // Please modify here
//hsuwill000/ESP01LLMSample
const char* apiUrl = "https://hsuwill000-ESP01LLMSample.hf.space/infer4";

// Create a String buffer to store user input
String inputBuffer = "";
// ===========================================

void setup() {
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi Connected");
  Serial.println("IP Address: " + WiFi.localIP().toString());
  Serial.println("---");
  // Prompt cursor
  Serial.print("Q: "); 
}

void loop() {
  // Read single character (suitable for terminals like TeraTerm)
  while (Serial.available()) {
    char inChar = (char)Serial.read();

    // Handle Backspace key, code is typically 8 or 127
    if (inChar == 8 || inChar == 127) {
      if (inputBuffer.length() > 0) {
        // Delete the last character in the buffer
        inputBuffer.remove(inputBuffer.length() - 1);
        // Terminal visual deletion: Backspace -> Space -> Backspace
        Serial.print("\b \b");
      }
    }
    // Handle newline (Enter key), typically \r or \n
    else if (inChar == '\n' || inChar == '\r') {
      // Avoid double triggering from \r\n, only send if the buffer has content
      if (inputBuffer.length() > 0) {
        Serial.println(); // Newline, preparing to display the answer
        
        // Send request
        sendToLLM(inputBuffer);
        
        // Clear buffer and reset interface
        inputBuffer = "";
        Serial.print("Q: "); 
      }
    }
    // Handle general characters (excluding non-printable characters)
    else if (isPrintable(inChar)) {
      inputBuffer += inChar; // Add to buffer
      Serial.print(inChar);  // [KEY] Display immediately on the terminal
    }
  }
}

void sendToLLM(String question) {
  if (WiFi.status() == WL_CONNECTED) {
    
    WiFiClientSecure client;
    client.setInsecure(); // Ignore SSL
    client.setTimeout(300000);
    HTTPClient http;
    
    // [Key Fix] Set timeout to 300000ms (300 seconds)
    // Solves error code -11 (Timeout)
    http.setTimeout(300000); 

    if (http.begin(client, apiUrl)) {
      http.addHeader("Content-Type", "application/json");

      // Simple JSON formatting (handling double quotes)
      question.replace("\"", "\\\""); 
      // Remove possible residual control characters
      question.trim(); 
      
      String payload = "{\"question\": \"" + question + "\"}";

      // Send POST
      int httpCode = http.POST(payload);

      if (httpCode > 0) {
        String responsePayload = http.getString();
        
        // Parse JSON
        DynamicJsonDocument doc(4096); // Slightly increase memory space
        DeserializationError error = deserializeJson(doc, responsePayload);

        if (!error) {
          const char* answer = doc["response"];
          Serial.print("A: ");
          if (answer) {
            Serial.println(answer);
          } else {
            Serial.println("(No response content)");
          }
        } else {
          Serial.print("JSON Parsing failed (content may be too long or format error): ");
          Serial.println(error.c_str());
          // For debugging: print raw response to see what happened
          // Serial.println(responsePayload); 
        }

      } else {
        Serial.print("HTTP Request failed, error code: ");
        Serial.println(httpCode);
        Serial.println("(-11 = Timeout, please check network or model response speed)");
      }
      http.end();
    } else {
      Serial.println("Cannot connect to server");
    }
  } else {
    Serial.println("WiFi Not Connected");
  }
  Serial.println(""); // Empty line
}
