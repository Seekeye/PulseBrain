#!/usr/bin/env python3
"""
Health Check Endpoint for Railway
Simple HTTP server to respond to health checks
"""

import http.server
import socketserver
import threading
import time
import json
from datetime import datetime

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            # Health check response
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "CryptoPulse Pro Bot",
                "version": "1.0.0"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def start_health_server(port=8000):
    """Start health check server in background"""
    try:
        with socketserver.TCPServer(("", port), HealthCheckHandler) as httpd:
            print(f"🏥 Health check server started on port {port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Health check server error: {e}")

def run_health_server():
    """Run health server in background thread"""
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    print("✅ Health check server running in background")

if __name__ == "__main__":
    run_health_server()
    # Keep the script running
    while True:
        time.sleep(1)
