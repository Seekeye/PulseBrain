#!/usr/bin/env python3
"""
Railway Health Check - Cumple con especificaciones de Railway
https://docs.railway.com/guides/healthchecks
"""

import os
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class RailwayHealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Railway busca específicamente /health
        if self.path == '/health':
            response = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "CryptoPulse Pro Bot",
                "version": "1.0.0"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    # Railway inyecta la variable PORT
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🏥 Starting Railway health check on port {port}")
    print(f"🌐 Health endpoint: http://localhost:{port}/health")
    print(f"⏰ Railway will check from: healthcheck.railway.app")
    
    try:
        server = HTTPServer(('', port), RailwayHealthHandler)
        print(f"✅ Health check server ready!")
        server.serve_forever()
    except Exception as e:
        print(f"❌ Health check server error: {e}")

if __name__ == "__main__":
    main()
