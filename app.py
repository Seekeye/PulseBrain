"""
Ultra Simple Health Check for Railway
Solo responde a /health sin dependencias
"""

import os
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting health check on port {port}")
    
    server = HTTPServer(('', port), HealthHandler)
    server.serve_forever()
