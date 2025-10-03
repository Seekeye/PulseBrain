#!/usr/bin/env python3
"""
Railway Bot - Bot principal con health check integrado
Ejecuta el bot completo + responde a /health
"""

import asyncio
import sys
import os
import threading
import http.server
import socketserver
import json
from datetime import datetime

# Agregar el directorio backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
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
        pass

def start_health_server():
    """Iniciar health check server en background"""
    try:
        port = int(os.environ.get('PORT', 8000))
        with socketserver.TCPServer(("", port), HealthHandler) as httpd:
            print(f"🏥 Health check server started on port {port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Health check server error: {e}")

async def start_main_bot():
    """Iniciar bot principal"""
    try:
        from continuous_bot_enhanced import EnhancedContinuousBot
        
        bot = EnhancedContinuousBot()
        await bot.initialize()
        await bot.run()
    except Exception as e:
        print(f"❌ Main bot error: {e}")

def main():
    print("🚀 Starting CryptoPulse Pro Bot with Health Check...")
    
    # Iniciar health server en thread separado
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Esperar un poco para que el health server se inicie
    import time
    time.sleep(2)
    
    # Iniciar bot principal
    try:
        asyncio.run(start_main_bot())
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

