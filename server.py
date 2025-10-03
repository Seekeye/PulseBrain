#!/usr/bin/env python3
"""
Simple HTTP Server with Health Check for Railway
"""

import os
import sys
import json
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

# Agregar backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

class HealthHandler(BaseHTTPRequestHandler):
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

def start_bot():
    """Iniciar bot en thread separado"""
    try:
        from continuous_bot_enhanced import EnhancedContinuousBot
        import asyncio
        
        async def run_bot():
            bot = EnhancedContinuousBot()
            await bot.initialize()
            await bot.run()
        
        asyncio.run(run_bot())
    except Exception as e:
        print(f"❌ Bot error: {e}")

def main():
    port = int(os.environ.get('PORT', 8000))
    
    # Iniciar bot en thread separado
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    
    # Iniciar servidor HTTP
    try:
        server = HTTPServer(('', port), HealthHandler)
        print(f"🏥 Health check server started on port {port}")
        print(f"🤖 Bot started in background")
        server.serve_forever()
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
