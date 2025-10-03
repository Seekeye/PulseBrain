#!/usr/bin/env python3
"""
Main entry point for Railway
Starts health check server + bot
"""

import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def start_health_server():
    """Start health check server"""
    try:
        port = int(os.environ.get('PORT', 8000))
        server = HTTPServer(('', port), HealthHandler)
        print(f"🏥 Health check server started on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"❌ Health server error: {e}")

def start_bot():
    """Start main bot"""
    try:
        print("🤖 Starting bot...")
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
    print("🚀 Starting CryptoPulse Pro Bot with Health Check...")
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    
    # Wait a bit for bot to initialize
    time.sleep(2)
    
    # Start health check server (this will block)
    start_health_server()

if __name__ == "__main__":
    main()