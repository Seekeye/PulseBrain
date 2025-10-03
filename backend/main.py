#!/usr/bin/env python3
"""
Main entry point for Railway
Ultra simple health check that works
"""

import os
import socket
import threading
import time

def health_check_server():
    """Start health check server"""
    port = int(os.environ.get('PORT', 8000))
    print(f"🏥 Starting health check on port {port}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', port))
    sock.listen(1)
    
    print(f"✅ Health check server ready on port {port}")
    
    while True:
        try:
            conn, addr = sock.accept()
            data = conn.recv(1024).decode()
            
            if '/health' in data:
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK"
                conn.send(response.encode())
                print(f"✅ Health check responded to {addr}")
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n"
                conn.send(response.encode())
            
            conn.close()
        except Exception as e:
            print(f"❌ Health check error: {e}")

def start_bot():
    """Start main bot in background"""
    try:
        print("🤖 Starting bot in background...")
        import sys
        sys.path.append('backend')
        
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
    print("🚀 Starting CryptoPulse Pro Bot...")
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    
    # Wait for bot to initialize
    time.sleep(3)
    
    # Start health check server (this will block)
    health_check_server()

if __name__ == "__main__":
    main()

    # Start health check server (this will block)
    start_health_server()

if __name__ == "__main__":
    main()
