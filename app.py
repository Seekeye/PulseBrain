#!/usr/bin/env python3
"""
Ultra Simple App for Railway
Solo health check + bot básico
"""

import os
import sys
import json
from datetime import datetime

# Agregar backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def health_check():
    """Health check simple"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "CryptoPulse Pro Bot",
        "version": "1.0.0"
    }

def main():
    """Función principal"""
    print("🚀 Starting CryptoPulse Pro Bot...")
    
    # Verificar que estamos en Railway
    port = os.environ.get('PORT', 8000)
    print(f"🌐 Running on port {port}")
    
    # Importar y ejecutar bot
    try:
        from continuous_bot_enhanced import EnhancedContinuousBot
        import asyncio
        
        async def run_bot():
            bot = EnhancedContinuousBot()
            await bot.initialize()
            await bot.run()
        
        # Ejecutar bot
        asyncio.run(run_bot())
        
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        # Mantener el proceso vivo para health check
        import time
        while True:
            time.sleep(60)

if __name__ == "__main__":
    main()
