import os
import socket

def start_server():
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting health check on port {port}")
    
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', port))
    sock.listen(1)
    
    while True:
        conn, addr = sock.accept()
        try:
            data = conn.recv(1024).decode()
            if '/health' in data:
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK"
                conn.send(response.encode())
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n"
                conn.send(response.encode())
        except:
            pass
        finally:
            conn.close()

if __name__ == "__main__":
    start_server()
