from flask import Flask
import socket

app = Flask(__name__)

@app.route('/')
def hello():
    hostname = socket.gethostname()
    ipa = socket.gethostbyname(hostname)
    return f"Hello, you're visiting page {hostname} - {ipa}!"

if __name__=='__main__':
    app.run(host='0.0.0.0', port=3000)