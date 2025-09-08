import os, sys, time, socket, webbrowser, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
APP  = os.path.join(HERE, "app.py")

def find_free_port(start=8501, limit=20):
    for p in range(start, start+limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start

def wait_for_port(port, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.25):
                return True
        except OSError:
            time.sleep(0.25)
    return False

def main():
    port = find_free_port(8501)
    cmd = [sys.executable, "-m", "streamlit", "run", APP,
           "--server.headless=true",
           "--browser.gatherUsageStats=false",
           f"--server.port={port}",
           "--server.address=127.0.0.1"]
    # start Streamlit
    proc = subprocess.Popen(cmd)
    # open browser once ready
    if wait_for_port(port, 20):
        webbrowser.open(f"http://127.0.0.1:{port}", new=1)
    proc.wait()

if __name__ == "__main__":
    main()
