_status_callback = None

def set_status_callback(cb):
    global _status_callback
    _status_callback = cb

def log_status(message: str):
    print(message)
    if _status_callback:
        try:
            _status_callback(message)
        except Exception:
            pass