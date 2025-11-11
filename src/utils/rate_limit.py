# rate_limit.py
import time, threading

_qps_lock = threading.Lock()
_events = []

def respect_qps(max_qps: int = 5):
    """
    Block briefly if we already made max_qps calls in the last second.
    Call this immediately before your outbound API request.
    """
    if max_qps <= 0:
        return
    with _qps_lock:
        now = time.time()
        # Drop events older than 1s
        while _events and (now - _events[0] > 1.0):
            _events.pop(0)
        if len(_events) >= max_qps:
            sleep_for = 1.0 - (now - _events[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        _events.append(time.time())
