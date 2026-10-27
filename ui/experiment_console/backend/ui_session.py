"""Last-window close detection with a grace period for page reloads."""
import threading
import time


class UISessions:
    def __init__(self, grace_s=3., clock=time.monotonic):
        self.grace_s=grace_s;self.clock=clock;self.clients={};self.lock=threading.Lock()

    def heartbeat(self, client_id):
        if not isinstance(client_id,str) or not client_id or len(client_id)>128:raise ValueError('Invalid UI session')
        with self.lock:self.clients[client_id]=None

    def close(self, client_id):
        with self.lock:
            if client_id in self.clients:self.clients[client_id]=self.clock()

    def should_shutdown(self):
        with self.lock:
            return bool(self.clients) and all(closed is not None and self.clock()-closed>=self.grace_s for closed in self.clients.values())
