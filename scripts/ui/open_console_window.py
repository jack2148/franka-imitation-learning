#!/usr/bin/env python3
"""Own one dedicated UI browser; closing it shuts down its console session."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('url');parser.add_argument('backend');parser.add_argument('--reopen',action='store_true')
    args=parser.parse_args()
    browser=os.environ.get('MIMIC_UI_BROWSER') or shutil.which('google-chrome') or shutil.which('google-chrome-stable') or shutil.which('chromium')
    if not browser:
        subprocess.run(['xdg-open',args.url],check=False)
        return 0  # pagehide/session beacons own the fallback tab lifecycle.
    port=args.url.rsplit(':',1)[-1].rstrip('/')
    profile=Path('/tmp')/('mimic-console-browser-'+port)
    command=[browser,'--app='+args.url,'--user-data-dir='+str(profile),
             '--no-first-run','--no-default-browser-check','--disable-background-mode']
    child=subprocess.Popen(command)
    if args.reopen:return 0
    def stop(_signum,_frame):
        if child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        try:
            request=urllib.request.Request(args.backend+'/api/shutdown',data=b'{}',headers={'Content-Type':'application/json'})
            urllib.request.urlopen(request,timeout=3).close()
        except OSError:pass
    return 0


if __name__=='__main__':raise SystemExit(main())
