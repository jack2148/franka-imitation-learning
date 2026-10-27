import { useEffect } from "react";
import { apiUrl } from "./api";

export function useConsoleLifecycle(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    let clientId=sessionStorage.getItem('mimic-ui-client');
    if (!clientId) { clientId=crypto.randomUUID();sessionStorage.setItem('mimic-ui-client',clientId); }
    const body=JSON.stringify({client_id:clientId});
    const heartbeat=() => { void fetch(apiUrl('/api/ui-session/heartbeat'),{
      method:'POST',headers:{'Content-Type':'application/json'},body,keepalive:true,
    }).catch(()=>{}); };
    const onClose=() => {
      navigator.sendBeacon(apiUrl('/api/ui-session/close'),new Blob([body],{type:'text/plain'}));
    };
    heartbeat();
    const timer=window.setInterval(heartbeat,2000);
    window.addEventListener('pagehide',onClose);
    return () => { window.clearInterval(timer);window.removeEventListener('pagehide',onClose); };
  },[enabled]);
}
