import atexit, sys
def _peak():
    m = sys.modules
    try:
        if 'torch' in m and m['torch'].cuda.is_available() and m['torch'].cuda.is_initialized():
            t = m['torch']; f, tot = t.cuda.mem_get_info()
            print(f"[PEAK] torch_alloc_GiB={t.cuda.max_memory_allocated()/2**30:.2f} torch_reserved_GiB={t.cuda.max_memory_reserved()/2**30:.2f} device_used_at_exit_GiB={(tot-f)/2**30:.2f} device_total_GiB={tot/2**30:.2f}", flush=True)
    except Exception as e: print('[PEAK] torch err', e, flush=True)
    try:
        if 'jax' in m:
            for d in m['jax'].local_devices():
                s = d.memory_stats() or {}
                print(f"[PEAK] jax {d} peak_GiB={s.get('peak_bytes_in_use',0)/2**30:.2f} limit_GiB={s.get('bytes_limit',0)/2**30:.2f}", flush=True)
    except Exception as e: print('[PEAK] jax err', e, flush=True)
atexit.register(_peak)
