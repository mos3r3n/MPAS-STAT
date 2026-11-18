"""Figure-job execution shared by the model-space and observation-space
graphics stages.

A *job* is a ``(func, args, kwargs)`` tuple whose function collects one
figure's data and draws it.  Jobs are self-contained and picklable, so they
can run sequentially or across a process pool.
"""
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


def lead_cfg(cfg, lead_hour):
    """Shallow cfg copy carrying a per-lead filename suffix (e.g. "_f006").

    _save_fig appends this suffix to every figure's base filename, so each
    forecast lead hour gets its own file (instead of overwriting the
    previous lead's) while landing in the SAME output folder as the
    corresponding background-mode plot type, e.g.
    profiles_1d/profile_lat_bin_profile_rms_f006.pdf next to
    profiles_1d/profile_lat_bin_profile_rms.pdf.
    """
    new_cfg = dict(cfg)
    new_cfg["_lead_suffix"] = f"_f{int(lead_hour):03d}"
    return new_cfg


def resolve_n_workers(cfg):
    """Worker count from cfg["n_workers"] (or cfg["parallel"]["n_workers"]), >= 1."""
    return max(1, int(cfg.get("n_workers", cfg.get("parallel", {}).get("n_workers", 1))))


def run_job(job):
    """Run one (func, args, kwargs) job, catching any error so a single bad
    figure never aborts the rest of the batch -- this matters even more once
    jobs run in a process pool, where an unhandled exception would otherwise
    just silently drop that figure and any jobs still queued behind it.

    Returns ``(True, None)`` on success or ``(False, traceback_text)``.
    """
    func, args, kwargs = job
    try:
        func(*args, **kwargs)
        return (True, None)
    except Exception:
        return (False, traceback.format_exc())


def run_jobs(jobs, n_workers):
    """Execute collect+plot jobs sequentially, or across a process pool
    when n_workers > 1.

    Process-based (not thread-based) parallelism is required here: each job
    creates and saves its own matplotlib figure, and a single process's
    pyplot state is not safe for concurrent figure creation. Each job's
    func/args/kwargs must be picklable (module-level functions plus plain
    dicts/lists/numpy arrays).
    """
    n_jobs = len(jobs)
    if n_jobs == 0:
        return

    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        for job in jobs:
            ok, err = run_job(job)
            if not ok:
                print(f"ERROR in {job[0].__name__}:\n{err}")
        return

    n_workers = min(n_workers, n_jobs)
    print(f"Running {n_jobs} plotting jobs across {n_workers} worker processes...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            ok, err = future.result()
            if not ok:
                print(f"ERROR in {job[0].__name__}:\n{err}")
