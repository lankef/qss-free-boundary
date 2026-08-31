"""Accuracy and memory check for quadcoil's ``jac_chunk_size`` on a GPU.

``jac_chunk_size`` splits the KKT adjoint VJP over the metric rows. That is
exact in real arithmetic, so the only question is how much the returned
derivatives move in floating point, and whether the memory saving is real.
A CPU run of a small case put the accuracy cost at ~1e-10 relative, but a
GPU run of the same case disagreed by ~40%, so this has to be measured on
the device that will actually be used.

Three things get measured:

1. ``intrace`` -- the decisive isolation. Inside a single compiled trace,
   sharing one adjoint matrix ``V``, the monolithic ``vmap`` is compared
   against the chunked loop. Nothing but the chunk boundaries can differ,
   so any disagreement here is chunking and chunking alone. Conditioning
   of the adjoint solve is reported alongside, since that is what sets the
   amplification factor.

2. ``single`` -- one end-to-end ``quadcoil`` call at a given chunk size,
   saving the derivatives and the peak device memory. Each of these runs in
   its own process so the peak memory number is not polluted by earlier
   runs (the allocator's peak counter is a running maximum).

3. the comparison table -- what a user actually sees end to end, reported
   with metrics that separate "the meaningful entries disagree" from
   "entries at the noise floor fail a pure relative test".

Everything is repeated with and without a small ``f_K`` regularization,
because the size of the chunking error is expected to track how
ill-conditioned the adjoint solve is rather than the chunk size as such.

Usage
-----
    python -u test_chunking.py            # everything (spawns subprocesses)
    python -u test_chunking.py intrace    # part 1 only, in this process
    python -u test_chunking.py single 8   # part 2 for jac_chunk_size=8

Environment knobs (all optional)::

    QUADCOIL_TESTS  path to quadcoil's tests/ dir (needs surfaces.json)
    OUTDIR          where to write the .npz derivative dumps
    MPOL, NTOR      current-potential resolution     (default 4, 4)
    NPHI, NTHETA    quadrature points per period     (default 8, 8)
    MAXITER         solver iterations                (default 200)
    CHUNKS          comma-separated sweep list       (default 1,5,8,20,40)
    REG             f_K weights to test, 0 = none    (default 0,1e-14)
"""
import os
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
QUADCOIL_TESTS = os.environ.get(
    'QUADCOIL_TESTS', '/scratch/lf2869/code/quadcoil/tests'
)
OUTDIR = os.environ.get('OUTDIR', os.path.join(HERE, 'out'))

MPOL = int(os.environ.get('MPOL', 4))
NTOR = int(os.environ.get('NTOR', 4))
NPHI = int(os.environ.get('NPHI', 8))
NTHETA = int(os.environ.get('NTHETA', 8))
MAXITER = int(os.environ.get('MAXITER', 200))
CHUNKS = [int(c) for c in os.environ.get('CHUNKS', '1,5,8,20,40').split(',')]
REGS = [float(r) for r in os.environ.get('REG', '0,1e-14').split(',')]

# surfaces.json is loaded by relative path inside load_test_data.
sys.path.insert(0, QUADCOIL_TESTS)
os.chdir(QUADCOIL_TESTS)


def _tag(reg):
    return 'unreg' if reg == 0.0 else f'reg{reg:g}'


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
def build_kwargs(reg):
    """quadcoil kwargs for a ``phi_dofs`` (vector metric) adjoint."""
    import jax.numpy as jnp
    from load_test_data import load_data

    _, plasma_surface, _, _, _ = load_data()
    nfp = plasma_surface.nfp
    kwargs = dict(
        nfp=nfp,
        stellsym=plasma_surface.stellsym,
        mpol=MPOL,
        ntor=NTOR,
        plasma_dofs=plasma_surface.get_dofs(),
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        net_poloidal_current_amperes=11884578.094260072,
        net_toroidal_current_amperes=0.,
        plasma_coil_distance=plasma_surface.minor_radius(),
        metric_name=('phi_dofs',),
        plasma_quadpoints_phi=jnp.linspace(0., 1. / nfp, NPHI, endpoint=False),
        plasma_quadpoints_theta=jnp.linspace(0., 1., NTHETA, endpoint=False),
        winding_quadpoints_phi=jnp.linspace(
            0., 1., NPHI * nfp, endpoint=False),
        winding_quadpoints_theta=jnp.linspace(0., 1., NTHETA, endpoint=False),
        maxiter=MAXITER,
    )
    if reg == 0.0:
        kwargs['objective_name'] = 'f_B'
    else:
        # A tiny f_K term barely changes the optimum but pulls the Hessian
        # away from singular, which is the knob we want to vary.
        kwargs['objective_name'] = ('f_B', 'f_K')
        kwargs['objective_weight'] = jnp.array([1., reg])
        kwargs['objective_unit'] = (None, None)
    return kwargs


def peak_gib():
    """Peak device bytes seen by the JAX allocator, in GiB (0 on CPU)."""
    import jax
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return 0.0
    return stats.get('peak_bytes_in_use', 0) / 1024 ** 3


def describe_device():
    import jax
    d = jax.devices()[0]
    print(f'jax {jax.__version__}  device {d.platform}:{d.device_kind}',
          flush=True)


# --------------------------------------------------------------------------
# Part 1: in-trace isolation
# --------------------------------------------------------------------------
def run_intrace():
    """Compare chunked vs monolithic vmap inside one trace, sharing one V."""
    import jax
    import jax.numpy as jnp
    from jax import debug, jacrev, vjp, vmap
    import quadcoil.quadcoil  # noqa: F401  (populate sys.modules)
    from quadcoil import quadcoil

    # The package __init__ rebinds the attribute `quadcoil` to the function,
    # so `import quadcoil.quadcoil as m` would give the function, not the
    # module. Go through sys.modules to reach the real module.
    qcmod = sys.modules['quadcoil.quadcoil']
    original = qcmod.adjoint_kkt

    describe_device()

    def diag_adjoint_kkt(f_metrics_flat, stationarity_data, y_flat, verbose,
                         jac_chunk_size=None):
        x_opt = stationarity_data['x_flat_precond']
        all_values = f_metrics_flat(x_opt, y_flat)
        J_x = jacrev(f_metrics_flat, argnums=0)(x_opt, y_flat)
        J_y = jacrev(f_metrics_flat, argnums=1)(x_opt, y_flat)

        if stationarity_data['constrained']:
            raise RuntimeError(
                'this diagnostic only covers the unconstrained KKT branch'
            )
        H_mat = stationarity_data['H_mat']
        grad_y_stationarity = stationarity_data['grad_y_stationarity']

        sv = jnp.linalg.svd(H_mat, compute_uv=False)
        V = jnp.linalg.lstsq(H_mat, J_x.T)[0].T
        res = (jnp.linalg.norm(H_mat @ V.T - J_x.T)
               / jnp.linalg.norm(J_x))

        _, vjp_fn = vjp(grad_y_stationarity, y_flat)
        f = lambda v: vjp_fn(v)[0]
        rows_full = vmap(f)(V)
        n = V.shape[0]

        debug.print('  n_metric_rows {n}   H {h}', n=n, h=H_mat.shape[0])
        debug.print('  cond(H) {c:.4e}   lstsq rel residual {r:.4e}',
                    c=sv[0] / sv[-1], r=res)
        debug.print('  max|J_x| {a:.4e}  max|J_y| {d:.4e}  max|V| {b:.4e}  '
                    'max|rows| {c:.4e}',
                    a=jnp.max(jnp.abs(J_x)), d=jnp.max(jnp.abs(J_y)),
                    b=jnp.max(jnp.abs(V)), c=jnp.max(jnp.abs(rows_full)))

        scale = jnp.max(jnp.abs(rows_full))
        for c in [c for c in CHUNKS if c < n] + [n]:
            rows_c = jnp.concatenate(
                [vmap(f)(V[i:i + c]) for i in range(0, n, c)], axis=0
            )
            d = jnp.abs(rows_full - rows_c)
            debug.print(
                '  chunk {c:>4}   max abs diff {a:.4e}   rel-to-max {b:.4e}',
                c=c, a=jnp.max(d), b=jnp.max(d) / scale,
            )

        return all_values, J_y - rows_full, {}

    try:
        qcmod.adjoint_kkt = diag_adjoint_kkt
        for reg in REGS:
            print(f'\n[in-trace] {_tag(reg)}  (one compile, one shared V)',
                  flush=True)
            out, _, _, _ = quadcoil(**build_kwargs(reg))
            jax.block_until_ready(out)
    finally:
        qcmod.adjoint_kkt = original


# --------------------------------------------------------------------------
# Part 2: one end-to-end run
# --------------------------------------------------------------------------
def run_single(chunk, reg):
    """One quadcoil call; save derivatives and report peak memory."""
    import jax
    import jax.numpy as jnp
    from quadcoil import quadcoil

    describe_device()
    kwargs = build_kwargs(reg)
    out, _, dofs, _ = quadcoil(**kwargs, jac_chunk_size=chunk)
    out, dofs = jax.block_until_ready((out, dofs))

    grads = out['phi_dofs']['grad']
    payload = {f'grad__{k}': np.asarray(v) for k, v in grads.items()}
    payload['value'] = np.asarray(out['phi_dofs']['value'])
    payload['phi'] = np.asarray(dofs['phi'])

    os.makedirs(OUTDIR, exist_ok=True)
    name = f'{_tag(reg)}_chunk{"none" if chunk is None else chunk}.npz'
    np.savez(os.path.join(OUTDIR, name), **payload)

    print(f'chunk={chunk} reg={reg:g}  n_rows={payload["value"].size}  '
          f'peak={peak_gib():.3f} GiB', flush=True)
    for k, v in grads.items():
        print(f'    {k:40s} max|g| = '
              f'{float(jnp.max(jnp.abs(v))):.6e}', flush=True)


# --------------------------------------------------------------------------
# Part 3: comparison
# --------------------------------------------------------------------------
def compare(reg):
    """Compare every chunked run against the unchunked reference."""
    tag = _tag(reg)
    ref_path = os.path.join(OUTDIR, f'{tag}_chunknone.npz')
    if not os.path.exists(ref_path):
        print(f'[compare] {tag}: no unchunked reference, skipping')
        return
    ref = np.load(ref_path)

    print(f'\n[compare] {tag}: chunked vs unchunked, end to end')
    print('  rel_to_max  = max|a-b| / max|a|, per gradient leaf')
    print('  elem_rel    = worst |a-b|/|a| over entries above 1e-8 of the '
          'leaf max')
    print('  n_fail      = entries failing allclose(rtol=1e-7, atol=0)')
    print(f'  {"chunk":>6} {"phi bitwise":>12} {"rel_to_max":>12} '
          f'{"elem_rel":>12} {"n_fail":>8} {"n_tot":>8} {"peak GiB":>9}')

    for chunk in CHUNKS + [None]:
        name = f'{tag}_chunk{"none" if chunk is None else chunk}.npz'
        path = os.path.join(OUTDIR, name)
        if not os.path.exists(path):
            continue
        cur = np.load(path)
        same_phi = bool(np.array_equal(cur['phi'], ref['phi']))

        worst_rel_to_max = 0.0
        worst_elem = 0.0
        n_fail = 0
        n_tot = 0
        for key in ref.files:
            if not key.startswith('grad__'):
                continue
            a = ref[key]
            b = cur[key]
            amax = float(np.max(np.abs(a)))
            d = np.abs(a - b)
            if amax > 0:
                worst_rel_to_max = max(worst_rel_to_max,
                                       float(np.max(d)) / amax)
                big = np.abs(a) > 1e-8 * amax
                if big.any():
                    worst_elem = max(
                        worst_elem,
                        float(np.max(d[big] / np.abs(a[big]))),
                    )
            n_fail += int(np.sum(d > 1e-7 * np.abs(b)))
            n_tot += int(a.size)

        peak = PEAKS.get((tag, chunk), float('nan'))
        print(f'  {str(chunk):>6} {str(same_phi):>12} '
              f'{worst_rel_to_max:>12.3e} {worst_elem:>12.3e} '
              f'{n_fail:>8} {n_tot:>8} {peak:>9.3f}')


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
PEAKS = {}


def spawn(args):
    """Run this script in a fresh process so peak memory is per-config."""
    cmd = [sys.executable, '-u', os.path.abspath(__file__)] + args
    print(f'\n$ {" ".join(args)}', flush=True)
    proc = subprocess.run(cmd, cwd=QUADCOIL_TESTS, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    print(proc.stdout, end='', flush=True)
    if proc.returncode != 0:
        print(f'  -> FAILED (exit {proc.returncode})', flush=True)
    return proc.stdout, proc.returncode


def main():
    argv = sys.argv[1:]

    if argv and argv[0] == 'intrace':
        run_intrace()
        return
    if argv and argv[0] == 'single':
        chunk = None if argv[1] == 'none' else int(argv[1])
        reg = float(argv[2]) if len(argv) > 2 else 0.0
        run_single(chunk, reg)
        return

    print('=' * 74)
    print(f'jac_chunk_size check   mpol={MPOL} ntor={NTOR} '
          f'grid={NPHI}x{NTHETA} maxiter={MAXITER}')
    print(f'chunks={CHUNKS} regs={REGS}')
    print(f'quadcoil tests: {QUADCOIL_TESTS}')
    print(f'output:         {OUTDIR}')
    print('=' * 74)

    spawn(['intrace'])

    for reg in REGS:
        # Unchunked last: it is the memory-heaviest and the most likely to
        # die, and the chunked runs are the ones we would lose.
        for chunk in [str(c) for c in CHUNKS] + ['none']:
            out, rc = spawn(['single', chunk, repr(reg)])
            if rc != 0:
                continue
            key = (_tag(reg), None if chunk == 'none' else int(chunk))
            for line in out.splitlines():
                if 'peak=' in line:
                    PEAKS[key] = float(
                        line.split('peak=')[1].split()[0]
                    )

    for reg in REGS:
        compare(reg)

    print('\nReading the table: rel_to_max is the honest accuracy cost of '
          'chunking.\nelem_rel is dominated by noise-floor entries and will '
          'look alarming even\nwhen nothing is wrong. phi bitwise=True '
          'confirms the forward solve is\nuntouched, so any difference is '
          'in the adjoint alone.')


if __name__ == '__main__':
    main()
