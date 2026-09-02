"""Accuracy and memory check for quadcoil's ``jac_chunk_size`` on a GPU.

``jac_chunk_size`` splits the KKT adjoint VJP over the metric rows. That is
exact in real arithmetic, so the only question is how much the returned
derivatives move in floating point, and whether the memory saving is real.
A CPU run of a small case put the accuracy cost at ~1e-10 relative, but a
GPU run of the same case disagreed by ~40%, so this has to be measured on
the device that will actually be used.

Why this was revisited
----------------------
``_chunked_vjp_rows`` used to chunk with a Python ``for`` loop, on the
recorded belief that ``lax.map`` gave catastrophically wrong derivatives
once more than one full batch was needed. That loop does not bound memory:
JAX emits it as one traced computation with ``n_batches`` call sites and
XLA's call inliner expands all of them, so every chunk's accumulators are
allocated together. The HLO dump for job 16733787 (220 metric rows,
``jac_chunk_size=32``) showed 23 co-live ``f64[32,64,34,3,220]``
accumulators making up 11.3 GiB of an 18.2 GiB peak. ``lax.map`` keeps the
chunk loop as a real ``while``, so it is the fix -- provided the
wrong-derivative claim does not survive a careful re-test.

It never was carefully tested. The old in-trace diagnostic *raised* for
``constrained=True``, which is the branch production actually uses, and
its default chunk sizes all divided the row count evenly, so the
"``n_batches > 1`` plus a remainder" regime where the failure was
reportedly visible was never hit. Both gaps are closed here.

Four things get measured:

1. ``intrace`` -- the decisive isolation. Inside a single compiled trace,
   sharing one adjoint matrix ``V``, the monolithic ``vmap`` is compared
   against a Python loop of ``vmap`` calls (the old implementation) and
   against ``lax.map(batch_size=...)`` (the new one). Runs on both the
   unconstrained and the constrained KKT branch. Conditioning of the
   adjoint solve is reported alongside, since that is what sets the
   amplification factor.

2. the batching regime, printed per chunk size as ``n = n_batches * c +
   remainder``, so it is visible at a glance whether the sweep reproduces
   production's shape (220 = 6*32 + 28) or only hits even divisions.

3. ``single`` -- one end-to-end ``quadcoil`` call at a given chunk size and
   a given implementation (``map`` or ``loop``), saving the derivatives and
   the peak device memory. Each of these runs in its own process so the
   peak memory number is not polluted by earlier runs (the allocator's peak
   counter is a running maximum).

4. the comparison table -- what a user actually sees end to end, reported
   with metrics that separate "the meaningful entries disagree" from
   "entries at the noise floor fail a pure relative test". ``map`` and
   ``loop`` are each compared against the unchunked reference and against
   each other at matching chunk size.

Everything is repeated with and without a small ``f_K`` regularization,
because the size of the chunking error is expected to track how
ill-conditioned the adjoint solve is rather than the chunk size as such.

Usage
-----
    python -u test_chunking.py                      # everything (subprocesses)
    python -u test_chunking.py intrace con          # part 1, constrained only
    python -u test_chunking.py single 6 0.0 map con # one end-to-end run

Environment knobs (all optional)::

    QUADCOIL_TESTS  path to quadcoil's tests/ dir (needs surfaces.json)
    OUTDIR          where to write the .npz derivative dumps
    MPOL, NTOR      current-potential resolution     (default 4, 4)
    NPHI, NTHETA    quadrature points per period     (default 8, 8)
    MAXITER         solver iterations                (default 200)
    CHUNKS          comma-separated sweep list       (default 1,5,6,7,20,40)
    REG             f_K weights to test, 0 = none    (default 0,1e-14)
    MODES           con,uncon                        (default con,uncon)
    IMPLS           map,loop                         (default map,loop)
    CON_FRAC        bound as a fraction of the       (default 0.5)
                    unconstrained f_max_B2_self
    CON_VALUE       explicit f_max_B2_self bound, skips the calibration solve
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
# The metric is 'phi_dofs', so the adjoint row count is the current-potential
# dof count: mpol*(2*ntor+1) + ntor, i.e. 40 at the default 4x4 (production
# is 220 at 10x10). These defaults are chosen against n = 40 so the sweep
# covers both even divisions and remainders, and so that c = 6 gives
# 40 = 6*6 + 4 -- six full batches plus a partial one, the same shape as
# production's 220 = 6*32 + 28. See print_batch_regime.
CHUNKS = [int(c) for c in os.environ.get('CHUNKS', '1,5,6,7,20,40').split(',')]
REGS = [float(r) for r in os.environ.get('REG', '0,1e-14').split(',')]
MODES = [m.strip() for m in os.environ.get('MODES', 'con,uncon').split(',')]
IMPLS = [m.strip() for m in os.environ.get('IMPLS', 'map,loop').split(',')]
CON_FRAC = float(os.environ.get('CON_FRAC', 0.5))

# surfaces.json is loaded by relative path inside load_test_data.
sys.path.insert(0, QUADCOIL_TESTS)
os.chdir(QUADCOIL_TESTS)


def _tag(reg, constrained):
    return ('con' if constrained else 'uncon') + '_' + (
        'unreg' if reg == 0.0 else f'reg{reg:g}'
    )


def _npz_name(reg, constrained, chunk, impl):
    c = 'none' if chunk is None else chunk
    return f'{_tag(reg, constrained)}_chunk{c}_{impl}.npz'


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
_CON_VALUE = {}


def constraint_target():
    """An ``f_max_B2_self`` bound that is actually active at the optimum.

    An inactive inequality leaves ``z = 0``, which makes the lower block of
    ``J_KKT`` vanish and turns the constrained branch back into the
    unconstrained one numerically -- the opposite of what we want to test.
    So the bound is calibrated against a cheap ``value_only`` unconstrained
    solve rather than guessed.
    """
    if 'v' in _CON_VALUE:
        return _CON_VALUE['v']
    env = os.environ.get('CON_VALUE')
    if env is not None:
        _CON_VALUE['v'] = float(env)
        return _CON_VALUE['v']

    from quadcoil import quadcoil

    kwargs = build_kwargs(0.0, constrained=False)
    kwargs['metric_name'] = ('f_max_B2_self',)
    out, _, _, _ = quadcoil(**kwargs, value_only=True)
    free = float(out['f_max_B2_self']['value'])
    _CON_VALUE['v'] = CON_FRAC * free
    print(f'  constraint calibration: unconstrained f_max_B2_self = {free:.6e},'
          f' bound set to {_CON_VALUE["v"]:.6e} ({CON_FRAC:g}x)', flush=True)
    return _CON_VALUE['v']


def build_kwargs(reg, constrained):
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
    if constrained:
        # Mirrors production (shared.py): one smoothed inequality, so
        # m_g = 1 and n_w = n_x + 1, exactly the shape in the HLO dump.
        bound = constraint_target()
        kwargs['constraint_name'] = ('f_max_B2_self',)
        kwargs['constraint_type'] = ('<=',)
        kwargs['constraint_unit'] = (bound,)
        kwargs['constraint_value'] = jnp.array([bound])
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


def print_batch_regime(n, indent='  '):
    """Show how each chunk size splits ``n`` rows.

    The failure the old docstring described was said to need more than one
    full batch. Printing this makes it obvious whether the sweep actually
    reaches that regime, or whether every chunk size divides ``n`` evenly
    and the interesting case is being skipped -- which is what the previous
    defaults (1,5,8,20,40 against n=40) did.
    """
    print(f'{indent}batching of n = {n} adjoint rows:', flush=True)
    for c in sorted({min(c, n) for c in CHUNKS} | {n}):
        nb, rem = divmod(n, c)
        note = ''
        if nb > 1 and rem:
            note = '   <- several full batches + remainder (production shape)'
        elif nb == 1 and rem:
            note = '   <- one full batch + remainder'
        elif nb <= 1:
            note = '   <- single batch, chunking inactive'
        print(f'{indent}  chunk {c:>4}   n_batches {nb:>4}   remainder '
              f'{rem:>4}{note}', flush=True)


# --------------------------------------------------------------------------
# The two chunking implementations
# --------------------------------------------------------------------------
def loop_chunked_vjp_rows(fn, V, jac_chunk_size):
    """The pre-fix implementation, reproduced here for the A/B.

    Kept in the test rather than in the package so the comparison survives
    the package moving on. Identical to what ``_chunked_vjp_rows`` was
    before the ``lax.map`` change, ``optimization_barrier`` included.
    """
    import jax.numpy as jnp
    from jax import lax, vmap

    if jac_chunk_size is None:
        return vmap(fn)(V)
    n = V.shape[0]
    outs = []
    for i in range(0, n, jac_chunk_size):
        chunk_out = vmap(fn)(V[i:i + jac_chunk_size])
        outs.append(lax.optimization_barrier(chunk_out))
    return jnp.concatenate(outs, axis=0)


# --------------------------------------------------------------------------
# Part 1: in-trace isolation
# --------------------------------------------------------------------------
def run_intrace(modes=None):
    """Compare chunked vs monolithic vmap inside one trace, sharing one V."""
    import jax
    import jax.numpy as jnp
    from jax import debug, jacrev, lax, vjp, vmap
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
        n = J_x.shape[0]

        if stationarity_data['constrained']:
            # Mirror the constrained branch of adjoint_kkt exactly:
            # J_KKT^T V^T = [J_x, 0]^T, then VJP of the KKT residual R_y.
            # This is the branch production runs and the one the old
            # diagnostic refused to look at.
            J_KKT_mat = stationarity_data['J_KKT_mat']
            m_g = stationarity_data['m_g']
            R_y = stationarity_data['R_y']
            sys_mat = J_KKT_mat.T
            rhs = jnp.concatenate(
                [J_x, jnp.zeros((n, m_g))], axis=1,
            ).T
            V = jnp.linalg.lstsq(sys_mat, rhs)[0].T
            res = jnp.linalg.norm(sys_mat @ V.T - rhs) / jnp.linalg.norm(J_x)
            sv = jnp.linalg.svd(J_KKT_mat, compute_uv=False)
            _, vjp_fn = vjp(R_y, y_flat)
            sys_name, sys_dim = 'J_KKT', J_KKT_mat.shape[0]
            n_active = jnp.sum(jnp.abs(stationarity_data['z_opt']) > 0)
        else:
            H_mat = stationarity_data['H_mat']
            grad_y_stationarity = stationarity_data['grad_y_stationarity']
            V = jnp.linalg.lstsq(H_mat, J_x.T)[0].T
            res = (jnp.linalg.norm(H_mat @ V.T - J_x.T)
                   / jnp.linalg.norm(J_x))
            sv = jnp.linalg.svd(H_mat, compute_uv=False)
            _, vjp_fn = vjp(grad_y_stationarity, y_flat)
            sys_name, sys_dim = 'H', H_mat.shape[0]
            n_active = None

        f = lambda v: vjp_fn(v)[0]
        rows_full = vmap(f)(V)

        # Static facts print at trace time, so they appear before every
        # debug.print below, which fires at execution time.
        print_batch_regime(n)
        print(f'  n_metric_rows {n}   adjoint system {sys_name} '
              f'({sys_dim}x{sys_dim})', flush=True)
        debug.print('  cond {c:.4e}   lstsq rel residual {r:.4e}',
                    c=sv[0] / sv[-1], r=res)
        if n_active is not None:
            debug.print('  nonzero multipliers {a} (0 would mean the '
                        'constraint is inactive and the test is vacuous)',
                        a=n_active)
        debug.print('  max|J_x| {a:.4e}  max|J_y| {d:.4e}  max|V| {b:.4e}  '
                    'max|rows| {c:.4e}',
                    a=jnp.max(jnp.abs(J_x)), d=jnp.max(jnp.abs(J_y)),
                    b=jnp.max(jnp.abs(V)), c=jnp.max(jnp.abs(rows_full)))

        scale = jnp.max(jnp.abs(rows_full))
        for c in [c for c in CHUNKS if c < n] + [n]:
            nb, rem = divmod(n, c)
            rows_loop = loop_chunked_vjp_rows(f, V, c)
            rows_map = lax.map(f, V, batch_size=c)
            d_loop = jnp.abs(rows_full - rows_loop)
            d_map = jnp.abs(rows_full - rows_map)
            d_lm = jnp.abs(rows_loop - rows_map)
            debug.print(
                '  chunk {c:>4} ({nb} x {c} + {rem})   loop vs full  '
                'max abs {a:.4e}  rel-to-max {b:.4e}',
                c=c, nb=nb, rem=rem,
                a=jnp.max(d_loop), b=jnp.max(d_loop) / scale,
            )
            debug.print(
                '  chunk {c:>4} ({nb} x {c} + {rem})   map  vs full  '
                'max abs {a:.4e}  rel-to-max {b:.4e}',
                c=c, nb=nb, rem=rem,
                a=jnp.max(d_map), b=jnp.max(d_map) / scale,
            )
            debug.print(
                '  chunk {c:>4} ({nb} x {c} + {rem})   map  vs loop  '
                'max abs {a:.4e}  rel-to-max {b:.4e}',
                c=c, nb=nb, rem=rem,
                a=jnp.max(d_lm), b=jnp.max(d_lm) / scale,
            )

        return all_values, J_y - rows_full, {}

    try:
        qcmod.adjoint_kkt = diag_adjoint_kkt
        for mode in (modes if modes is not None else MODES):
            constrained = mode == 'con'
            for reg in REGS:
                print(f'\n[in-trace] {_tag(reg, constrained)}  '
                      f'(one compile, one shared V)', flush=True)
                kwargs = build_kwargs(reg, constrained)
                out, _, _, _ = quadcoil(**kwargs)
                jax.block_until_ready(out)
    finally:
        qcmod.adjoint_kkt = original


# --------------------------------------------------------------------------
# Part 2: one end-to-end run
# --------------------------------------------------------------------------
def run_single(chunk, reg, impl='map', constrained=True):
    """One quadcoil call; save derivatives and report peak memory.

    ``impl`` selects the chunking implementation inside the package:
    ``'map'`` leaves ``_chunked_vjp_rows`` alone, ``'loop'`` swaps in the
    pre-fix Python loop. The in-trace diagnostic isolates a single ``V``;
    this exercises the whole ``_quadcoil_pure`` JIT, which is where the
    original wrong-derivative report came from and where the memory blowup
    was measured.
    """
    import jax
    import jax.numpy as jnp
    import quadcoil.solvers.kkt_adjoint as kkt_mod
    from quadcoil import quadcoil

    describe_device()
    kwargs = build_kwargs(reg, constrained)

    # adjoint_kkt looks _chunked_vjp_rows up in its own module globals, so
    # rebinding the attribute is enough. It only takes effect on a fresh
    # trace, though: _quadcoil_pure's jit cache is keyed on the function
    # object, so a second call in the same process would silently reuse the
    # first implementation's compilation. main() gives every config its own
    # process, which is also what keeps the peak-memory numbers honest.
    original = kkt_mod._chunked_vjp_rows
    if impl == 'loop':
        kkt_mod._chunked_vjp_rows = loop_chunked_vjp_rows
    elif impl != 'map':
        raise ValueError(f'unknown impl {impl!r}, expected map or loop')
    try:
        out, _, dofs, _ = quadcoil(**kwargs, jac_chunk_size=chunk)
        out, dofs = jax.block_until_ready((out, dofs))
    finally:
        kkt_mod._chunked_vjp_rows = original

    grads = out['phi_dofs']['grad']
    payload = {f'grad__{k}': np.asarray(v) for k, v in grads.items()}
    payload['value'] = np.asarray(out['phi_dofs']['value'])
    payload['phi'] = np.asarray(dofs['phi'])

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(os.path.join(OUTDIR, _npz_name(reg, constrained, chunk, impl)),
             **payload)

    n = payload['value'].size
    if chunk:
        nb, rem = divmod(n, chunk)
        regime = f'  batches={nb}x{chunk}+{rem}'
    else:
        regime = '  batches=unchunked'
    print(f'chunk={chunk} reg={reg:g} impl={impl} '
          f'{"constrained" if constrained else "unconstrained"}  '
          f'n_rows={n}{regime}  peak={peak_gib():.3f} GiB', flush=True)
    for k, v in grads.items():
        print(f'    {k:40s} max|g| = '
              f'{float(jnp.max(jnp.abs(v))):.6e}', flush=True)


# --------------------------------------------------------------------------
# Part 3: comparison
# --------------------------------------------------------------------------
def _diff(ref, cur):
    """Worst disagreement between two saved gradient dumps.

    Returns ``(rel_to_max, elem_rel, n_fail, n_tot)``. ``rel_to_max``
    normalizes by the leaf maximum and is the honest number; ``elem_rel``
    is per-entry and blows up on the noise floor, so it is reported but
    restricted to entries above 1e-8 of the leaf max.
    """
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
            worst_rel_to_max = max(worst_rel_to_max, float(np.max(d)) / amax)
            big = np.abs(a) > 1e-8 * amax
            if big.any():
                worst_elem = max(
                    worst_elem, float(np.max(d[big] / np.abs(a[big]))),
                )
        n_fail += int(np.sum(d > 1e-7 * np.abs(b)))
        n_tot += int(a.size)
    return worst_rel_to_max, worst_elem, n_fail, n_tot


def _load(reg, constrained, chunk, impl):
    path = os.path.join(OUTDIR, _npz_name(reg, constrained, chunk, impl))
    return np.load(path) if os.path.exists(path) else None


def compare(reg, constrained):
    """Compare every chunked run against the unchunked reference.

    Also compares ``map`` against ``loop`` at matching chunk size. Both
    are exact in real arithmetic, so a large entry in that column is the
    signal the old docstring warned about, and a small one retires it.
    """
    tag = _tag(reg, constrained)
    ref = _load(reg, constrained, None, 'map')
    if ref is None:
        print(f'[compare] {tag}: no unchunked reference, skipping')
        return

    print(f'\n[compare] {tag}: chunked vs unchunked, end to end')
    print('  rel_to_max  = max|a-b| / max|a|, per gradient leaf')
    print('  elem_rel    = worst |a-b|/|a| over entries above 1e-8 of the '
          'leaf max')
    print('  n_fail      = entries failing allclose(rtol=1e-7, atol=0)')
    print('  vs_loop     = same rel_to_max, map against loop at this chunk')
    print(f'  {"chunk":>6} {"impl":>6} {"phi bitwise":>12} {"rel_to_max":>12} '
          f'{"elem_rel":>12} {"n_fail":>8} {"n_tot":>8} {"peak GiB":>9} '
          f'{"vs_loop":>12}')

    for chunk in CHUNKS + [None]:
        for impl in (IMPLS if chunk is not None else ['map']):
            cur = _load(reg, constrained, chunk, impl)
            if cur is None:
                continue
            same_phi = bool(np.array_equal(cur['phi'], ref['phi']))
            rel_to_max, elem_rel, n_fail, n_tot = _diff(ref, cur)

            vs_loop = ''
            if impl == 'map' and chunk is not None:
                other = _load(reg, constrained, chunk, 'loop')
                if other is not None:
                    vs_loop = f'{_diff(other, cur)[0]:.3e}'

            peak = PEAKS.get((tag, chunk, impl), float('nan'))
            print(f'  {str(chunk):>6} {impl:>6} {str(same_phi):>12} '
                  f'{rel_to_max:>12.3e} {elem_rel:>12.3e} '
                  f'{n_fail:>8} {n_tot:>8} {peak:>9.3f} {vs_loop:>12}')


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
        run_intrace([argv[1]] if len(argv) > 1 else None)
        return
    if argv and argv[0] == 'single':
        chunk = None if argv[1] == 'none' else int(argv[1])
        reg = float(argv[2]) if len(argv) > 2 else 0.0
        impl = argv[3] if len(argv) > 3 else 'map'
        constrained = (argv[4] if len(argv) > 4 else 'con') == 'con'
        run_single(chunk, reg, impl, constrained)
        return

    print('=' * 90)
    print(f'jac_chunk_size check   mpol={MPOL} ntor={NTOR} '
          f'grid={NPHI}x{NTHETA} maxiter={MAXITER}')
    print(f'chunks={CHUNKS} regs={REGS} modes={MODES} impls={IMPLS}')
    print(f'quadcoil tests: {QUADCOIL_TESTS}')
    print(f'output:         {OUTDIR}')
    print('=' * 90)

    for mode in MODES:
        spawn(['intrace', mode])

    for mode in MODES:
        for reg in REGS:
            # Unchunked last: it is the memory-heaviest and the most likely
            # to die, and the chunked runs are the ones we would lose.
            jobs = [(str(c), impl) for c in CHUNKS for impl in IMPLS]
            jobs.append(('none', 'map'))
            for chunk, impl in jobs:
                out, rc = spawn(['single', chunk, repr(reg), impl, mode])
                if rc != 0:
                    continue
                key = (_tag(reg, mode == 'con'),
                       None if chunk == 'none' else int(chunk),
                       impl)
                for line in out.splitlines():
                    if 'peak=' in line:
                        PEAKS[key] = float(line.split('peak=')[1].split()[0])

    for mode in MODES:
        for reg in REGS:
            compare(reg, mode == 'con')

    print('\nReading the table: rel_to_max is the honest accuracy cost of '
          'chunking.\nelem_rel is dominated by noise-floor entries and will '
          'look alarming even\nwhen nothing is wrong. phi bitwise=True '
          'confirms the forward solve is\nuntouched, so any difference is '
          'in the adjoint alone.\n\nThe two numbers the lax.map change hangs '
          'on: vs_loop on the constrained\nrows at a chunk size with a '
          'nonzero remainder (must be at the same order\nas rel_to_max, not '
          'orders above it), and peak GiB for map against loop\nat the same '
          'chunk size (map should be markedly lower).')


if __name__ == '__main__':
    main()
