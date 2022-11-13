#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 11/03/2022
# ---------------------------------------------------------------------------
""" ode_solver.py

Solver for ODEs

Adapted from solve_ivp() of scipy package. Source:
https://github.com/scipy/scipy/blob/v1.8.0/scipy/integrate/_ivp/ivp.py

"""
# ---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import RK45, LSODA
from scipy.optimize import OptimizeResult

MESSAGES = {
    -1: "The solver was unsuccesfull in solving the equations",
    0: "The solver successfully reached the end of the integration interval.",
    1: "A termination event occurred.",
    2: "The solver successfully reached an equilibrium."
}

METHODS = {'RK45': RK45, 'LSODA': LSODA}


class OdeResult(OptimizeResult):
    """Adapted from OdeResult of scipy package. Source:
    https://github.com/scipy/scipy/blob/v1.8.0/scipy/integrate/_ivp/ivp.py
    """
    pass


def solve_ode(
    fun, t_span, y0, n_steps=1000, vectorized=False, args=None, save_partial=None,
    rtol=1e-3, atol=1e-6, method="RK45", stop_on_collapse=False, N_p=None, N_a=None,
    extinct_threshold=None, stop_on_equilibrium=False, equi_tol=1e-7, **options
):
    """Adapted from solve_ivp() of scipy package. Source:
    https://github.com/scipy/scipy/blob/v1.8.0/scipy/integrate/_ivp/ivp.py

    Uses explicit Runge-Kutta method of order 5(4) [1]_.
    The error is controlled assuming accuracy of the fourth-order
    method, but steps are taken using the fifth-order accurate
    formula (local extrapolation is done). A quartic interpolation
    polynomial is used for the dense output [2]_. Can be applied in
    the complex domain.

    Solve an initial value problem for a system of ODEs.
    This function numerically integrates a system of ordinary differential
    equations given an initial value::
        dy / dt = f(t, y)
        y(t0) = y0
    Here t is a 1-D independent variable (time), y(t) is an
    N-D vector-valued function (state), and an N-D
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.
    Some of the solvers support integration in the complex domain, but note
    that for stiff ODE solvers, the right-hand side must be
    complex-differentiable (satisfy Cauchy-Riemann equations [11]_).
    To solve a problem in the complex domain, pass y0 with a complex data type.
    Another option always available is to rewrite your problem for real and
    imaginary parts separately.
    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively, it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`. The choice between the two
        options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for stiff solvers).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be lower than the lowest value that can
        be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always lower
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.
    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).
    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    """
    if save_partial is not None:
        if not isinstance(save_partial, dict):
            raise TypeError(
                "save_partial should be a dict containing:\n,"
                "\t- tuple of range to save partial",
                "(inclusive, must be continuous domain)\n",
                "\t- step size between saving",
                "(if step size is equal to 0,",
                "than the data in partial range is not saved)"
            )

        save_inds = save_partial.get("ind")
        if save_inds[0] >= save_inds[1]:
            raise ValueError(
                'save_partial["ind"][0] should be smaller than save_partial["ind"][1]'
            )
        save_partial_period = save_partial.get("save_period")

    # check input
    # if stop_on_collapse is True, N_p and N_a should be given
    if stop_on_collapse:
        if N_p is None or N_a is None or extinct_threshold is None:
            raise ValueError(
                (
                    "If stop_on_collapse = True, provide N_p, N_a and ",
                    "extinct_threshold to solver"
                )
            )

    t_iter = 0  # keep track of number of iterations

    t0, tf = map(float, t_span)
    if t0 >= tf:
        raise ValueError("tf should be larger than t0")
    if not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps should be an integer and have a value larger than 1")
    t_eval = np.linspace(t0, tf, n_steps)
    t_eval_i = 0

    # Wrap the user's fun in lambdas to hide the
    # additional parameters.  Pass in the original fun as a keyword
    # argument to keep it in the scope of the lambda.
    if args is not None:
        fun = lambda t, x, fun=fun: fun(t, x, *args)

    if method in METHODS:
        method = METHODS[method]
    solver = method(fun, t0, y0, tf, rtol=rtol, atol=atol, **options)

    ts = []
    ys = []
    ts_partial = []
    ys_partial = []
    collapse_time = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        sol = None

        # The value in t_eval equal to t will be included
        t_eval_i_new = np.searchsorted(t_eval, t, side='right')
        t_eval_step = t_eval[t_eval_i:t_eval_i_new]

        if t_eval_step.size > 0:
            if sol is None:
                sol = solver.dense_output()
            ts.append(t_eval_step)

            ys_sol = sol(t_eval_step)
            if save_partial is not None:

                ys.append(
                    np.concatenate((ys_sol[:save_inds[0]], ys_sol[save_inds[1]+1:]))
                )
                # save to partial only each period as specified in save_partial_period
                if (
                    save_partial_period is not None and save_partial_period > 0 and
                    t_iter % save_partial_period == 0
                ):
                    ts_partial.append(t_eval_step)
                    ys_partial.append(ys_sol[save_inds[0]:save_inds[1]+1])
            else:
                ys.append(ys_sol)
            t_eval_i = t_eval_i_new

            # check if abundances pollinators below extinct_threshold:
            # set solver.status to finished
            if stop_on_collapse:
                if (ys[-1][N_p:N_p+N_a] < extinct_threshold).all():
                    status = 1
                    collapse_time = t
                    # print(ys[-1][N_p:N_p+N_a])
                # if (ys[-1][0:N_p+N_a] > 7).any():
                #     print("yes")
                #     status = 2

            # check if equilibrium has been reached by checking if all derivatives
            # given by fun are close to zero within a given tolerance
            if stop_on_equilibrium:
                f = fun(t_eval_step[-1], ys_sol[:, -1])
                if t_iter > 1:
                    if np.less(np.abs(f), equi_tol*np.ones(len(f))).all():
                        status = 2

        # always save last of partial
        if sol is None:
            sol = solver.dense_output()

        if (
            save_partial is not None and save_partial_period is not None and
            (isinstance(status, int) and status >= 0)
        ):
            ys_sol = sol(t_eval_step)

            if not ys_partial:
                ts_partial.append(t_eval_step)
                ys_partial.append(ys_sol[save_inds[0]:save_inds[1]+1])
            ts_partial.append(t_eval_step)
            ys_partial.append(ys_sol[save_inds[0]:save_inds[1]+1])
        t_iter += 1

    message = MESSAGES.get(status, message)

    if ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)
    if ts_partial:
        ts_partial = np.hstack(ts_partial)
        ys_partial = np.hstack(ys_partial)

    return OdeResult(
        t=ts, y=ys, t_partial=ts_partial, y_partial=ys_partial, status=status,
        message=message, success=status >= 0, collapse_time=collapse_time
    )
