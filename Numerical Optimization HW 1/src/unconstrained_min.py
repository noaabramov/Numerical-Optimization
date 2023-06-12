import numpy as np

def minimize(f, x0, obj_tol, param_tol, max_iter, method):
    """
    Minimize a function using various optimization methods.

    Parameters:
    - f: The objective function to be minimized.
    - x0: The initial guess for the optimization variables.
    - obj_tol: The tolerance for the change in the objective function value.
    - param_tol: The tolerance for the change in the optimization variables.
    - max_iter: The maximum number of iterations.
    - method: The optimization method to use ('Gradient Descent', 'Newton', 'BFGS', or 'SR1').

    Returns:
    - final_location: The optimized values of the variables.
    - final_objective: The final value of the objective function.
    - success: True if the optimization was successful, False otherwise.
    - path: A dictionary containing the optimization path (values of variables at each iteration).
    """
    iteration = 0
    x = np.array([float(xi) for xi in x0])
    success = False
    path = dict(path=[], values=[])
    path['path'] = [x]
    compute_hessian = False if method == 'Gradient Descent' else True
    objectives = [f(x, compute_hessian)]
    fx, gradient, hessian = f(x, compute_hessian)
  
    path['values']=[fx]
    dir = 0 
    while iteration < max_iter:
        if method == 'Gradient Descent':
            dir = -gradient

        elif method == 'Newton':
            try:
                dir =  np.linalg.solve(hessian, -gradient)
            except:
                break

        elif method == 'BFGS':
            try:
                dir= np.linalg.solve(hessian, -gradient)
            except:
                break

        elif method == 'SR1':
            try:
                dir= np.linalg.solve(hessian, -gradient)
            except:
                break
        step_len = compute_step_length(f, x, fx, gradient, dir)
        prev_x = x.copy()
        prev_fx = fx.copy()
        x = np.add(x, dir * step_len)
        compute_hessian = True if method == 'Newton' else False
        fx, gradient, hessian = f(x, compute_hessian)
        path['path'].append(x.copy())
        path['values'].append(fx)

        if abs(fx - prev_fx) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not gradient.any():
            success = True
            break

        iteration += 1

    final_location = x
    final_objective = objectives[-1]

    return final_location, final_objective, success, path


def compute_step_length(f, x, val, gradient, dir, alpha=0.1, beta=0.05):
    """
    Compute the step length for the line search in optimization.

    Parameters:
    - f: The objective function.
    - x: The current values of the optimization variables.
    - val: The current value of the objective function.
    - gradient: The gradient of the objective function at x.
    - dir: The search direction.
    - alpha: The sufficient decrease parameter.
    - beta: The step length reduction parameter.

    Returns:
    - step_len: The computed step length.
    """
    step_len = 1.0
    curr_val, _, _ = f(x + dir * step_len, False)

    while curr_val > val + alpha * step_len * gradient.dot(dir):
        step_len *= beta
        curr_val, _, _ = f(x + step_len * dir, False)

    return step_len
