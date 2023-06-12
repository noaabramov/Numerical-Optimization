import numpy as np

def minimize(f, x0, obj_tol, param_tol, max_iter, method):
    iteration = 0
    x = [float(xi) for xi in x0]
    success = False
    path = [x]
    compute_hessian = False if method == 'gradient_descent' else True
    objectives = [f(x, compute_hessian)]
    fx, gradient, hessian = f(x,compute_hessian)
    dir = 0 
    while iteration < max_iter:
        if method == 'gradient_descent':
            dir = -gradient

        elif method == 'newton':
            try:
                dir =  np.linalg.solve(hessian, -gradient)
            except:
                break

        elif method == 'bfgs':
            try:
                dir= np.linalg.solve(hessian, -gradient)
            except:
                break

        elif method == 'sr1':
            try:
                dir= np.linalg.solve(hessian, -gradient)
            except:
                break
        step_len = compute_step_length(f, x, fx, gradient, dir)
        x = np.add(x, dir * step_len)
        compute_hessian = True if method == 'newton' else False
        fx, gradient, hessian = f(x, compute_hessian)
        path.append(x)
        objectives.append(fx)

        # Check termination conditions
        if abs(fx - objectives[-2]) < obj_tol:
            success = True
            break

        if np.linalg.norm(x - path[-2]) < param_tol:
            success = True
            break

        iteration += 1

    final_location = x
    final_objective = objectives[-1]

    return final_location, final_objective, success, path



def compute_step_length(f, x, val, gradient, dir, alpha=0.01, beta=0.5):
    # wolfe conditions with backtracing
    step_len = 1.0
    x_update = np.add(x, step_len* dir) 
    curr_val, _, _ = f(x_update, False)

    while (curr_val - (val + alpha * step_len * gradient.dot(dir))).any():
        step_len *= beta
        x_new = np.add(x, step_len* dir) 
        curr_val, _, _ = f(x_new, False)

    return step_len


