def rk4_step(rhs, psi, t, dt, params):
    k1 = rhs(psi, t, params)
    k2 = rhs(psi + 0.5*dt*k1, t + 0.5*dt, params)
    k3 = rhs(psi + 0.5*dt*k2, t + 0.5*dt, params)
    k4 = rhs(psi + dt*k3, t + dt, params)
    return psi + dt*(k1 + 2*k2 + 2*k3 + k4)/6
