
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

nelx, nely = 100, 100
Lx, Ly = 1, -1

mesh = RectangleMesh(nelx, nely, Lx, Ly)

V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
v = Function(V)
s = Function(V)
phi = TestFunction(V)

x, y = SpatialCoordinate(mesh)
c2 = Function(V).interpolate(conditional(y > Ly/2, Constant(2.5**2), Constant(3.5**2)))

# Mostra o modelo de velocidades

T = 1
nt = 1000
dt = T/nt
t = 0
step = 0

f0 = 10
x0 = [0.5, -0.02]

from utils import ricker, disturb_dof

# Mostra o sinal da fonte ao longo do tempo

# Inicia registro do tiro
coordinates = mesh.coordinates.dat.data
receivers = coordinates[coordinates[:,1]==0][:-1]
shot_record = u.at(receivers)
# Inicia registro da solução
output = File('output/saida.pvd')
output.write(u)

while t <= T:
    step += 1

    # Expressão (4a)
    u -= dt / 2 * v
    t += dt / 2

    # Prepara fonte
    disturb_dof(s, x0, h=ricker(f0, t))
    # Expressão (4b)
    v += assemble(dt * c2 * (inner(nabla_grad(u), nabla_grad(phi)) - s*phi)* dx) / assemble(phi*dx)

    # Expressão (4a) novamente
    u -= dt / 2 * v
    t += dt / 2

    # Registra sinal nos receptores
    shot_record = np.vstack((shot_record, u.at(receivers)))

    if step % 10 == 0:
        output.write(u, time=t)
        if step % 50 == 0:
            print("resolvendo para tempo t=", t)

# Mostra o registro do tiro
try:
  fig, axes = plt.subplots()
  colors = plt.imshow(shot_record, extent=[0, 1, 1, 0], cmap='gray')
  plt.title('Shot record')
  plt.xlabel('X position (km)')
  plt.ylabel('Time (s)')
  fig.colorbar(colors)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)
