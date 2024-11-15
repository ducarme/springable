import os
import numpy as np
import matplotlib.pyplot as plt
from src.springable.readwrite.interpreting import text_to_behavior
from src.springable.optimization import optimize, stochastically_optimize, globally_optimize
from src.springable.simulation import simulate_model, solve_model
from src.springable.readwrite.fileio import read_results, read_model

model_path = os.path.join('models', 'two_springs_optimization_bezier.csv')


# target_behavior = text_to_behavior(
#     'ZIGZAG2(u_i=[1.3268749685414143; 1.6860730608594303; 2.3710554694658783];'
#     'f_i=[7.536790226294599; 3.0761488268776898; 7.2369151742329585]; delta=0.01)', 1.0)
# target_behavior = text_to_behavior('ZIGZAG2(u_i=[3*0.5891266386215825; 3*1.2792464152925795; 3*1.8515408642392601]; f_i=[1.8141746494516173; 3.7633624878522856; 5.41267527419131]; delta=0.02)', 1.0)
# target_behavior = text_to_behavior(
#      'BEZIER2(u_i=[1.2624142256176771; -0.050496569024707005; 1.015542110385776]; f_i=[7.08697764820214; -1.4344717478828262; 2.0890601138414553])', 1.0)
# target_behavior = text_to_behavior('BEZIER2(u_i=[1/3; 2/3; 3/3]; f_i=[7.08697764820214; -1.4344717478828262; 2.0890601138414553])', 1.0)
# target_behavior = text_to_behavior('BEZIER2(u_i=[1.296078604967482; 2.0591378702297223; 2.895136624083207]; f_i=[8.686311259197558; 0.4147577398306268; 3.8383312508676948])', 1.0)
# target_behavior = text_to_behavior('BEZIER2(u_i=[19.895565654138686; 6.449643624494517; 13.081224586687952]; f_i=[7.061988060530336; 0.4897265028460369; 1.5142995973899769])', 1.0)
#target_behavior = text_to_behavior('BEZIER2(u_i=[1.6046687490073586; 0.7013412364542653; 2.249902686545283]; f_i=[2.5138831042621135; 0.11488268776898547; 2.4389143412467043])', 1.0)
target_behavior = text_to_behavior('BEZIER2(u_i=[13.30929487179487; -1.7147435897435912; 4.074519230769228]; f_i=[5.793161899387801; -1.0991276036418522; 1.3595253769710656])', 1.0)
a = target_behavior._a
b = target_behavior._b
x = np.linspace(0, 1, 100)

mdl = read_model(model_path)
res = solve_model(mdl, solver_settings='custom_solver_settings.toml')
u, f = res.get_equilibrium_path()
uu = res.get_displacements()
ff = res.get_forces()

fig, ax = plt.subplots()
ax.plot(np.arange(uu.shape[0]), uu[:, 2])
ax.plot(np.arange(uu.shape[0]), uu[:, 4])
plt.show()

fig, ax = plt.subplots()
ax.plot(a(x), b(x))
ax.plot(u, f)
plt.show()

fig, ax = plt.subplots()
ax.plot(a(x)-x, b(x))
plt.show()

optimal_design_parameters = optimize(model_path, a, b, 20)
print(optimal_design_parameters)


mdl = read_model(model_path, optimal_design_parameters)
res = solve_model(mdl, solver_settings='custom_solver_settings.toml')
u, f = res.get_equilibrium_path()
displacements = res.get_displacements()
mdl = read_model(model_path, optimal_design_parameters)
assmb = mdl.get_assembly()
dof_indices = assmb.get_free_dof_indices()
q0 = assmb.get_coordinates()
f1 = []
f2 = []
for i in range(x.shape[0]):
    q = q0.copy()
    q[dof_indices] += [x[i], a(x[i])]
    assmb.set_coordinates(q)
    internal_force_vector = assmb.compute_elastic_force_vector()[dof_indices]
    f1.append(internal_force_vector[0])
    f2.append(internal_force_vector[1])
assmb.set_coordinates(q0)


fig, ax = plt.subplots()
ax.plot(a(x), b(x))
ax.plot(u, f)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, b(x))
ax.plot(x, 0*x)
ax.plot(x, f2)
ax.plot(x, f1)
plt.show()

bhs = [el.get_behavior() for el in assmb.get_elements()]
fig, ax = plt.subplots()
ax.plot(x, bhs[0].gradient_energy(bhs[0].get_natural_measure()+x)[0])
ax.plot(x, bhs[1].gradient_energy(bhs[0].get_natural_measure()+x)[0])

plt.show()
