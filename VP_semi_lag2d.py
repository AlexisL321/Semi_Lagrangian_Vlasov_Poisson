import numpy as np
from pylab import *
from numpy.linalg import solve
from numpy.linalg import qr
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.fft import fft, fftfreq, fft2, ifft2
import sympy as sp
from scipy.interpolate import RectBivariateSpline
from mpmath import invertlaplace
import mpmath as mp
#from scipy.interpolate import RegularGridInterpolator
from scipy.fft import ifft
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time


def make_mesh(Lx, Ly, LV, nx, ny, nvx, nvy):
	xs = linspace(0, L, nx, endpoint=False)
	hx = xs[1]-xs[0]

	ys = linspace(0, L, ny, endpoint=False)
	hy = ys[1]-ys[0]

	vxs = linspace(-LV, LV, nvx, endpoint=False)
	hvx = vxs[1]-vxs[0]

	vys = linspace(-LV, LV, nvy, endpoint=False)
	hvy = vys[1]-vys[0]

	X4, Y4, Vx4, Vy4 = np.meshgrid(xs, ys, vxs, vys, indexing='ij')
	return xs, ys, vxs, vys, hx, hy, hvx, hvy, X4, Y4, Vx4, Vy4

def semilag_xy(delta, f, xs, ys, vxs, vys, hx, hy):
	Xs = np.concatenate([xs, [xs[-1] + hx]])
	Ys = np.concatenate([ys, [ys[-1] + hy]])
	for k, vx in enumerate(vxs):
		for l, vy in enumerate(vys):
			F = np.vstack([np.hstack([f[:, :, k, l], f[:, 0:1, k, l]]),
				np.hstack([f[0:1, :, k, l], f[0:1, 0:1, k, l]])])
			interp = RectBivariateSpline(Xs, Ys, F, kx=3, ky=3)

			for i, x in enumerate(xs):
				for j, y in enumerate(ys):
					x_depart = (x - vx * delta) % (xs[-1] - xs[0]) + xs[0]
					y_depart = (y - vy * delta) % (ys[-1] - ys[0]) + ys[0]
					f[i, j, k, l] = interp(x_depart, y_depart)
	return f

def semilag_vxy(delta, f, Ex, Ey, vxs, vys, hvs, hvy):
	Vxs = np.concatenate([vxs, [vxs[-1] + hvx]])
	Vys = np.concatenate([vys, [vys[-1] + hvy]])
	for i, x in enumerate(Ex[:, 0]):
		for j, y in enumerate(Ey[0, :]):
			F = np.vstack([np.hstack([f[i, j, :, :], f[i, j, :, 0:1]]),
				np.hstack([f[i, j, 0:1, :], f[i, j, 0:1, 0:1]])])
			interp = RectBivariateSpline(Vxs, Vys, F, kx=3, ky=3)

			for k, vx in enumerate(vxs):
				for l, vy in enumerate(vys):
					vx_depart = (vx-Ex[i, j]*delta)%(vxs[-1]-vxs[0])+ vxs[0]
					vy_depart = (vy-Ey[i, j]*delta)%(vys[-1]-vys[0])+vys[0]
					f[i, j, k, l] = interp(vx_depart, vy_depart)
	return f

def compute_rho(f, f_eq):
	rho = hvx * hvy * (np.sum(f_eq, axis=(2, 3)) - np.sum(f, axis=(2,3)))
	return rho 

def compute_E_from_rho(rho):
	nx, ny = rho.shape
	kx = 2 * np.pi * fftfreq(nx, d=Lx/nx)
	ky = 2 * np.pi * fftfreq(ny, d=Ly/ny)
	kx, ky = np.meshgrid(kx, ky, indexing='ij')
	k_square = kx**2 + ky**2
	k_square[0,0] = 1.0
	rhohat = fft2(rho)
	Uhat = -rhohat / k_square
	Uhat[0,0] = 0.0

	Exhat = 1j * kx * Uhat
	Eyhat = 1j * ky * Uhat
	Ex = np.real(ifft2(Exhat))
	Ey = np.real(ifft2(Eyhat))

	return Ex, Ey

def compute_E(f, f_eq):
	return compute_E_from_rho(compute_rho(f, f_eq))

def electric_energy(Ex, Ey):
	energy_density = 0.5 * (Ex**2 + Ey**2)
	energy = hx * hy * np.sum(energy_density)
	return energy

def time_step_forward(dt, f, H, f_eq, Ex, Ey):
	f = semilag_xy(0.5*dt, f, xs, ys, vxs, vys, hx, hy)
	f_star = f.copy()
	Ex, Ey = compute_E(f, f_eq)
	Ex_total = Ex + H[0]
	Ey_total = Ey + H[1]
	f = semilag_vxy(dt, f, Ex, Ey, vxs, vys, hvx, hvy)
	f = semilag_xy(0.5*dt, f, xs, ys, vxs, vys, hx, hy)
	return f, Ex, Ey, f_star, Ex_total, Ey_total

def run_forward(f_iv, H, f_eq):
	f = f_iv.copy()
	t = 0.0
	ees = []
	fs = [f.copy()]
	Ex_stars = []
	Ey_stars = []
	f_stars = []
	Exs = []
	Eys = []

	for i in range(num_steps):
		Ex, Ey = compute_E(f.copy(), f_eq)
		Exs.append(Ex.copy())
		Eys.append(Ey.copy())

		ee = electric_energy(Ex, Ey)
		ees.append(ee)

		f,_,_, f_star, Ex_total_star, Ey_total_star = time_step_forward(dt,
			f, H, f_eq, Ex, Ey)

		fs.append(f.copy())
		f_stars.append(f_star.copy())
		Ex_stars.append(Ex_total_star.copy())
		Ey_stars.append(Ey_total_star.copy())

		t += dt
	
	return fs, ees, f_stars, Ex_stars, Ey_stars, Exs, Eys

nx = 16
ny = nx
nvx = 16
nvy = nvx
dt = 0.1
t_final = 40

L = 10 * np.pi
Ly = L
Lx = L
LV = 6.0
xs, ys, vxs, vys, hx, hy, hvx, hvy, X4, Y4, Vx4, Vy4 = make_mesh(Lx, Ly, LV, 
	nx, ny, nvx, nvy)
alpha = 0.5
mu1 = 2.4
mu2 = -2.4
epsilon = 0.001
k_0 = 0.2
f_eq_x = np.exp(-0.5 * ((Vx4 - mu1) ** 2))
f_eq_y = np.exp(-0.5 * ((Vy4) ** 2))
f_eq = alpha * f_eq_x * f_eq_y \
	+ (1-alpha) * np.exp(-0.5 * (Vx4 - mu2)**2) * f_eq_y

f_iv = (1.0 + epsilon * np.cos(k_0 * X4)) \
	* (1.0 + epsilon * np.cos(k_0 * Y4)) * f_eq

num_steps = int(t_final/dt)
t_values = np.linspace(0, t_final, num_steps)
dt = t_values[1] - t_values[0]

H = np.zeros((len(xs), len(ys)))

fs, ees, f_stars, Ex_stars, Ey_stars, Exs, Eys = run_forward(f_iv, H, f_eq)

plt.figure(figsize = (10,6))
plt.plot(t_values, ees)
plt.xlabel('Time')
plt.ylabel('Electric Energy')
plt.title('Electric Energy vs Time')
plt.grid(True)
plt.show()

def plot_vs(vs, x_or_y):
	plt.figure(figsize=(10,6))
	if x_or_y == 'x':
		plt.plot(vs, fs[num_steps - 1][1, 1, :, 1], label=f'f_eq at x={xs[1]:.2f} y={ys[1]:.2f}')
	else:
		plt.plot(vs, fs[num_steps - 1][1, 1, 1, :], label=f'f_eq at x={xs[1]:.2f} y={ys[1]:.2f}')
	plt.xlabel('v'+x_or_y)
	plt.ylabel('f')
	plt.title(f'f(v) at x = {xs[1]:.2f} y={ys[1]:.2f}')
	plt.legend()
	plt.grid(True)
	plt.show()

plot_vs(vxs, 'x')
plot_vs(vys, 'y')

def plot_result(x, y, v, density, zlabel):
	x, y = np.meshgrid(x, y)

	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111, projection='3d')

	surface = ax.plot_surface(x, y, v, facecolors=plt.cm.viridis(density),
		rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.9)
	
	mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
	mappable.set_array(density)
	fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Density")

	ax.set_xlabel('X Position')
	ax.set_ylabel('Y Position')
	ax.set_zlabel(zlabel)
	ax.set_title('distribution of velocity')

	plt.show()


plot_result(xs, ys, np.argmax(np.sum(fs[num_steps - 1], axis=3), axis=2),
np.sum(fs[num_steps - 1],axis=(2,3)), 'distribution on x-y 2d grid')
plot_result(xs, ys, np.argmax(np.sum(fs[num_steps - 1], axis=2), axis=2),
np.sum(fs[num_steps - 1],axis=(2,3)), 'distribution on x-y 2d grid')

#plot_result(xs, ys, f_eq[1, 1, 1, :], f_eq[1, 1, 1, :], 'y')


