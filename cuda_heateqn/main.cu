// HPC133 heat diffusion equation solution
// written in thrust by matthew thoms
// next steps: CAN WE PLOT THIS IN C++?? PLEASE?? LMAO??

#include <thrust/universal_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "params.h"

// kernel for initializing position values from indices
struct posn_init {

	double x1;
	double x2;
	int maxdim;

	posn_init(double& _x1, double& _x2, int& _maxdim) {
		x1 = _x1;
		x2 = _x2;
		maxdim = _maxdim;
	}

	__host__ __device__
		double operator()(int pos) {
		return x1 + ((pos - 1) * (x2 - x1)) / maxdim;
	}
};

// kernel for initializing heat densities
struct dens_init {

	double x1;
	double x2;

	dens_init(double& _x1, double& _x2) {
		x1 = _x1;
		x2 = _x2;
	}

	__host__ __device__
		double operator()(thrust::tuple<double, double> ij) {
		double xdim = thrust::get<0>(ij);
		double ydim = thrust::get<1>(ij);
		
		double a = 1 - fabs(1 - 4 * fabs((xdim - (x1 + x2) / 2) / (x2 - x1)));
		double b = 1 - fabs(1 - 4 * fabs((ydim - (x1 + x2) / 2) / (x2 - x1)));

		return a * b;
	}
};

// kernel for heat equation timestep
struct heat_evolve {

	double D;
	double x1;
	double x2;
	double dx;
	double dt;

	heat_evolve(double& _D, double& _x1, double& _x2, double& _dx, double& _dt) {
		D = _D;
		x1 = _x1;
		x2 = _x2;
		dx = _dx;
		dt = _dt;
	}

	__host__ __device__
		double operator()(thrust::tuple<double, double, double, double, double, double, double> info) {
		double x = thrust::get<0>(info);
		double y = thrust::get<1>(info);

		if (x == x1 || x == x2 || y == x1 || y == x2) {
			return 0.0;
		}

		double top = thrust::get<2>(info);
		double left = thrust::get<3>(info);
		double center = thrust::get<4>(info);
		double right = thrust::get<5>(info);
		double bottom = thrust::get<6>(info);

		double laplacian = top + left + right + bottom - 4 * center;

		return center + (D / (dx * dx) * dt * laplacian);

	}
};

// export heat density as a .csv file
// i tried to get so many plotting libraries to function and none of them did. ouch oof owie
// plot in python for better results
void export_dens(thrust::universal_vector<double>& x,
	thrust::universal_vector<double>& y,
	thrust::universal_vector<double>& d,
	int timestep,
	int entries) {

	std::string filepath = "densdata" + std::to_string(timestep) + ".csv";
	std::ofstream densdata(filepath);

	for (int idx = 0; idx < entries; idx++) {
		densdata << x[idx] << "," << y[idx] << "," << d[idx] << "\n";
	}
}

int main() {

	// compute derived parameters
	int nrows = ((x2 - x1) / dx);
	int ncols = nrows;
	int npnts_x = ncols + 2;
	int npnts_y = nrows + 2;
	int tpnts = npnts_x * npnts_y;

	double dt = 0.25 * dx * dx / D;
	int nstep = runtime / dt;
	int nper = outtime / dt;
	if (nper == 0) {
		nper = 1;
	}

	// creating "grid" of x and y indices (cheese it with 1D vectors - numpy meshgrid approach)
	thrust::universal_vector<double> x(tpnts);
	thrust::universal_vector<double> y(tpnts);

	thrust::universal_vector<int> tr_modulo(tpnts, npnts_x);
	thrust::universal_vector<int> tr_divide(tpnts, npnts_y);
	
	thrust::sequence(x.begin(), x.end());
	thrust::sequence(y.begin(), y.end());

	thrust::transform(x.begin(), x.end(), tr_modulo.begin(), x.begin(), thrust::modulus<int>());
	thrust::transform(y.begin(), y.end(), tr_divide.begin(), y.begin(), thrust::divides<int>());

	// initialize position coordinates
	thrust::transform(x.begin(), x.end(), x.begin(), posn_init(x1, x2, ncols));
	thrust::transform(y.begin(), y.end(), y.begin(), posn_init(x1, x2, nrows));

	// initializing heat density at t = 0
	thrust::universal_vector<double> dens(tpnts, 0.0);
	thrust::universal_vector<double> densnext(tpnts, 0.0);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
		dens.begin(),
		dens_init(x1, x2));

	// create permutation iterators for laplacian (maybe find a way to simplify this... ew!)
	std::vector<int> x_vec(tpnts);
	thrust::sequence(x_vec.begin(), x_vec.end());

	std::vector<int> xmrow_vec = x_vec;
	std::rotate(xmrow_vec.begin(), xmrow_vec.begin() + npnts_x, xmrow_vec.end());
	thrust::universal_vector<int> xmrow = xmrow_vec;

	std::vector<int> xm1_vec = x_vec;
	std::rotate(xm1_vec.begin(), xm1_vec.begin() + 1, xm1_vec.end());
	thrust::universal_vector<int> xm1 = xm1_vec;

	std::vector<int> xp1_vec = x_vec;
	std::rotate(xp1_vec.rbegin(), xp1_vec.rbegin() + 1, xp1_vec.rend());
	thrust::universal_vector<int> xp1 = xp1_vec;

	std::vector<int> xprow_vec = x_vec;
	std::rotate(xprow_vec.rbegin(), xprow_vec.rbegin() + npnts_x, xprow_vec.rend());
	thrust::universal_vector<int> xprow = xprow_vec;

	auto xmrow_iter_begin = thrust::make_permutation_iterator(dens.begin(), xmrow.begin());
	auto xm1_iter_begin = thrust::make_permutation_iterator(dens.begin(), xm1.begin());
	auto xp1_iter_begin = thrust::make_permutation_iterator(dens.begin(), xp1.begin());
	auto xprow_iter_begin = thrust::make_permutation_iterator(dens.begin(), xprow.begin());

	auto xmrow_iter_end = thrust::make_permutation_iterator(dens.begin(), xmrow.end());
	auto xm1_iter_end = thrust::make_permutation_iterator(dens.begin(), xm1.end());
	auto xp1_iter_end = thrust::make_permutation_iterator(dens.begin(), xp1.end());
	auto xprow_iter_end = thrust::make_permutation_iterator(dens.begin(), xprow.end());

	// do the simulation!
	double simtime = 0.0 * dt;

	export_dens(x, y, dens, 0, tpnts);

	std::cout << "recorded initial state!\n";

	for (int s = 0; s < nstep; s++) {
		
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), xmrow_iter_begin, xm1_iter_begin, dens.begin(), xp1_iter_begin, xprow_iter_begin)),
			thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), xmrow_iter_end, xm1_iter_end, dens.end(), xp1_iter_end, xprow_iter_end)),
			densnext.begin(),
			heat_evolve(D, x1, x2, dx, dt));

		thrust::swap(dens, densnext);

		if ((s + 1) % nper == 0) {

			export_dens(x, y, dens, s, tpnts);

			std::cout << "recorded at timestep " << s << "!\n";
		}

		simtime += dt;
	}

}