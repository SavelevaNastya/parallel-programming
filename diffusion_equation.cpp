#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <math.h>
#include <omp.h>
#include <ctime>
#include <chrono>

constexpr auto N = 256;
constexpr auto pi = 3.14159;
constexpr auto sz = (N-2)*(N-2) + 1;

using namespace std;

int main() {
	double x[N], y[N], myError[sz];
	double u[N][N], v[N][N], u_new[N][N];
	int i, k, j = 0;
	double delta_x = 1.0 / (N - 1);
	double delta_t = delta_x * delta_x / 4.0;
	double u_0, diff;
	double D = 1.0;

	u[0][0] = 0.0;
	u[N - 1][N - 1] = 0;
	
	for (i = 0; i < N; i++) {

		x[i] = i * delta_x;
		y[i] = i * delta_x;
	}

	for (i = 0; i < N; i++) {
		for (k = 0; k < N; k++) {
			u_0 = sin(pi * x[i]) * sin(pi * y[k]);
			u[i][k] = u_0;
			v[i][k] = sin(pi * x[i]) * sin(pi * y[k]) * exp(-delta_t);
		}
	}
	//unsigned int start = clock();
	using picoseconds = std::chrono::duration<long long, std::pico>;
	auto t0 = std::chrono::steady_clock::now();
omp_set_num_threads(4);
#pragma omp parallel for
	for (i = 1; i < N - 1; i++) {
		//cout << /*"i = " << i << " k = " << k << " j = " << j << */" thread = " << omp_get_thread_num() << "\n";
		cout << "	threads = " << omp_get_num_threads() << endl;
		for (k = 1; k < N - 1; k++) {
			u_new[i][k] = u[i][k] + delta_t * D * (u[i + 1][k] - 4 * u[i][k] + u[i - 1][k] + u[i][k + 1] + u[i][k - 1]) / (2 * delta_x * delta_x);
			u[i][k] = u_new[i][k];
			diff = fabs(v[i][k] - u[i][k]);
			myError[j] = diff;
			j++;
		}
	}
	//start = clock() - start;
	auto t1 = std::chrono::steady_clock::now();
	auto d = picoseconds{ t1 - t0 } / N;
	//cout << "	TIME(parallel) =	" << ((float)start) / CLOCKS_PER_SEC << " second" << endl;
	
	std::cout << "	Time = " << d.count() << " ps\n";
	for (i = 0; i < sz; ++i)
	{
		if (myError[0] < myError[i])
			myError[0] = myError[i];
	}
	diff = myError[0];
	
	cout << "	max error(parallel) = " << diff << endl;
	return 0;
}