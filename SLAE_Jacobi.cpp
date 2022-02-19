 
//////////////Последовательный код решения СЛАУ методом Якоби


#include <iostream>
#include <math.h>
#include <ctime>

using namespace std;
constexpr auto N = 1000;
constexpr auto eps = 0.1;

void myPrint(double* X) {
	for (int i = 0; i < N; i++) {
		cout << X[i] << endl;
	}
}

void mult_matvec(double** A, double* b, double* Ab) {
	for (int i = 0; i < N; i++) {
		Ab[i] = 0.0;
		for (int j = 0; j < N; j++) {
			if ( i != j)
				Ab[i] += A[i][j] * b[j];
		}
	}
}

void subtract_vec(double* x, double* y) {
	for (int i = 0; i < N; i++) {
		x[i] = x[i] - y[i];
	}
}

double normaInf(double* x_1, double* x_2) {
	double norm = 0;

	norm = fabs(x_1[0] - x_2[0]);
	for (int i = 0; i < N; i++) {
		if (fabs(x_1[i] - x_2[i]) > norm) {
			norm = fabs(x_1[i] - x_2[i]);
		}
	}
	return norm;
}

void update(double* Xnew, double* X) {
	for (int i = 0; i < N; i++) {
		X[i] = Xnew[i];
	}
}

void devide(double* X, double** A) {
	for (int i = 0; i < N; i++) {
		X[i] /= A[i][i];
	}
}

void assignment(double* X, double* F) {
	for (int i = 0; i < N; i++) {
		X[i] = F[i];
	}
}

void Jacobi(double** A, double* F, double* X)
{
	double* TempX = new double[N];
	double* Ax = new double[N];
	
	double norm;
	int iter = 0;

	for (int i = 0; i < N; i++) {
		Ax[i] = 0.0;
	}

	do {
		iter++;
		//cout << "Iteration = " << iter << endl;
		assignment(TempX, F);
		mult_matvec(A, X, Ax);

		subtract_vec(TempX, Ax);
		devide(TempX, A);

		norm = normaInf(X, TempX);

		update(TempX, X);
		//cout << iter << " " << "norma = " << norm << endl;
	} while (norm > eps);
	cout << "	Number of iteration = " << iter << endl;
	delete[] TempX;
	delete[] Ax;
}

void initialization(double** A, double* F, double* X) {
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j) {
				A[i][j] = 5000.0;
			}
			else {
				A[i][j] = 2.5;
			}
		}
		F[i] = i;
		X[i] = i * 0.48;
	}
}

void freeMem(double** A, double* F, double* X) {
	
	for (int i = 0; i < N; i++) {
		delete[] A[i];
	}
	delete[] A;
	delete[] X;
	delete[] F;
}

int main(int argc, char* argv[]) {

	unsigned int Allstart = clock();

	double* X = new double[N];
	double* F = new double[N];
	double** A = new double* [N];
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
	}
	cout << "\n	Matrix dimension = " << N << endl;
	initialization(A, F, X);
	unsigned int start = clock();
	Jacobi(A, F, X);
	cout << endl;
	start = clock() - start;
	cout << "	Jacobi time =	" << ((float)start) / CLOCKS_PER_SEC << " second" << endl;
	//myPrint(X);
	freeMem(A, F, X);
	
	Allstart = clock() - Allstart;
	cout << "	All time =	" << ((float)Allstart) / CLOCKS_PER_SEC << " second" << endl;
	return 0;
}