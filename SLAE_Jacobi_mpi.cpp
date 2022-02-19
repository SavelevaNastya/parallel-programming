 //////////////										     ѕараллельный код решени€ —Ћј” методом якоби

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <ctime>

using namespace std;
#define N  4096
#define eps 1.0e-8

void vecPrint(double* X, int rank, int RowNum) {
	for (int i = rank* RowNum; i < (rank + 1) * RowNum; i++) {
		std::cout << "X[" << i  << "] = " << X[i] << endl;
	}
}

void matPrint(double** A,  int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << "A[" << i << "][" << j << "] = " << A[i][j] << endl;
		}
	}
}

void mult_matvec(double** A, double* b, double* Ab, int rank, int RowNum) {
	
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		for (int j = 0; j < N; j++) {
			if (i != j) 
				Ab[i] += A[i][j] * b[j];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void mult_matvec_(double** A, double* b, double* Ab, int rank, int RowNum) {

	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		for (int j = 0; j < N; j++) {
				Ab[i] += A[i][j] * b[j];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void subtract_vec(double* x, double* y, double* rez, int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		rez[i] = x[i] - y[i];
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void abs_subtract_vec(double* x, double* y, double* z,int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		z[i] = fabs(x[i] - y[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

double normaInf(double* x, int rank, int RowNum) {
	double norm = 0.0;
	
	/*norm = fabs(x[0]);
	for (int i = 0; i < N; i++) {
		if (fabs(x[i]) > norm) {
			norm = fabs(x[i]);
		}
	}*/

	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++)
	{
		MPI_Allreduce(&x[i], &norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return norm;
}

void update(double* Xnew, double* X, int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		X[i] = Xnew[i];
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void devide(double* X, double** A, int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		X[i] /= A[i][i];
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void assignment(double* X, double* F, int rank, int RowNum) {
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		X[i] = F[i];
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void initialization(double** A, double* X, double* F, int rank, int RowNum) {
	//srand(static_cast<unsigned int>(time(NULL)));

	for (int i = rank * RowNum; i < (rank + 1) * RowNum; ++i)
	{
		for (int j = 0; j < N; ++j) //for symmetry (int j = i; j < N; ++j)
		{
			if (i == j) {
				A[i][j] = rand();
			}
			else {
				A[i][j] = 0.0;
			}
			/*A[i][j] = rand() % 100;
			A[i][j] = A[j][i];*/
		}
		F[i] = rand();
		X[i] = i * 0.48;
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void freeMem(double** A, double* F, double* X) {
	
	for (int i = 0; i < N; i++) {
		free(A[i]);
	}
	free(A);
	free(X);
	free(F);
}
void Jacobi(double** A, double* F, double* X, int rank, int RowNum){
	/*double* TempX = new double[N];
	double* Ax = new double[N];
	double* x = new double[N];
	double* rezX = new double[N];*/
	double* TempX;
	double* Ax;
	double* x;
	double* rezX;
	TempX = (double*)malloc(N * sizeof(double));
	Ax = (double*)malloc(N * sizeof(double));
	rezX = (double*)malloc(N * sizeof(double));

	double norm = 0.0;
	int iter = 0;

	for (int i = 0; i < N; i++) {
		Ax[i] = 0.0;
		TempX[i] = 0.0;
		rezX[i] = 0.0;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	do {
		iter++;
		assignment(TempX, F, rank, RowNum);
		mult_matvec(A, X, Ax, rank, RowNum);
		subtract_vec(TempX, Ax, TempX, rank, RowNum);
		devide(TempX, A, rank, RowNum);

		abs_subtract_vec(X, TempX, rezX, rank, RowNum);
		norm = normaInf(rezX, rank, RowNum);
		if (rank == 0) { std::cout << "NORMA = " << norm << " " << "ITER " << iter << endl; }

		update(TempX, X, rank, RowNum);

		MPI_Barrier(MPI_COMM_WORLD);

	} while (norm > eps);
	if (rank == 0) { std::cout << "	Number of iteration = " << iter << endl; }
	free(TempX);
	free(Ax);
	free(rezX);

}

void checkAnswer(double** A, double* X, double* F, int rank, int RowNum) {
	//double* Ax = new double[N];
	double* Ax = (double*)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) {
		Ax[i] = 0.0;
	}
	mult_matvec_(A, X, Ax, rank, RowNum);
	int k = 0;
	for (int i = rank * RowNum; i < (rank + 1) * RowNum; i++) {
		if (fabs(Ax[i] - F[i]) < 0.001) {
			//cout << "Ax[" << i << "] = F[" << i << "] = " << Ax[i] << endl;
			//cout << "Ax[" << i << "] = " << Ax[i] << " F[" << i << "] = " << F[i] << endl;
			k++;
		}
		else {
			cout << "Ax[" << i << "] = " << Ax[i] << " F[" << i << "] = " << F[i] << endl;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) { std::cout << "	K = " << k << endl; }
}
int main(int argc, char* argv[]) {

	MPI_Init(NULL, NULL);
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	if (world_rank == 0) { std::cout << "	number of proccess = " << world_size << endl; }
	int RowNum = N / world_size;
	
	/*double* X = new double[N];
	double* F = new double[N];
	double** A = new double* [N];
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
	}*/
	double* X = (double*)malloc(N * sizeof(double));
	double* F = (double*)malloc(N * sizeof(double));
	double** A;
	A = (double**)malloc(N * sizeof(double*));
	for (int i = 0; i < N; i++)
	{
		A[i] = (double*)malloc(N * sizeof(double));
	}
	if (world_rank == 0) {
		std::cout << "\n	Matrix dimension = " << N << endl;
	}

	initialization(A, X, F, world_rank, RowNum);

	unsigned int start = clock();
	Jacobi(A, F, X, world_rank, RowNum);
	std::cout << endl;
	start = clock() - start;
	cout << "	Jacobi time =	" << ((float)start) / CLOCKS_PER_SEC << " second" << endl;

	checkAnswer(A, X, F, world_rank, RowNum);
	freeMem(A, F, X);

	MPI_Finalize();
	return 0;
}