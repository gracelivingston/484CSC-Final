#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

float *random_Ns(size_t);
float gprod(float *, size_t);

int main(int argc, char **argv) {
        MPI_Init(NULL, NULL);

        srand48(time(NULL));
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        float *src_data;

        int n = -1;
        if (rank == 0) {
                if (argc != 2) {
                        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
                        return 1;
                }
        }

        n = atoi(argv[1]);

        if ((rank == 0) && (n < 1 || n > 8)) {
                fprintf(stderr, "<n> must be 1 - 8.\n");
                return 1;
        }

        src_data = random_Ns(n);

        int n_per_proc = n / size;
        int n_per_proc_r = n % size;

        float *proc_data = malloc(sizeof(float) * n_per_proc);

        float prod = 1.;
        MPI_Scatter(
                src_data,
                n_per_proc,
                MPI_FLOAT,
                proc_data,
                n_per_proc,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        float q = gprod(proc_data, n_per_proc);

        MPI_Reduce(&q, &prod, 1, MPI_FLOAT, MPI_PROD, 0, MPI_COMM_WORLD);

        double t = MPI_Wtime() - t0;

        if (rank == 0) {

                printf("Completed in %lfs\n", t);

                for(size_t i = 0; i < n_per_proc_r; i++) {
                        prod *= src_data[n_per_proc * size + i];
                }

                float gm = pow(prod, 1. / (float)n);

                printf("%f\n", gm);
        }

        MPI_Finalize();
        return 0;
}

float *random_Ns(size_t n) {
        float *Ns = malloc(sizeof(float) * n);

        for (size_t i = 0; i < n; i++) {
                Ns[i] = (0.01 + drand48() + drand48()) * 100.0;
        }

        return Ns;
}

float gprod(float *Ns, size_t n) {
        float p = 1;
        for (size_t i = 0; i < n; i++)
                p *= Ns[i];
        return p;
}
