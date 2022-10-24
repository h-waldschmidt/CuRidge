#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cusolver_utils.h>

namespace ridge
{
    std::vector<double> solve(std::vector<double> const &rhs_matrix, std::vector<double> const &lhs_vector, int const m, int const n, double const lamda)
    {
        cusolverDnHandle_t cusolverH = NULL;
        cublasHandle_t cublasH = NULL;
        cudaStream_t stream{};

        // create lamda diagonal matrix
        std::vector<double> rhs_t_rhs_prod(n * n, 0);

        // create data for solution
        std::vector<double> x(n, 0);

        // device/gpu memory pointers
        double *d_rhs_matrix = nullptr;
        double *d_rhs_t_rhs_prod = nullptr;
        double *d_lamda_diagonal = nullptr;
        double *d_tau = nullptr;
        double *d_lhs_vector = nullptr;
        double *d_rhs_t_lhs_prod = nullptr;
        int *d_info = nullptr;
        double *d_work = nullptr;

        int lwork_geqrf = 0;
        int lwork_ormqr = 0;
        int lwork = 0;
        int info = 0;

        double const one = 1;
        int const nrhs = 1;
        // create handlers
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUBLAS_CHECK(cublasCreate(&cublasH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        // gpu memory allocation and copying of data
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rhs_matrix), sizeof(double) * rhs_matrix.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rhs_t_rhs_prod), sizeof(double) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_lhs_vector), sizeof(double) * lhs_vector.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rhs_t_lhs_prod), sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

        CUDA_CHECK(cudaMemcpyAsync(d_rhs_matrix, rhs_matrix.data(), sizeof(double) * rhs_matrix.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_lhs_vector, lhs_vector.data(), sizeof(double) * lhs_vector.size(), cudaMemcpyHostToDevice, stream));

        // compute intermediate computations before qr algorithm
        // variables needed for multiplications
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasOperation_t no_transpose = CUBLAS_OP_N;
        double const alpha = 1.0;
        double const beta = 0.0;
        // compute rhs_matrix^T * rhs_matrix + d_lamda_diagonal = d_lamda_diagonal
        int const lda = n;
        int const ldb = n;
        int const ldc = n;
        CUBLAS_CHECK(cublasDgemm(cublasH, transpose, no_transpose, m, n, n, &alpha, d_rhs_matrix, lda, d_rhs_matrix, ldb, &beta, d_rhs_t_rhs_prod, ldc));

        // to add lamda_diagonal, copy the data into cpu memory and add lamda on diagonal and then copy it back into gpu memory
        CUDA_CHECK(cudaMemcpyAsync(rhs_t_rhs_prod.data(), d_rhs_t_rhs_prod, sizeof(double) * rhs_t_rhs_prod.size(), cudaMemcpyDeviceToHost, stream));
        for (int i = 0; i < n; i++)
        {
            rhs_t_rhs_prod[i * (n + 1)] += lamda;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_rhs_t_rhs_prod, rhs_t_rhs_prod.data(), sizeof(double) * rhs_t_rhs_prod.size(), cudaMemcpyHostToDevice, stream));

        // compute rhs_matrix^T * lhs_vector = d_rhs_t_lhs_prod
        int const incx = 1;
        int const incy = 1;
        CUBLAS_CHECK(cublasDgemv(cublasH, transpose, m, n, &alpha, d_rhs_matrix, lda, d_lhs_vector, incx, &beta, d_rhs_t_lhs_prod, incy));

        // free unused memory
        CUDA_CHECK(cudaFree(d_rhs_matrix));
        CUDA_CHECK(cudaFree(d_lhs_vector));

        // query working space of geqrf and ormqr
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork_geqrf));

        CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m,
                                                   d_A, lda, d_tau, d_B, ldb, &lwork_ormqr));

        lwork = std::max(lwork_geqrf, lwork_ormqr);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

        // compute qr of d_lamda_diagonal and check if succeded
        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, n, n, d_rhs_t_rhs_prod, lda, d_tau, d_work, lwork, d_info));

        CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::printf("after geqrf: info = %d\n", info);
        if (0 > info)
        {
            std::printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }

        // compute Q^T * d_rhs_t_lhs_prod
        CUSOLVER_CHECK(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, nrhs, n, d_rhs_t_rhs_prod, lda,
                                        d_tau, d_rhs_t_lhs_prod, ldb, d_work, lwork, d_info));

        /* check if QR is good or not */
        CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::printf("after ormqr: info = %d\n", info);
        if (0 > info)
        {
            std::printf("%d-th parameter is wrong \n", -info);
            exit(1);
        }

        // solve QRx = d_rhs_t_lhs_prod
        CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, nrhs, &one, d_rhs_t_rhs_prod, lda, d_rhs_t_lhs_prod, ldb));

        CUDA_CHECK(cudaMemcpyAsync(x.data(), d_rhs_t_lhs_prod, sizeof(double) * x.size(), cudaMemcpyDeviceToHost,
                                   stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // free resources
        CUDA_CHECK(cudaFree(d_rhs_t_rhs_prod));
        CUDA_CHECK(cudaFree(d_tau));
        CUDA_CHECK(cudaFree(d_rhs_t_lhs_prod));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_work));

        CUBLAS_CHECK(cublasDestroy(cublasH));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaDeviceReset());
    }
}