#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

static __thread double *double_buffer;
static __thread int remaining_double_count = 0;
static __thread double **pointer_buffer;
static __thread int remaining_pointer_count = 0;

void init_double_buffer(double *buffer, int size)
{
    double_buffer = buffer;
    remaining_double_count = size;
}

void init_pointer_buffer(double **buffer, int size)
{
    pointer_buffer = buffer;
    remaining_pointer_count = size;
}

double *custom_double_alloc(int count)
{
    assert(remaining_double_count >= count);
    remaining_double_count -= count;
    double *out = double_buffer;
    double_buffer += count;
    return out;
}

double **custom_pointer_alloc(int count)
{
    assert(remaining_pointer_count >= count);
    remaining_pointer_count -= count;
    double **out = pointer_buffer;
    pointer_buffer += count;
    return out;
}

double **create_array(int m, int n)
{
    //    double* values = calloc(m*n, sizeof(double));
    double *values = custom_double_alloc(m * n);
    //    double **rows = malloc(m*sizeof(double*));
    double **rows = custom_pointer_alloc(m);
    for (int i = 0; i < m; ++i)
    {
        rows[i] = values + i * n;
    }
    return rows;
}

void initialize_array(double **loc, double inputs[], int m, int n)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            loc[i][j] = inputs[j * m + i];
}

double fro_norm(double **mat, int n)
{
    double out = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out += pow(mat[i][j], 2);
    return sqrt(out);
}

double **mat_mult(double **mat1, double **mat2, int m, int d, int n)
{
    double **out = create_array(m, n);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            out[i][j] = 0;

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < d; ++k)
                out[i][j] += mat1[i][k] * mat2[k][j];

    return out;
}

double **mat_sub(double **mat1, double **mat2, int m, int n)
{
    double **out = create_array(m, n);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            out[i][j] = mat1[i][j] - mat2[i][j];

    return out;
}

double **transpose(double **mat, int m, int n)
{
    double **out = create_array(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            out[j][i] = mat[i][j];

    return out;
}

double kl_dist(double x[11], double y[11])
{
    double d_buffer[64];
    double *p_buffer[64];

    init_double_buffer(d_buffer, sizeof(d_buffer));
    init_pointer_buffer(p_buffer, sizeof(p_buffer));

    double **diff = create_array(2, 1);
    initialize_array(diff, (double[2]){x[0] - y[0], x[1] - y[1]}, 2, 1);

    double **cov1 = create_array(2, 2);
    initialize_array(cov1, (double[4]){x[2], x[3], x[3], x[4]}, 2, 2);

    double **inv1 = create_array(2, 2);
    initialize_array(inv1, (double[4]){x[5], x[6], x[6], x[7]}, 2, 2);

    double **inv_sqrt1 = create_array(2, 2);
    initialize_array(inv_sqrt1, (double[4]){x[8], x[9], x[9], x[10]}, 2, 2);

    double **cov2 = create_array(2, 2);
    initialize_array(cov2, (double[4]){y[2], y[3], y[3], y[4]}, 2, 2);

    double **inv2 = create_array(2, 2);
    initialize_array(inv2, (double[4]){y[5], y[6], y[6], y[7]}, 2, 2);

    double **inv_sqrt2 = create_array(2, 2);
    initialize_array(inv_sqrt2, (double[4]){y[8], y[9], y[9], y[10]}, 2, 2);

    double **I = create_array(2, 2);
    initialize_array(I, (double[4]){1, 0, 0, 1}, 2, 2);

    double A = .5 * fro_norm(mat_sub(mat_mult(mat_mult(inv_sqrt2, cov1, 2, 2, 2), inv_sqrt2, 2, 2, 2), I, 2, 2), 2);
    double B = .5 * fro_norm(mat_sub(mat_mult(mat_mult(inv_sqrt1, cov2, 2, 2, 2), inv_sqrt1, 2, 2, 2), I, 2, 2), 2);

    double C = 1 / sqrt(2) * sqrt(mat_mult(mat_mult(transpose(diff, 2, 1), inv1, 1, 2, 2), diff, 1, 2, 1)[0][0]);
    double D = 1 / sqrt(2) * sqrt(mat_mult(mat_mult(transpose(diff, 2, 1), inv2, 1, 2, 2), diff, 1, 2, 1)[0][0]);

    return A + B + C + D;
}

int main()
{
    double x[11] = {
        2.,
        4.,
        0.92064502,
        0.24332643,
        0.25388742,
        1.45467181,
        -1.3941616,
        5.27492206,
        1.13397695,
        -0.41081394,
        2.25968006};

    double y[11] = {
        4.,
        6.,
        0.04480065,
        0.17672284,
        0.96730425,
        79.91013646,
        -14.59928098,
        3.70103455,
        8.82134985,
        -1.4470395,
        1.26771892};

    printf("%f", kl_dist(x, y));
    return 0;
}

/*
gcc -fPIC -shared -o C:\Users\anaki\Documents\GitHub\LINSCAN\linscan_c.so C:\Users\anaki\Documents\GitHub\LINSCAN\linscan_c.c
*/