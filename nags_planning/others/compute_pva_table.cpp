#include <iostream>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <chrono>

typedef struct pva_table pva_table;
struct pva_table {
    int dim_num;
    int *dim_size;
    uint64_t *dim_interval;
    uint64_t table_size;
    double *table;
    double rez;
    double *pva_limit;

    void construct_pva_table(int dim1_size, int dim2_size, int dim3_size, int dim4_size, double resolution) {
        this->dim_num = 4;
        this->dim_size = (int*)malloc(sizeof(int)*this->dim_num);
        this->dim_interval = (uint64_t*)malloc(sizeof(uint64_t)*this->dim_num);

        this->dim_size[0] = dim1_size;
        this->dim_size[1] = dim2_size;
        this->dim_size[2] = dim3_size;
        this->dim_size[3] = dim4_size;

        this->dim_interval[3] = 1;
        this->dim_interval[2] = dim_interval[3] * dim4_size;
        this->dim_interval[1] = dim_interval[2] * dim3_size;
        this->dim_interval[0] = dim_interval[1] * dim2_size;

        this->table_size = this->dim_interval[0] * dim1_size;
        this->table = (double*)malloc(sizeof(double)*this->table_size);

        this->rez = resolution;

        this->pva_limit = (double*)malloc(sizeof(double)*3);
        this->pva_limit[0] = this->rez*double(dim1_size/2);
        this->pva_limit[1] = this->rez*double(dim2_size/2);
        this->pva_limit[2] = this->rez*double(dim4_size/2);
    }

    void compute_idx_from_pva(double dlt_p, double v0, double vf, double a0,
                              int &idx1, int &idx2, int &idx3, int &idx4) {
        idx1 = round(dlt_p/this->rez) + this->dim_size[0]/2;
        idx2 = round(v0/this->rez) + this->dim_size[1]/2;
        idx3 = round(vf/this->rez) + this->dim_size[2]/2;
        idx4 = round(a0/this->rez) + this->dim_size[3]/2;
    }

    double query_pva_table(double dlt_p, double v0, double vf, double a0) {
        if (fabs(dlt_p) > this->pva_limit[0]) static_assert("Invalid input!", "");
        if (fabs(v0) > this->pva_limit[1]) static_assert("Invalid input!", "");
        if (fabs(vf) > this->pva_limit[1]) static_assert("Invalid input!", "");
        if (fabs(a0) > this->pva_limit[2]) static_assert("Invalid input!", "");

        int idx1, idx2, idx3, idx4;
        this->compute_idx_from_pva(dlt_p, v0, vf, a0, idx1, idx2, idx3, idx4);

        uint64_t idx = idx1*this->dim_interval[0] + idx2*this->dim_interval[1] +
                       idx3*this->dim_interval[2] + idx4*this->dim_interval[3];

        // std::cout << "idx: " << idx << std::endl;

        return this->table[idx];
    }

    void pva_table2csv(const std::string &str) {
        std::ofstream outfile;
        outfile.open(str, std::ios::out);

        for (int i = 0; i < 4; ++i) outfile << std::to_string(this->dim_size[i]) << ',';
        outfile << std::to_string(this->rez) << std::endl;

        for (uint64_t i = 0; i < this->table_size-1; ++i) outfile << std::to_string(this->table[i]) << ',';
        outfile << std::to_string(this->table[table_size-1]);

        outfile.close();
    }

    void csv2pva_table(const std::string &str) {
        int tmp_dim_size[4];
        double tmp_rez;

        std::ifstream infile(str, std::ios::in);
        std::string tmp_str;

        for (int i = 0; i < 4; ++i) {
            getline(infile, tmp_str, ',');
            tmp_dim_size[i] = std::stoi(tmp_str);
        }

        getline(infile, tmp_str);
        tmp_rez = std::stod(tmp_str);

        this->construct_pva_table(tmp_dim_size[0], tmp_dim_size[1],
                                  tmp_dim_size[2], tmp_dim_size[3], tmp_rez);

        for (uint64_t i = 0; i < this->table_size; ++i) {
            getline(infile, tmp_str, ',');
            this->table[i] = std::stod(tmp_str);
        }
    }

    void free_pva_table() {
        free(this->pva_limit);
        free(this->table);
        free(this->dim_interval);
        free(this->dim_size);
    }
};

void compute_pva_table(double p_limit, double v_limit, double a_limit, double rez,
                       double dlt_T, double dlt_t, double max_T, pva_table *table);
void csv2pva_table(const std::string &str, pva_table *table);

int main() {
    pva_table *test_table1;

    test_table1 = (pva_table*)malloc(sizeof(pva_table));

    auto start_t = std::chrono::steady_clock::now();

    compute_pva_table(3.2, 1.5, 2, 0.1, 0.2, 0.1, 15, test_table1);

    auto finish_t = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish_t - start_t);
    double compute_t = duration.count()/(double)1000;

    std::cout << "Compute time is: " << compute_t << std::endl;

    test_table1->pva_table2csv("p3-2_v1-5_a2_res0-1.csv");

    test_table1->free_pva_table();

    free(test_table1);

    return 0;
}

bool check_motion_primitives(double p0, double v0, double a0, double pf, double vf, double af,
                             double T, double dlt_t, double p_limit, double v_limit, double a_limit)
{
    int times = T / dlt_t;

    // % calculate optimal jerk controls by Mark W. Miller
    double dlt_a = af - a0;
    double dlt_v = vf - v0 - a0*T;
    double dlt_p = pf - p0 - v0*T - 0.5*a0*T*T;

    //%  if vf is not free
    double alpha = dlt_a*60/pow(T,3) - dlt_v*360/pow(T,4) + dlt_p*720/pow(T,5);
    double beta = -dlt_a*24/pow(T,2) + dlt_v*168/pow(T,3) - dlt_p*360/pow(T,4);
    double gamma = dlt_a*3/T - dlt_v*24/pow(T,2) + dlt_p*60/pow(T,3);

    double p, v, a, tt;

    for(int jj=0; jj<times; jj++)
    {
        tt = (jj + 1) * dlt_t;
        p = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0/2*pow(tt,2) + v0*tt + p0;
        v = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0*tt + v0;
        a = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0;

        if (fabs(p) > p_limit) return false;
        if (fabs(v) > v_limit) return false;
        if (fabs(a) > a_limit) return false;
    }

    return true;
}

// The unit of max_ite_t should be millisecond
void compute_pva_table(double p_limit, double v_limit, double a_limit, double rez,
                       double dlt_T, double dlt_t, double max_T, pva_table *table)
//@requires p_limit > 0 && v_limit > 0 && a_limit > 0 && rez > 0;
//@requires T != NULL;
{
    int dim1_size = 2*floor(p_limit/rez) + 1;
    int dim2_size = 2*floor(v_limit/rez) + 1;
    int dim3_size = dim2_size;
    int dim4_size = 2*floor(a_limit/rez) + 1;

    table->construct_pva_table(dim1_size, dim2_size, dim3_size, dim4_size, rez);

    double p0, pf, v0, vf, a0, af;
    p0 = af = 0.0;

    double T = dlt_T;

    uint64_t idx = 0;

    // Fill in the table
    for (int idx1 = 0; idx1 < dim1_size; ++idx1) {
        for (int idx2 = 0; idx2 < dim2_size; ++idx2) {
            for (int idx3 = 0; idx3 < dim3_size; ++idx3) {
                for (int idx4 = 0; idx4 < dim4_size; ++idx4, ++idx) {
                    pf = rez*(double)(idx1 - dim1_size/2);
                    v0 = rez*(double)(idx2 - dim2_size/2);
                    vf = rez*(double)(idx3 - dim3_size/2);
                    a0 = rez*(double)(idx4 - dim4_size/2);

                    T = dlt_T;

                    // Too slow to calculate idx
                    // idx = idx1 * dim1_intvl + idx2 * dim2_intvl + idx3 * dim4_size + idx4;

                    // auto ite_start_t = std::chrono::steady_clock::now();

                    while (true) {
                        if (check_motion_primitives(p0, v0, a0, pf, vf, af, T, dlt_t, p_limit, v_limit, a_limit)) {
                            table->table[idx] = T;
                            break;
                        }

                        if (T >= max_T) {
                            table->table[idx] = -1.0;
                            break;
                        }

                        T += dlt_T;
                    }
                }
            }
        }
    }
}
