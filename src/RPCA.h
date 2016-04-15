//
// Created by Marvin Hao on 2016-04-08.
//

#ifndef RPCA_RPCA_H
#define RPCA_RPCA_H

#include <iostream>
#include <armadillo>
#include <math.h>


using namespace std;
using namespace arma;


class RPCASolver {

public:
    RPCASolver(const mat &data) :
            data(data),
            nRow(data.n_rows),
            nCol(data.n_cols),
            Ahat(mat()),
            Ehat(mat()) { };

    mat getOrigin() const { return (solved) ? data : NULL; }

    mat getLowRank() const { return (solved) ? Ahat : NULL; }

    mat getSparse() const { return (solved) ? Ehat : NULL; }

private:
    bool solved = false;

protected:
    const uword nRow, nCol;
    const mat data;
    mat Ahat, Ehat;
    bool converged = false;

    virtual void initialize() = 0;

    virtual void iterate() = 0;

public:
    void solve() {
        if (solved)
            return;
        initialize();
        while (!converged) {
            iterate();
        }
        solved = true;

    }

};

class InexactRPCASolver : public RPCASolver {

protected:
    double lambda;
    const double tol;
    const int maxIter;
    const double muFactor, rhoFactor;

    mat Y;
    double normTwo, normInf, dualNorm, dnorm;

    double mu, muBar, rho;
    int iter = 0;
    int totalSVD = 0;
    int svp = 0;
    double stopCriterion = 1;

    virtual void initialize();

    virtual void iterate();


public:
    // catch-all constructor
    InexactRPCASolver(const mat &data, double lambda, double tol, int maxIter, double muFactor, double rhoFactor);

    InexactRPCASolver(const mat &data, double lambda, double tol, int maxIter);

    InexactRPCASolver(const mat &data, double lambda, double tol);

    InexactRPCASolver(const mat &data, double lambda);

    InexactRPCASolver(const mat &data);


    static void matrixGenerator(int m, int n, int r, double p, mat &lowrank, mat &sparse) {
        mat left(m, r);
        mat right(r, n);
        while (1) {
            left.randn();
            right.randn();
            lowrank = left * right;
            if (arma::rank(lowrank) == r)
                break;
        }

        if (p == 0)
            sparse = mat(lowrank);
        else {
            sparse = 1000 * sprandn(m, n, p);
        }
    }

};

class RPInexactRPCASolver : public InexactRPCASolver{
protected:
    const int rank;

    // used during iteration
    mat P;
    int k;

    virtual void initialize();

    virtual void iterate();

public:
    RPInexactRPCASolver(const mat &data, double lambda, double tol, int maxIter, double muFactor, double rhoFactor,
                        int rank);

    RPInexactRPCASolver(const mat &data, double lambda, double tol, int maxIter, int rank);

    RPInexactRPCASolver(const mat &data, double lambda, double tol, int rank);

    RPInexactRPCASolver(const mat &data, double lambda, int rank);

    RPInexactRPCASolver(const mat &data, int rank);

};


#endif //RPCA_RPCA_H
