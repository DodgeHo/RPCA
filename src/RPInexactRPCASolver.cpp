//
// Created by Marvin Hao on 2016-04-09.
//

#include "RPCA.h"

RPInexactRPCASolver::RPInexactRPCASolver(const mat &data, double tol, double lambda, int maxIter, double muFactor,
double rhoFactor, int rank) :
        InexactRPCASolver(data, tol, lambda, maxIter, muFactor, rhoFactor),
        rank(rank){}

RPInexactRPCASolver::RPInexactRPCASolver(const mat &data, double tol, double lambda, int maxIter, int rank) :
        InexactRPCASolver(data, tol, lambda, maxIter),
        rank(rank){}

RPInexactRPCASolver::RPInexactRPCASolver(const mat &data, double tol, double lambda, int rank) :
        InexactRPCASolver(data, tol, lambda),
        rank(rank){}

RPInexactRPCASolver::RPInexactRPCASolver(const mat &data, double tol, int rank) :
        InexactRPCASolver(data, tol),
        rank(rank){}

RPInexactRPCASolver::RPInexactRPCASolver(const mat &data, int rank) :
        InexactRPCASolver(data),
        rank(rank){}

void RPInexactRPCASolver::initialize() {
    InexactRPCASolver::initialize();
    k = 3 * rank;
    P = mat(k, nRow);
    P = P.randn();
    P = P / sqrt(k);
}

void RPInexactRPCASolver::iterate() {
    iter++;

    mat tmpT = data - Ahat + (1 / mu) * Y;
    Ehat = tmpT - clamp(tmpT, -lambda / mu, lambda / mu);

    mat tmpA = data - Ehat + (1 / mu) * Y;
    mat AProj = P * tmpA;
    mat AProjTran = AProj * AProj.t();


    // for svd, after random projection
    vec s;
    mat U, V;
    svd_econ(U, s, V, AProjTran);

    s = sqrt(s);
    vec newDiag = s.elem(find(s > (1 / mu)));
    // It doesn't shrink
    svp = newDiag.n_elem;

    if (svp == 0)
        Ahat = zeros(nRow, nCol);
    else{
        U = U.cols(0, (uword) svp - 1);
        V = AProj.t() * U * diagmat(1 / newDiag.head((uword) svp));
        Ahat = tmpA * V * V.t();
    }
    totalSVD++;

    mat Z = data - Ahat - Ehat;
    Y = Y + mu * Z;
    mu = (rho * mu < muBar) ? rho * mu : muBar;

    stopCriterion = norm(Z, "fro") / dnorm;
    cout << stopCriterion << endl;
    if (stopCriterion < tol)
        converged = true;
    if (!converged && iter > maxIter)
        converged = true;
}

/*
int main(int argc, char **argv) {
    mat lowrank, sparse;
    InexactRPCASolver::matrixGenerator(1000, 1000, 1000 * 0.05, 0.05, lowrank, sparse);
    //cout << lowrank << endl << sparse << endl;

    mat data = lowrank + sparse;
    RPInexactRPCASolver rpca(data, 1000 * 0.05);
    rpca.solve();

    //cout << rpca.getLowRank() << endl;
    //cout << rpca.getSparse() << endl;


}
 */