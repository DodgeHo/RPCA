//
// Created by Marvin Hao on 2016-04-08.
//

#include "RPCA.h"

InexactRPCASolver::InexactRPCASolver(const mat &data, double tol, double lambda, int maxIter, double muFactor,
                                     double rhoFactor) :
        RPCASolver(data),
        tol(tol),
        maxIter(maxIter),
        muFactor(muFactor),
        rhoFactor(rhoFactor) { this->lambda = lambda; }

InexactRPCASolver::InexactRPCASolver(const mat &data, double tol, double lambda, int maxIter) :
        InexactRPCASolver(data, tol, lambda, maxIter, 1.25, 1.5) { }

InexactRPCASolver::InexactRPCASolver(const mat &data, double tol, double lambda) :
        InexactRPCASolver(data, tol, lambda, 1000) { }

InexactRPCASolver::InexactRPCASolver(const mat &data, double tol) :
        InexactRPCASolver(data, tol, 1 / sqrt(data.n_rows)) { }

InexactRPCASolver::InexactRPCASolver(const mat &data) :
        InexactRPCASolver(data, 1e-7) { }


void InexactRPCASolver::initialize() {
    Y = data;
    normTwo = norm(Y, 2);
    normInf = (abs(Y)).max() / lambda;
    dualNorm = max(normTwo, normInf);
    Y = Y / dualNorm;
    dnorm = norm(data, "fro");
    totalSVD = 0;
    mu = muFactor / normTwo;
    muBar = mu * 1e7;
    rho = rhoFactor;

    Ahat = zeros(nRow, nCol);
    Ehat = zeros(nRow, nCol);
    iter = 0;
}

void InexactRPCASolver::iterate() {
    iter++;

    mat tmpT = data - Ahat + (1 / mu) * Y;
    Ehat = tmpT - clamp(tmpT, -lambda / mu, lambda / mu);

    // for svd
    vec s;
    mat U, V;
    mat D = data - Ehat + (1 / mu) * Y;
    svd_econ(U, s, V, D);

    vec newDiag = s.elem(find(s > (1 / mu)));
    newDiag = newDiag - 1 / mu;
    svp = newDiag.n_elem;

    if (svp == 0)
        Ahat = zeros(nRow, nCol);
    else
        Ahat = U.cols(0, (uword) svp - 1) * diagmat(newDiag.head((uword) svp)) * V.cols(0, (uword) svp - 1).t();

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



