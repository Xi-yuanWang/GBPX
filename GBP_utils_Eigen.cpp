#include <cstdio>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <cmath>
#include <Eigen/SparseCore>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
namespace py = pybind11;

/* utils */

typedef double FEATURE_TYPE;
typedef int INDICE_TYPE;
typedef std::pair<INDICE_TYPE, INDICE_TYPE> COORD_TYPE;
typedef Eigen::SparseVector<FEATURE_TYPE> SparseVec;
typedef Eigen::SparseMatrix<FEATURE_TYPE> SparseMat;
typedef Eigen::Matrix<FEATURE_TYPE, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<FEATURE_TYPE, 1, Eigen::Dynamic> Vec;
const FEATURE_TYPE FLOAT_ERROR = 1e-9;

template <class T>
inline const T &RandomSelect(const std::vector<T> &arr)
{
    return arr[random() % arr.size()];
}

struct HashCoord
{
    std::hash<INDICE_TYPE> hasher;
    bool operator()(COORD_TYPE c_) const
    {
        return hasher((c_.first << 16) + c_.second);
    }
};

class Graph
{
public:
    INDICE_TYPE nV;
    INDICE_TYPE nFeature;
    int L;
    std::vector<std::vector<INDICE_TYPE>> edge;
    Vec deg;
    Vec degnegr;
    std::vector<Mat> Q;
    std::vector<Mat> R;
    Mat X;
    FEATURE_TYPE r;

    Graph(INDICE_TYPE nV_, INDICE_TYPE nFeature_, INDICE_TYPE L_, FEATURE_TYPE r_)
        : nV(nV_), nFeature(nFeature_), L(L_), r(r_)
    {
        edge.resize(nV);
        deg.resize(nV);
        degnegr.resize(nV);
        Q.resize(L + 1, Mat::Zero(nV,nFeature));
        R.resize(L + 1, Mat::Zero(nV,nFeature));
    }

    void fromNumpy(py::array_t<INDICE_TYPE, py::array::c_style> edgeFrom,
                   py::array_t<INDICE_TYPE, py::array::c_style> edgeTo)
    {
        INDICE_TYPE e = edgeFrom.size();
        INDICE_TYPE a, b;
        for (INDICE_TYPE i = 0; i < e; ++i)
        {
            a = edgeFrom.at(i);
            b = edgeTo.at(i);
            edge[a].push_back(b);
        }
        for (INDICE_TYPE i = 0; i < nV; ++i)
        {
            deg(i) = edge[i].size();
        }
        degnegr=deg;
        degnegr.array().pow(-r);
    }

    void getX(py::EigenDRef<const Mat> feature)
    {
        X = feature;
    }
    // input one col of ColumnNormalized (D^r X)
    void precompute(const FEATURE_TYPE rmax)
    {
        std::cout << nV << ' ' << nFeature<<' '<<rmax << std::endl;
        R[0]=degnegr.asDiagonal()*X;
        
        for (int l = 0; l < L; ++l)
        {
            Mat& r=R[l];
            Mat& rp=R[l+1];
            Mat& q=Q[l];
            for(int f=0;f<nFeature;++f)
            for(int i=0;i<nV;++i)
            {
                FEATURE_TYPE val=r(i,f);
                if(std::abs(val)>rmax)
                {
                    r(i,f)=0;
                    q(i,f)=val;
                    for(INDICE_TYPE j:edge[i])
                    {
                        rp(j,f)+=val/deg(j);
                    }
                }
            }
        }
        Q[L] += R[L];
        R.pop_back(); // R[L]=0
        /*
        std::unordered_set<COORD_TYPE, HashCoord> curque;
        std::unordered_set<COORD_TYPE, HashCoord> nextque;
        curque.clear();
        for (INDICE_TYPE i = 0; i < nV; ++i)
        {
            R[0].row(i) = X.row(i) * degnegr[i];
        }
        for (INDICE_TYPE i = 0; i < nV; ++i)
        for (INDICE_TYPE j = 0; j < nFeature; ++j)
        {
            if (std::abs(R[0].array()(i, j)) > rmax)
            {
                curque.emplace(i, j);
            }
        }
        // std::cout<<R[0]<<std::endl;

        std::cout << "R[0]" << curque.size() << std::endl;

        for (int l = 0; l < L; ++l)
        {
            nextque.clear();
            R[l + 1].array() = 0;
            for (COORD_TYPE cur : curque)
            {
                FEATURE_TYPE f = R[l].array()(cur.first, cur.second);
                R[l].array()(cur.first, cur.second) = 0;
                Q[l].array()(cur.first, cur.second) = f;
                for (INDICE_TYPE prev : edge[cur.first])
                {
                    R[l + 1].array()(prev, cur.second) += f / deg[prev];
                    if (std::abs(R[l + 1](prev, cur.second)) > rmax)
                    {
                        nextque.emplace(prev, cur.second);
                    }
                }
            }
            swap(curque, nextque);
            std::cout << "R[l]" << curque.size() << std::endl;
        }
        Q[L] += R[L];
        R.pop_back(); // R[L]=0
        */
    }

    void randomWalk(INDICE_TYPE nr,
        std::vector<SparseVec>& S, INDICE_TYPE s)
    {
        S.clear();
        S.resize(L + 1, SparseVec(nV));
        FEATURE_TYPE weight = 1 / (FEATURE_TYPE)nr;
        INDICE_TYPE ts;
        S[0].insert(s) = 1;
        for (INDICE_TYPE i = 0; i < nr; ++i)
        {
            ts = s;
            for (int step = 1; step <= L; ++step)
            {
                INDICE_TYPE next = RandomSelect(edge[ts]);
                ts = next;
                S[step].coeffRef(ts) += weight;
            }
        }
    }
    // 由getP 调用randomWalk函数
    Vec getP(const std::vector<FEATURE_TYPE> &wl, INDICE_TYPE nr, INDICE_TYPE s)
    {
        Vec P(nFeature);
        std::vector<SparseVec> S;
        randomWalk(nr,S,s);
        for (int l = 0; l <= L; ++l)
        {
            P += wl[l] * Q[l].row(s);
        }
        for (int l = 0; l < L; ++l)
        {
            for (int t = 0; t <= l; ++t)
            {
                P += wl[l] * (S[l - t].transpose() * R[t]);
            }
        }
        for (int t = 0; t < L; ++t)
        {
            P += wl[L] * (S[L - t].transpose() * R[t]);
        }
        return P;
    }
};

// ----------------
// Python interface
// ----------------

PYBIND11_MODULE(GBP_utils_Eigen, m)
{
    m.doc() = "A extension";
    py::class_<Graph>(m, "Graph")
        .def(py::init<INDICE_TYPE, INDICE_TYPE, INDICE_TYPE, FEATURE_TYPE>(), "nV,nFeature,L,r")
        .def("fromNumpy", &Graph::fromNumpy)
        .def("getX", &Graph::getX)
        .def("preCompute", &Graph::precompute)
        .def("randomWalk", &Graph::randomWalk)
        .def("getP", &Graph::getP);
}