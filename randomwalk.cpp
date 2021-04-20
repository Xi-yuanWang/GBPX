#include <cstdio>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <omp.h>
typedef double FEATURE_TYPE;
typedef int INDICE_TYPE;
typedef std::pair<INDICE_TYPE, INDICE_TYPE> COORD_TYPE;

namespace py = pybind11;

const FEATURE_TYPE FLOAT_ERROR = 1e-12;

/*
struct HashPair
{
    INDICE_TYPE operator()(const COORD_TYPE &coord_)
    {
        return std::hash<INDICE_TYPE>()((coord_.first << 16) + coord_.second);
    }
};
*/

template <class T>
inline const T RandomSelect(const std::vector<T> &arr)
{
    if (arr.empty())
        return -1;
    return arr[random() % arr.size()];
}

class SparseVec
{
public:
    std::unordered_map<INDICE_TYPE, FEATURE_TYPE> dat;
    SparseVec()
    {
        dat.clear();
    }
    FEATURE_TYPE operator[](INDICE_TYPE coord_) const
    {
        auto iter = dat.find(coord_);
        if (iter != dat.end())
        {
            return iter->second;
        }
        return 0;
    }
    void set(INDICE_TYPE coord_, FEATURE_TYPE f_)
    {
        dat.emplace(coord_, f_);
    }
    void add(INDICE_TYPE coord_, FEATURE_TYPE f_)
    {
        auto iter = dat.find(coord_);
        if (iter != dat.end())
        {
            iter->second += f_;
        }
        dat.emplace(coord_, f_);
    }
    void del(INDICE_TYPE coord_)
    {
        dat.erase(coord_);
    }
    FEATURE_TYPE innerProd(const SparseVec &s_) const
    {
        const SparseVec *a = this;
        const SparseVec *b = &s_;
        if (a->dat.size() > b->dat.size())
        {
            std::swap(a, b);
        }
        FEATURE_TYPE ret = 0;
        for (auto p : a->dat)
        {
            ret += (p.second) * (b->operator[](p.first));
        }
        return ret;
    }

    void add(const SparseVec &s_)
    {
        for (auto p : s_.dat)
        {
            add(p.first, p.second);
        }
    }
    void mul(FEATURE_TYPE f)
    {
        for (auto iter = dat.begin(); iter != dat.end(); ++iter)
        {
            iter->second *= f;
        }
    }

    void mul(INDICE_TYPE coord_, FEATURE_TYPE f)
    {
        auto iter = dat.find(coord_);
        if (iter != dat.end())
        {
            iter->second *= f;
        }
    }

    void fromNumpy(py::array_t<FEATURE_TYPE, py::array::c_style> array)
    {
        INDICE_TYPE s = array.size();
        dat.clear();
        FEATURE_TYPE tmp;
        for (INDICE_TYPE i = 0; i < s; ++i)
        {
            tmp = array.at(i);
            if (std::abs(tmp) > FLOAT_ERROR)
            {
                dat.emplace(i, tmp);
            }
        }
    }
    void maxmin_normalize()
    {
        if (dat.empty())
            return;
        FEATURE_TYPE minval = std::min(0.0, dat.begin()->second);
        FEATURE_TYPE maxval = std::max(0.0, minval);
        for (auto cur : dat)
        {
            minval = std::min(cur.second, minval);
            maxval = std::max(cur.second, maxval);
        }
        FEATURE_TYPE delta = maxval - minval;
        if (delta > FLOAT_ERROR)
        {
            FEATURE_TYPE factor = 1 / delta;
            for (auto iter = dat.begin(); iter != dat.end(); ++iter)
            {
                iter->second = (iter->second - minval) * factor;
            }
        }
    }

    void std_normalize(INDICE_TYPE dim)
    {
        if (dat.empty())
            return;
        FEATURE_TYPE aversq = 0;
        FEATURE_TYPE aver = 0;
        for (auto cur : dat)
        {
            aversq += cur.second * cur.second;
            aver += cur.second;
        }
        aver /= dim;
        aversq /= dim;
        FEATURE_TYPE delta = std::sqrt(aversq - aver * aver);
        if (delta > FLOAT_ERROR)
        {
            FEATURE_TYPE factor = 1 / delta;
            for (auto iter = dat.begin(); iter != dat.end(); ++iter)
            {
                iter->second = (iter->second - aver) * factor;
            }
        }
    }

    void L1_normalize()
    {
        FEATURE_TYPE L1 = 0;
        for (auto cur : dat)
        {
            L1 += std::abs(cur.second);
        }
        if (L1 > FLOAT_ERROR)
        {
            FEATURE_TYPE factor = 1 / L1;
            for (auto iter = dat.begin(); iter != dat.end(); ++iter)
            {
                iter->second *= factor;
            }
        }
    }

    void clean()
    {
        for (auto iter = dat.begin(); iter != dat.end();)
        {
            if (std::abs(iter->second) < FLOAT_ERROR)
            {
                iter = dat.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    void show()
    {
        std::cout << dat.size() << std::endl;
        for (auto cur : dat)
        {
            std::cout << cur.first << ':' << cur.second << ' ';
        }
        std::cout << std::endl;
    }
};

class Graph
{
public:
    INDICE_TYPE nV;
    INDICE_TYPE nFeature;
    FEATURE_TYPE r;
    int L;
    std::vector<std::vector<INDICE_TYPE>> edge;
    std::vector<FEATURE_TYPE> deg;
    std::vector<FEATURE_TYPE> degnegr; // deg^{-r}
    // 第一维是feature，第二维L
    std::vector<std::vector<SparseVec>> Q;
    std::vector<std::vector<SparseVec>> R;
    std::vector<SparseVec> P;
    std::vector<SparseVec> S;
    std::vector<SparseVec> X;

    Graph(INDICE_TYPE nV_, INDICE_TYPE nFeature_, int L_, FEATURE_TYPE r_) 
        : nV(nV_), nFeature(nFeature_), L(L_), r(r_)
    {
        edge.resize(nV);
        deg.resize(nV, 0);
        degnegr.resize(nV);
        Q.resize(nFeature);
        R.resize(nFeature);
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            Q[f].resize(L + 1);
            R[f].resize(L + 1);
        }

        P.resize(nFeature);
        X.resize(nFeature);
    }

    bool haveEdge(INDICE_TYPE a, INDICE_TYPE b)
    {
        if (a >= 0 && a < nV)
        {
            if (find(edge[a].begin(), edge[a].end(), b) != edge[a].end())
            {
                return true;
            }
        }
        return false;
    }

    // pyG格式，每条边在两个方向出现
    void fromNumpy(py::array_t<INDICE_TYPE, py::array::c_style> edgeFrom, py::array_t<INDICE_TYPE, py::array::c_style> edgeTo)
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
            deg[i] = edge[i].size();
        }
        for (INDICE_TYPE i = 0; i < nV; ++i)
        {
            degnegr[i] = pow(deg[i], -r);
        }
    }

    void getX(py::array_t<FEATURE_TYPE, py::array::f_style> feature)
    {
        auto F = feature.unchecked<2>();
        FEATURE_TYPE tmp;
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            X[f].dat.clear();
            for (INDICE_TYPE i = 0; i < nV; ++i)
            {
                tmp = degnegr[i] * F(i, f); // fortorn 的访问顺序也是先行后列
                if (std::abs(tmp) > FLOAT_ERROR)
                {
                    X[f].set(i, tmp);
                }
            }
            // std::cout<<X[f].dat.size()<<std::endl;
        }

    }

    void preCompute_nonorm(FEATURE_TYPE rmax)
    {
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            R[f][0] = X[f];
            preComputeOneCol(f, rmax);
        }
    }
    void preCompute_stdnorm(FEATURE_TYPE rmax)
    {
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            R[f][0] = X[f];
            R[f][0].std_normalize(nV);
            preComputeOneCol(f, rmax);
        }
    }

    void preCompute_minmaxnorm(FEATURE_TYPE rmax)
    {
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            R[f][0] = X[f];
            R[f][0].maxmin_normalize();
            preComputeOneCol(f, rmax);
        }
    }

    // push from R[f][0],with R[f][l]=0, if l>0
    // quevec[l]: possible element waiting for push in R[l]
    // quevec is not const
    void pushOneCol(INDICE_TYPE f_indice,
                std::vector<std::unordered_set<INDICE_TYPE>> &quevec_, const FEATURE_TYPE rmax)
    {
        // std::cout << "push col:" << f_indice << std::endl;
        std::vector<std::unordered_set<INDICE_TYPE>>& quevec = quevec_;
        quevec.resize(L + 1);

        for (int l = 0; l < L; ++l)
        {
            for (INDICE_TYPE cur : quevec[l])
            {
                FEATURE_TYPE f = R[f_indice][l][cur];
                R[f_indice][l].del(cur);
                Q[f_indice][l].add(cur, f);
                for (INDICE_TYPE prev : edge[cur])
                {
                    R[f_indice][l + 1].add(prev, f / deg[prev]);
                    if (std::abs(R[f_indice][l + 1][prev]) > rmax)
                    {
                        quevec[l + 1].emplace(prev);
                    }
                }
            }
            // quevec[l].clear(); // save memory
        }
        // R^L
        Q[f_indice][L].add(R[f_indice][L]);
        R[f_indice][L].dat.clear();
        /*
        if(f_indice==0)
        {
            //std::cout<<rmax<<std::endl;
            for(INDICE_TYPE l=0;l<=L;++l)
            {
                std::cout<<"l="<<l<<std::endl;
                std::cout<<quevec[l].size()<<std::endl;
                std::cout<<"R";
                R[f_indice][l].show();
                std::cout<<'Q';
                Q[f_indice][l].show();
            }
        }
        */
        // std::swap(q[L], r[L]);
        // r[L].dat.clear(); //?? GBP paper Algorithm1 13行有误??
    }

    // input one col of ColumnNormalized (D^r X)
    void preComputeOneCol(INDICE_TYPE f_indice, FEATURE_TYPE rmax) // 处理一维feature
    {
        // std::cout << "pre col:" << f_indice << std::endl;
        std::vector<std::unordered_set<INDICE_TYPE>> quevec;
        quevec.resize(1);
        for (auto p : R[f_indice][0].dat)
        {
            if (std::abs(p.second) > rmax)
            {
                quevec[0].emplace(p.first);
            }
        }
        pushOneCol(f_indice, quevec, rmax);
    }

    // 插入无向边, 并进行残差更新
    // 假定不进行归一化
    bool insertEdge(INDICE_TYPE a, INDICE_TYPE b)
    {
        if (haveEdge(a, b))
            return 0;
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            for (int i = 0; i < 2; ++i)
            {
                FEATURE_TYPE dega = deg[a];
                FEATURE_TYPE degnegra = pow(dega + 1, -r);

                R[f][0].add(a, (degnegra - degnegr[a]) * X[f][a]);
                for (INDICE_TYPE l = 1; l <= L; ++l)
                {
                    FEATURE_TYPE tmp = Q[f][l - 1][b] / (dega + 1);
                    for (INDICE_TYPE next : edge[a])
                    {
                        tmp -= Q[f][l - 1][next] / (dega + 1) / dega;
                    }
                    R[f][l].add(a, tmp);
                }

                std::swap(a, b);
            }
        }
        edge[a].push_back(b);
        edge[b].push_back(a);
        deg[a] += 1;
        deg[b] += 1;
        degnegr[a] = std::pow(deg[a], -r);
        degnegr[b] = std::pow(deg[b], -r);
        return 1;
    }

    // 插入无向边, 并进行残差更新
    // 假定不进行归一化
    bool insertEdgeAndPush(INDICE_TYPE a, INDICE_TYPE b, FEATURE_TYPE rmax)
    {
        if (haveEdge(a, b))
            return 0;
        #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            for (int i = 0; i < 2; ++i)
            {
                FEATURE_TYPE dega = deg[a];
                FEATURE_TYPE degnegra = pow(dega + 1, -r);

                R[f][0].add(a, (degnegra - degnegr[a]) * X[f][a]);
                for (INDICE_TYPE l = 1; l <= L; ++l)
                {
                    FEATURE_TYPE tmp = Q[f][l - 1][b] / (dega + 1);
                    for (INDICE_TYPE next : edge[a])
                    {
                        tmp -= Q[f][l - 1][next] / (dega + 1) / dega;
                    }
                    R[f][l].add(a, tmp);
                }

                std::swap(a, b);
            }
        }
        edge[a].push_back(b);
        edge[b].push_back(a);
        deg[a] += 1;
        deg[b] += 1;
        degnegr[a] = std::pow(deg[a], -r);
        degnegr[b] = std::pow(deg[b], -r);
        std::vector<std::unordered_set<INDICE_TYPE>> quevec;
        quevec.resize(1);
        quevec[0].emplace(a);
        quevec[0].emplace(b);
        quevec.resize(L + 1, quevec[0]);
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            pushOneCol(f, quevec, rmax);
        }

        return 1;
    }

    void randomWalk(INDICE_TYPE nr, INDICE_TYPE s)
    {
        FEATURE_TYPE weight = 1 / (FEATURE_TYPE)nr;
        S.clear();
        S.resize(L + 1);
        INDICE_TYPE ts;
        S[0].add(s, 1);
        for (INDICE_TYPE i = 0; i < nr; ++i)
        {
            ts = s;
            for (int step = 1; step <= L; ++step)
            {
                ts = RandomSelect(edge[ts]);
                if (ts < 0)
                    break;
                S[step].add(ts, weight);
            }
        }
    }
    // for a single feature of embedding of s
    // a single number
    // return w^[rr](l=0->L)[(wl*Ql)+ (t=0->l) S(l-t)R(t)] [s][f_indice]
    FEATURE_TYPE getP(std::vector<FEATURE_TYPE> &wl, INDICE_TYPE s, INDICE_TYPE f_indice)
    {
        std::vector<SparseVec> &q = Q[f_indice], &r = R[f_indice];
        FEATURE_TYPE ret = 0;
        for (int l = 0; l <= L; ++l)
        {
            ret += wl[l] * q[l][s];
        }
        for (int l = 0; l <= L; ++l)
        {
            for (int t = 0; t <= l; ++t)
            {
                ret += wl[l] * (S[l - t].innerProd(r[t]));
            }
        }
        return ret / degnegr[s];
    }
    std::vector<FEATURE_TYPE> getPVec(std::vector<FEATURE_TYPE> &wl, INDICE_TYPE s)
    {
        std::vector<FEATURE_TYPE> ret;
        ret.resize(nFeature);
        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            ret[f] = getP(wl, s, f);
        }
        return ret;
    }
};

PYBIND11_MODULE(GBP_utils, m)
{
    py::add_ostream_redirect(m);
    m.doc() = "A extension";
    py::class_<SparseVec>(m, "SparseVec")
        .def(py::init<>())
        .def("fromNumpy", &SparseVec::fromNumpy)
        .def("show", &SparseVec::show)
        .def("innerProd", &SparseVec::innerProd);

    py::class_<Graph>(m, "Graph")
        .def(py::init<INDICE_TYPE, INDICE_TYPE, int, FEATURE_TYPE>(), "nV,nFeature,L")
        .def("fromNumpy", &Graph::fromNumpy)
        .def("getX", &Graph::getX)
        .def("preComputeStd", &Graph::preCompute_stdnorm)
        .def("preComputeMM", &Graph::preCompute_minmaxnorm)
        .def("preComputeNone", &Graph::preCompute_nonorm)
        .def("randomWalk", &Graph::randomWalk)
        .def("getPVec", &Graph::getPVec);
}
