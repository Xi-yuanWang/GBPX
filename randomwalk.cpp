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
typedef double FEATURE_TYPE;
typedef int INDICE_TYPE;
typedef std::pair<INDICE_TYPE, INDICE_TYPE> COORD_TYPE;

namespace py = pybind11;

struct HashPair
{
    INDICE_TYPE operator()(const COORD_TYPE &coord_)
    {
        return std::hash<INDICE_TYPE>()(coord_.first << 16 + coord_.second);
    }
};
template <class T>
inline const T &RandomSelect(const std::vector<T> &arr)
{
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
    void fromNumpy(py::array_t<FEATURE_TYPE, py::array::c_style> array)
    {
        INDICE_TYPE s = array.size();
        dat.clear();
        FEATURE_TYPE tmp;
        static const FEATURE_TYPE epsilon = 1e-8;
        for (INDICE_TYPE i = 0; i < s; ++i)
        {
            tmp = array.at(i);
            if (abs(tmp) > epsilon)
            {
                dat.emplace(i, tmp);
            }
        }
    }
    void normalize()
    {
        if(dat.empty())return;
        FEATURE_TYPE minval=std::min(0.0,dat.begin()->second);
        FEATURE_TYPE maxval=std::max(0.0,minval);
        for (auto cur : dat)
        {
            minval=std::min(cur.second,minval);
            maxval=std::max(cur.second,maxval);
        }
        if(maxval-minval<1e-7)return;
        FEATURE_TYPE delta=maxval-minval;
        for(auto iter=dat.begin();iter!=dat.end();++iter)
        {
            iter->second=(iter->second-minval)/delta;
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
    std::vector<FEATURE_TYPE> degnegr;// deg^{-r}
    std::vector<std::vector<SparseVec>> Q;
    std::vector<std::vector<SparseVec>> R;
    std::vector<SparseVec> P;
    std::vector<SparseVec> S;

    Graph(INDICE_TYPE nV_, INDICE_TYPE nFeature_, int L_,FEATURE_TYPE r_) : 
        nV(nV_), nFeature(nFeature_), L(L_), r(r_)
    {
        edge.resize(nV);
        deg.resize(nV, 0);
        degnegr.resize(nV);
        Q.resize(nFeature);
        R.resize(nFeature);
        P.resize(nFeature);
    }

    void addEdge(const std::vector<COORD_TYPE> &edgeList)
    {
        for (auto cur : edgeList)
        {
            edge[cur.first].push_back(cur.second);
            edge[cur.second].push_back(cur.first);
        }
        for (INDICE_TYPE i = 0; i < nV; ++i)
        {
            deg[i] = edge[i].size();
        }
    }

    void addEdge(INDICE_TYPE a, INDICE_TYPE b)
    {
        edge[a].push_back(b);
        edge[b].push_back(a);
        deg[a] += 1;
        deg[b] += 1;
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
    }
    void preCompute(py::array_t<FEATURE_TYPE, py::array::f_style> feature,FEATURE_TYPE rmax)
    {
        SparseVec tmpV;
        auto F = feature.unchecked<2>();
        FEATURE_TYPE tmp;

        for(INDICE_TYPE i=0;i<nV;++i)
        {
            degnegr[i]=pow(deg[i],-r);
        }

        for (INDICE_TYPE f = 0; f < nFeature; ++f)
        {
            tmpV.dat.clear();
            for (INDICE_TYPE i = 0; i < nV; ++i)
            {
                tmp = degnegr[i]*F(i,f);// fortorn 的访问顺序也是先行后列
                // col normalize??
                // 取epsilon 会莫名少加数
                if (tmp!=0)
                {
                    tmpV.set(i, tmp);
                }
            }
            tmpV.normalize();
            // tmpV.show();
            preComputeOneCol(tmpV,f,rmax);
        }
    }
    // input one col of ColumnNormalized (D^r X)
    void preComputeOneCol(SparseVec& X, INDICE_TYPE f_indice, FEATURE_TYPE rmax) // 处理一维feature
    {
        std::vector<SparseVec> &q = Q[f_indice];
        std::vector<SparseVec> &r = R[f_indice];
        q.resize(L + 1);
        r.resize(L + 1);
        std::unordered_set<INDICE_TYPE> curque;
        std::unordered_set<INDICE_TYPE> nextque;

        r[0] = X;

        for (std::pair<INDICE_TYPE, FEATURE_TYPE> p : X.dat)
        {
            if (abs(p.second) > rmax)
            {
                curque.emplace(p.first);
            }
        }

        for (int l = 0; l < L; ++l)
        {
            nextque.clear();
            for (INDICE_TYPE cur : curque)
            {
                FEATURE_TYPE f = r[l][cur];
                r[l].del(cur);
                q[l].add(cur, f);
                for (INDICE_TYPE prev : edge[cur])
                {
                    r[l + 1].add(prev, f / deg[prev]);
                    if (abs(r[l + 1][prev]) > rmax)
                    {
                        nextque.emplace(prev);
                    }
                }
            }
            swap(curque, nextque);
        }
        std::swap(q[L],r[L]);
        r[L].dat.clear(); //?? GBP paper Algorithm1 13行有误??
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
        return ret /degnegr[s];
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
        .def("innerProd",&SparseVec::innerProd);

    py::class_<Graph>(m, "Graph")
        .def(py::init<INDICE_TYPE, INDICE_TYPE, int,FEATURE_TYPE>(), "nV,nFeature,L")
        .def("addEdge", static_cast<void (Graph::*)(INDICE_TYPE, INDICE_TYPE)>(&Graph::addEdge))
        .def("fromNumpy", &Graph::fromNumpy)
        .def("preCompute", &Graph::preCompute)
        .def("randomWalk", &Graph::randomWalk)
        .def("getPVec", &Graph::getPVec);
}