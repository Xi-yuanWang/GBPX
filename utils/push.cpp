// 如果FEATURE_TYPE 与python numpy的类型不符
//，则会导致隐式复制，导致修改传入的参数没有作用
#include <algorithm>
#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <iostream>
#include <omp.h>
namespace py = pybind11;

/* utils */

typedef double FEATURE_TYPE;
typedef int INDICE_TYPE;
const FEATURE_TYPE FLOAT_ERROR = 1e-9;

template <class T>
inline const T &RandomSelect(const std::vector<T> &arr)
{
    return arr[random() % arr.size()];
}

class EdgeList
{
public:
    INDICE_TYPE nFeature;
    INDICE_TYPE L, nV;
    std::vector<std::vector<INDICE_TYPE>> edge;

    EdgeList(INDICE_TYPE nV_, INDICE_TYPE L_)
        : nV(nV_), L(L_)
    {
        // std::cout << "nV=" << nV << " L=" << L << std::endl;
        edge.resize(nV);
    }

    void addEdge(py::array_t<INDICE_TYPE>& edgeFrom,
                 py::array_t<INDICE_TYPE>& edgeTo)
    {
        INDICE_TYPE e = edgeFrom.size();
        INDICE_TYPE a, b;
        for (INDICE_TYPE i = 0; i < e; ++i)
        {
            a = edgeFrom.at(i);
            b = edgeTo.at(i);
            edge[a].push_back(b);
        }
        // std::cout<<"addEdge"<<e<<" "<<edge[200000].size()<<std::endl;
    }

    // 插入无向边, 并进行残差更新
    // 假定不进行归一化
    bool push(py::array_t<INDICE_TYPE> dnode,
              py::array_t<FEATURE_TYPE> Drev,
              py::array_t<FEATURE_TYPE> Q_,
              py::array_t<FEATURE_TYPE> R_,
              py::array_t<FEATURE_TYPE> rmax)
    {
        auto Q = Q_.mutable_unchecked<3>();
        auto R = R_.mutable_unchecked<3>();
        std::vector<INDICE_TYPE> rootnode;
        std::vector<std::vector<INDICE_TYPE>> res;
        std::vector<std::vector<INDICE_TYPE>> tres;
        INDICE_TYPE nF = R.shape(2);
        for (size_t i = 0, s = dnode.size(); i < s; ++i)
        {
            rootnode.push_back(dnode.at(i));
        }
        tres.resize(nF, std::vector<INDICE_TYPE>());
        res.resize(nF, rootnode);
        for (int l = 0; l < L; ++l)
        {
            // #pragma omp parallel for
            for (INDICE_TYPE f = 0; f < nF; ++f)
            {
                tres[f]=rootnode;
                FEATURE_TYPE bound = rmax.at(f);
                FEATURE_TYPE sample=0;
                for (auto iter = res[f].begin(); iter != res[f].end(); ++iter)
                {
                    INDICE_TYPE node = *iter;
                    FEATURE_TYPE tmp = R(l, node, f);
                    sample+=tmp*tmp;
                    if (tmp > bound || tmp < -bound)
                    {
                        R(l, node, f) = 0;
                        Q(l, node, f) += tmp;
                        for (INDICE_TYPE next : edge[node])
                        {
                            tres[f].push_back(next);
                            R(l + 1, next, f) += Drev.at(next) * tmp;
                        }
                    }
                }
                std::swap(tres[f], res[f]);
            }
        }
        // #pragma omp parallel for
        for (INDICE_TYPE f = 0; f < nF; ++f)
        {
            for (INDICE_TYPE node : res[f])
            {
                Q(L, node, f) += R(L, node, f);
                R(L, node, f) = 0;
            }
        }
        return 1;
    };
    // ret shape:len(snode), nr, L+1
    // put indice in it
    void randomWalk(py::array_t<INDICE_TYPE>& snode,
                    py::array_t<FEATURE_TYPE>& ret_,
                    py::array_t<FEATURE_TYPE>& R_,
                    INDICE_TYPE nr)
    {
        if (nr == 0)
            return;
        auto ret = ret_.mutable_unchecked<3>();
        auto R = R_.unchecked<3>();
        INDICE_TYPE ns = snode.size(), nF = ret.shape(2);
        FEATURE_TYPE weight = 1.0 / nr;
        // #pragma omp parallel for
        for (INDICE_TYPE i = 0; i < ns; ++i)
        {
            INDICE_TYPE s = snode.at(i);
            if (edge[s].empty())
                continue;
            std::vector<INDICE_TYPE> curstep;
            curstep.resize(nr, snode.at(i));
            for (int l = 0; l < L; ++l)
                for (INDICE_TYPE f = 0; f < nF; ++f)
                    ret(i, f, l) += R(l, s, f);

            for (int t = 1; t <= L; ++t)
            {
                for (auto iter = curstep.begin(); iter != curstep.end(); ++iter)
                {
                    *iter = RandomSelect(edge[*iter]);
                }
                for (int l = t; l <= L; ++l)
                    for (INDICE_TYPE cur : curstep)
                        for (INDICE_TYPE f = 0; f < nF; ++f)
                            ret(i, f, l) += weight * R(l - t, cur, f);
            }
        }
    }
};
// ----------------
// Python interface
// ----------------

PYBIND11_MODULE(PPRPush, m)
{
    m.doc() = "A extension";
    py::class_<EdgeList>(m, "EdgeList")
        .def(py::init<INDICE_TYPE, INDICE_TYPE>(), "nV,nFeature,L")
        .def("addEdge", &EdgeList::addEdge)
        .def("push", &EdgeList::push)
        .def("randomWalk", &EdgeList::randomWalk);
}