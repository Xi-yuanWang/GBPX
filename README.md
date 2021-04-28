# GBPX

```bash
cd utils
cmake .
make 
cd ..
```

建议修改CMakeFiles/make.flag。加入-O3选项优化。

已知的问题：Graph类应该传入np.double类型的array。



requirement

numpy

scipy

pybind11

pytorch