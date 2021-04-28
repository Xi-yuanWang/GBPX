# GBPX

```bash
pip install pybind11
ln -s $PYBIND11PATH ./utils/pybind11
cd utils
cmake .
make 
cd ..
```
需要在utils目录下放入pybind11文件夹的软连接。

建议修改CMakeFiles/make.flag。加入-O3选项优化。

已知的问题：Graph类应该传入np.double类型的array。

具体的使用方式见elliptic/anal.ipynb

requirement

numpy

scipy

pybind11

pytorch
