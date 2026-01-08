【add_order更新日志】
2026/1/8
- parse_matrix_copy_reorder.py中新增reorder_double_row_csr_minhash函数
- AclNNInvocation/test_reorder.sh中新增MODE变量（default/reorder），作为main函数输入参数
- AclNNInvocation/inc新增re_handler类，用于处理aiCore返回的结果
- 其他：结果正确性测试通过，性能测试发现只有小部分矩阵性能提升，大部分性能下降。（推测：可能重排序破坏了矩阵原有的结构特征，导致功不抵过。还需要进一步的聚合手段进行优化，如列浓缩）

////////////////////

- 无 bias 的 matmul，基本上是按照 MatmulCustomMultiCore 的 sample 代码（带 bias 玩的版本）改的
- 源代码在 ./MatmulCustom 下
- 相应的改了 Acl 的代码和脚本

```
C = A * B
```

soc_version = 910B2

1. 编译
```
bash install.sh
```

如果 msopgen 命令报错类似如下
```
2025-09-29 21:42:30 (1031) - [ERROR] The path CooSpmmCustom.json should not be written by user group or others, which will cause security risks
```
请执行
```
chmod -R go-w .
```
将当前目录下的所有文件/目录的非用户写权限去除，即可正常运行。

2. 部署

执行生成的 Op 目录下的 build_out 下的 .run 文件
（第一步执行完最后会输出该 run 文件的路径）

3. 调用 & 测试
```
cd AclNNInvocation/
bash run.sh
```
