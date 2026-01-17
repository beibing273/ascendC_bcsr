【add_order更新日志】


2026/1/8
- parse_matrix_copy_reorder.py中新增reorder_double_row_csr_minhash函数
- AclNNInvocation/test_reorder.sh中新增MODE变量（default/reorder），作为main函数输入参数
- AclNNInvocation/inc新增re_handler类，用于处理aiCore返回的结果
- 其他：结果正确性测试通过，性能测试发现只有小部分矩阵性能提升，大部分性能下降。（推测：可能重排序破坏了矩阵原有的结构特征，导致功不抵过。还需要进一步的聚合手段进行优化，如列浓缩）

2026/1/13
- parse_matrix_copy_reorder.py中新增reorder_double_DTC函数
- 其他：结果正确性测试通过，DTC双重排的性能测试大部分情况下性能优于之前的双重排，但依然不如默认情况下的效果。

2026/1/16
- 新增列凝聚操作
- 其他：列凝聚正确性人工检验通过，进行列凝聚效果测试，证明了行重排序对列凝聚的有效性。结果如下（nozero_rate指的是非零元素在总存储元素的比值）：

<img width="1707" height="961" alt="Figure_0 2" src="https://github.com/user-attachments/assets/f8fc2633-5d44-444d-872b-231b63ac5f2a" />
<img width="284" height="182" alt="屏幕截图 2026-01-16 162609" src="https://github.com/user-attachments/assets/bb288c79-95bb-43bd-bed7-c874653e7317" />

#######################################

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
---------------------------------------------
【add_reorder执行测试】：将run.sh文件中的
```
bash test.sh
```
改为
```
bash test_reorder.sh
```
即可
