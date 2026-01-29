【incompleted_condense分支】：不完全列浓缩，就是只要相邻的con_thres个列有非零元素则保留这con_thres个列，目的是减少CopyInB时占用MTE2的数据搬运次数，缺点是相比于完全列浓缩TCBlock数肯定会大的多可以设置阈值con_thres(1,2,4,8,16)，重排和非重排都可以设置，与add_reorder相比，算子工程文件,host侧tiling文件,kernel侧均增加了一个Attr值con_thres

【日志】

2026/01/29
- AclnnInvocation文件夹中1.parse和parse_reorder中列浓缩逻辑修改，增加阈值 2.src文件夹下op_runner,operator_desc均做了属性和上传逻辑修改，main函数中smode增加“condense”和“reorder_condense”,对应的所有test.sh（就是那一堆.sh文件）均做了正确性修改和参数适配
- 算子部分的算子工程文件,host侧tiling文件,kernel侧均增加了一个Attr值con_thres，kernel侧新增con_thres和CUBE_BLOCK_K_TILE（就是当前阈值下一个TCblock中列块的个数，即CUBE_BLOCK_K/con_thres），copyInB部分由CUBE_BLOCK_K_TILE个ND2NZparam替换了DataCopyParam
- 正确性测试已通过，阈值1,2,4,8,16应该都可以取。就是效果不咋地，结构性矩阵还好些，非结构性矩阵效果很差，重排序可能会缓解阈值增大时的不利影响，但是功不抵过（我测试的那几个矩阵是这样）。推测原因1.是工程问题中的稀疏度很小，导致找到两个或多个相邻非零列很困难。2.目前的不完全列浓缩逻辑是在完全列浓缩基础上改的，导致相邻列块（也就是相邻的con_thres个列）选取不自由，即列块的起始列一定是con_thres的倍数，后续看安排修改。
- AclnnInvocation文件夹的.sh文件说明：主要区别在于1.MODE(分为"default","reorder","condense","reorder_condense");2.sample_dir(即预处理后生成bin文件的地址)3.测试时间记录（也就是那些加_time的.sh文件）


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

