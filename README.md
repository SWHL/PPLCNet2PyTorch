#### 2021-11-11 update
- 添加基于pytorch的推理代码,并可以得到正确结果
- 结果示例：
    ```python
    # 运行pp_lcnet.py
    python pp_lcnet.py

    # 结果：
    # 0 tench, Tinca tinca: 0.9202795624732971

    # 运行pytorch_lcnet.py
    python pytorch_lcnet.py

    # 结果：
    # 0 tench, Tinca tinca: 0.9877434968948364
    ```

#### 2021-11-10 update
- 基于paddlepaddle框架可以正确推理，并得到正确结果

#### 2021-11-05 update
- 整个PP-LCNet跑通，可以正常跑

#### 参考资料
- [ngnquan/PP-LCNet](https://github.com/ngnquan/PP-LCNet)
- [frotms/PP-LCNet-Pytorch](https://github.com/frotms/PP-LCNet-Pytorch)