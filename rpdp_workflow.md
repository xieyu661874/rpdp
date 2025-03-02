mermaid
```
graph LR
    classDef startend fill:#F5EBFF,stroke:#BE8FED,stroke-width:2px
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef decision fill:#FFF6CC,stroke:#FFBC52,stroke-width:2px

    A([开始]):::startend --> B(初始化阶段):::process
    B --> B1(服务器计算采样概率估计器 <br> $F(\varepsilon) = \alpha \cdot e^{\beta \cdot \varepsilon}$):::process
    B1 --> B2(服务器将估计器 $F$ 分发给所有客户端):::process
    B2 --> B3(客户端根据隐私预算 $\varepsilon_{i,j}$ 和 $F$ <br> 计算记录采样概率 $q_{i,j}=F(\varepsilon_{i,j})$):::process
    B3 --> C(客户端级采样阶段):::process
    C --> C1(中央服务器开始第 $t$ 轮训练):::process
    C1 --> C2(服务器以概率 $\lambda$ 进行 Poisson 采样 <br> 选择客户端子集 $\hat{\mathcal{C}}^t$):::process
    C2 --> C3(服务器将最新全局模型 $\mathbf{x}^t$ 发送给选中客户端):::process
    C3 --> D(记录级采样阶段):::process
    D --> D1(选中客户端开始本地训练，设置本地 SGD 步数 $\tau$):::process
    D1 --> D2(在每轮 $r$ ($1 \leq r \leq \tau$) 的训练中 <br> 客户端以概率 $q_{i,j}$ 进行 Poisson 采样 <br> 选择记录组成小批量 $S_r$):::process
    D2 --> D3(客户端基于小批量 $S_r$ 进行 DP - SGD 迭代更新本地模型):::process
    D3 --> D4(客户端完成 $\tau$ 步迭代后 <br> 将模型更新上传给中央服务器):::process
    D4 --> E(全局模型更新阶段):::process
    E --> E1(服务器聚合客户端上传的模型更新 <br> 更新全局模型 $\mathbf{x}^{t + 1}$):::process
    E1 --> F{训练是否结束?}:::decision
    F -- 否 --> C1
    F -- 是 --> G([结束]):::startend
    ```