# 术语规范：不再使用「重建地板 / reconstruction floor」

## 官方/常用表述（优先）

| 场景 | 中文 | 英文（与 VQ/FSQ、全参考评价文献一致） |
|---|---|---|
| 核心对象 | **重建残差** / **编解码重建误差** | **reconstruction residual** / **encode–decode reconstruction error** |
| 定义式 | 真实帧 \(s'\) 与 \(D(Q(E(s')))\) 之差（或距离） | \(s'-D(Q(E(s')))\) 或 \(d(s',D(Q(E(s'))))\) |
| 现象 | 有损视觉分词导致真实帧**通常不能被精确复现** | raw frames are **generally not exactly reproducible** by the frozen tokenizer |
| 评测动作 | **编解码重建误差审计**（encode–decode 后测 LPIPS/MSE） | encode–decode reconstruction audit |
| 对 GRPO 的后果 | 目标侧残差与候选误差交互，**改变组内排序** | target residual interacts with candidate errors and **changes within-group ranking** |
| 错误做法 | 从奖励中**减去常数重建误差**（不改变排序） | subtracting a **constant reconstruction error** (does not fix ranking) |
| 正确做法 | 以 **\(D(Q(E(s')))\) 为训练期比较目标**（重建校准） | score against **tokenizer reconstruction** as training target |

## 废弃写法

- ~~重建地板~~、~~误差地板~~、~~floor~~（作术语）
- ~~reconstruction floor~~（正文尽量不用；若历史代码/路径含 floor 可保留路径名）
- ~~地板腐蚀~~ → 改为「重建残差干扰排序 / 改变组内相对排序」

## 一句话定义（可进 preliminaries）

冻结视觉编码器 \(E\)、量化器 \(Q\) 与解码器 \(D\) 将真实帧映射为  
\(\tilde s'=D(Q(E(s')))\)。  
\(d(s',\tilde s')\) 即为该接口下的 **编解码重建误差**；对应差 \(s'-\tilde s'\) 为 **重建残差**。  
它来自有损离散表征，与候选采样无关；但在组相对后训练中，残差与各候选误差的交互可改变 GRPO 使用的排序。
