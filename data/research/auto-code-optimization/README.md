# 自动代码优化强相关论文精选

> 数据范围：2026-07-01 至 2026-09-04；生成时间：2026-09-04T05:42:43+00:00。
> 以 SemOpt 的“程序分析/策略知识 + LLM 自动改写 + 正确性与真实性能验证”为研究锚点，只保留强相关论文。

共筛选 **81** 篇（去重候选论文 5361 篇，模型评估 2151 篇）。

## 快速索引

1. [LLM4LLM: Bridging Kernel Benchmarks and Real Deployment via Closed-Loop Agentic Optimization](#1) — 0.97
2. [CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language](#2) — 0.96
3. [KernelArc: A Multi-Agent Framework for GPU Kernel Optimization](#3) — 0.96
4. [RepoOMP: Repository-Aware Hotspot OpenMP Parallelization via Dependency-Aware Context Reduction](#4) — 0.96
5. [AgenticCANN: Automated Ascend C Operator Generation via Knowledge-Augmented Agentic Evolution](#5) — 0.96
6. [PerfAgent: Profiler-Guided Iterative Refinement for Repository-Level Code Optimization](#6) — 0.96
7. [Beyond the Need for Speed: Energy-Aware Code Generation via Simulation-Guided Reinforcement Learning](#7) — 0.96
8. [Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory](#8) — 0.95
9. [FABRICA: Agentic CUDA-to-CSL Translation and Optimization for Wafer-Scale Systems](#9) — 0.95
10. [Accelerated Genetic Programming Hyper-Heuristics for Simulation-Based Scheduling via Agentic AI](#10) — 0.95
11. [AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification](#11) — 0.95
12. [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](#12) — 0.95
13. [Effect of Abstractions and Prompting Strategies on LLM-Guided High-Performance Optimizations](#13) — 0.95
14. [HLSmith: An Expert-Guided Agentic Framework for C/C++-to-HLS Translation](#14) — 0.95
15. [SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System](#15) — 0.95
16. [Kernel Forge: An Agent Harness for LLM-based Generation and Optimization of CUDA Kernels](#16) — 0.95
17. [RLPF: Reinforcement Learning from Performance Feedback for Code Generation](#17) — 0.95
18. [Multi-level Code Optimization via Mixture of Prompts](#18) — 0.95
19. [VPR-Evolve: Multi-Agent-Driven Algorithm Evolution for FPGA Place and Route](#19) — 0.95
20. [Multi-Source and Cross-Scenario Strategy-Guided Code Optimization](#20) — 0.95
21. [Technical Report: AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell](#21) — 0.95
22. [From Custom-Fit to Portable: Bridging the Gap Between Synthesized and Engineered GPU Query Execution](#22) — 0.95
23. [Copper: Unifying Correctness and Performance Specification in Code Generation](#23) — 0.95
24. [Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation](#24) — 0.95
25. [HIERA: Workload-Aware Planning Across Implementation Spaces for GPU Kernel Optimization](#25) — 0.94
26. [Compiler-Grounded Hierarchical Diagnosis for LLM-Based Triton Kernel Optimization](#26) — 0.94
27. [MKEvolve: A Modular Multi-Agent Framework for Kernel Code Generation](#27) — 0.94
28. [FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications](#28) — 0.94
29. [PERFOPT-Bench: Evaluating Coding Agents on Software Performance Optimization](#29) — 0.94
30. [DSEffi-Bench: Demystifying Large Language Models' Capability in Efficient Data Science Code Generation](#30) — 0.93
31. [DataKernelBench: Can LLMs Optimize Database Queries on GPUs?](#31) — 0.93
32. [RAGas: Retrieval-Augmented Gas Optimization for Smart Contracts with Continuous Knowledge Integration](#32) — 0.93
33. [T-LLM Compiler: Trusted LLM-based Code Optimization and Verification Framework](#33) — 0.93
34. [PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX](#34) — 0.93
35. [A Barrier-Free Synchronization Algorithm for Multi-Engine AI Accelerators](#35) — 0.93
36. [Enhancing SLMs for Sustainable Code Optimization in Radio-Astronomy](#36) — 0.93
37. [Harness Engineering for LLM-Driven GPU Kernel Generation](#37) — 0.93
38. [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](#38) — 0.93
39. [Nova: An End-to-End MLIR Compiler for Deep Learning](#39) — 0.93
40. [Every Kernel Is a Join: Automatic Multi-GPU Parallelism for AI Computations in Einsummable](#40) — 0.92
41. [Enhancing the Power of Polyhedral-Based Optimizations with Coordinate-Based Hill Climbing](#41) — 0.92
42. [Semantics-Guided Automatic Tensorization for Multiobjective Evolutionary Algorithms: A Multi-Agent Framework](#42) — 0.92
43. [HyperCut: Fast Inter-Layer Scheduling via Directed Hypergraph and Early Filtering](#43) — 0.92
44. [ComFuse: Fusing Complex Memory-Intensive Subgraphs with Compute-Intensive Kernels For Modern GPU Architectures](#44) — 0.92
45. [Rethinking Agentic Kernel Generation for Emerging Accelerators](#45) — 0.92
46. [CONQuER: Hardware-Aware Mixed-Precision Quantisation with Online-Calibrated Surrogates](#46) — 0.92
47. [JAXBench: Benchmarking Autonomous TPU Kernel Optimization](#47) — 0.92
48. [Ciphertext- and Polynomial-Level Optimization for Fully Homomorphic Encryption](#48) — 0.92
49. [Breaking Database Lock-in: Agentic Regeneration of High Performance Storage Readers for Database Bypass](#49) — 0.92
50. [QuTuner: Feature- and Learning-Guided Optimization Pass Tuning for Quantum Compilers](#50) — 0.92
51. [Understanding Agent-Based Patching of Compiler Missed Optimizations](#51) — 0.92
52. [Tensor Seeks Layout: Formalizing Layout Selection for ML Compilers](#52) — 0.91
53. [Portable to Efficient: Auto-Tuning Hardware-Agnostic GPU Kernels in Julia](#53) — 0.91
54. [Validation-Centric AI-Assisted GPU Porting of a 250,000+ Line Legacy Weather Simulation Code](#54) — 0.91
55. [WarmTuner: Program-Specific Warm Starts for Compiler Autotuning via Offline-to-Online Reinforcement Learning](#55) — 0.91
56. [CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits](#56) — 0.91
57. [Can Coding Agents Implement Missed Compiler Optimizations? Evaluating LLM Agents on LLVM Peephole Optimizations](#57) — 0.91
58. [CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling](#58) — 0.90
59. [Hierarchical Shared Memory-Aware Optimization for TRSM on GPU Platforms](#59) — 0.90
60. [XRFix: Exploring Performance Bug Repair of Extended Reality Applications with Large Language Models](#60) — 0.90
61. [Can Large Language Models Recover Semantic Optimization Opportunities That Compilers Miss?](#61) — 0.90
62. [KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation](#62) — 0.90
63. [Cross-Model Cross-Language AI Coding Agent Performance: Accuracy and Speed of Parallel CLRS Algorithms](#63) — 0.90
64. [Demonstrating GenDB: Instance-Optimized and Customized Query Processing Code Generation via LLM Agents](#64) — 0.90
65. [Pattern-Guided Design Space Exploration for FPGA Accelerator Design](#65) — 0.90
66. [Rethinking Code Performance Benchmarks for LLMs](#66) — 0.90
67. [Optimus: A Generic Operator-Level PyTorch Model Transformation Framework](#67) — 0.90
68. [Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?](#68) — 0.90
69. [Integrating a Python Dynamical core into ICON](#69) — 0.89
70. [Memory Allocation for Constant-Bounded Programs](#70) — 0.88
71. [RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks](#71) — 0.88
72. [GPU Offload in Rust: Portable, Safe, and Fast](#72) — 0.88
73. [An eightfold equivalence-preserving speedup of the JUNO OMILREC vertex and energy reconstruction](#73) — 0.88
74. [Compiling Bioinformatics Recurrences](#74) — 0.88
75. [EffiHolmes: Differential Profiling-Guided Repository Level Time Inefficiency Fix Localization](#75) — 0.87
76. [Gaming Without an Attacker: Benchmark Fingerprinting in LLM-Driven Search Under Selection Pressure](#76) — 0.86
77. [TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters](#77) — 0.86
78. [Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap in Modern Domain-Specific Languages](#78) — 0.86
79. [The Unseen Delta: Characterizing the Compiler Optimization Landscape via Top-Down Differential Analysis](#79) — 0.85
80. [What Do AI Agents Actually Change? An Empirical Taxonomy of Mutation Patterns in Performance-Improving Pull Requests](#80) — 0.85
81. [EvoMem: Memory-Augmented Evolution for Code Optimization](#81) — 0.82

---

<a id="1"></a>
## 1. [LLM4LLM: Bridging Kernel Benchmarks and Real Deployment via Closed-Loop Agentic Optimization](https://arxiv.org/abs/2608.21836)

- **相关度**：0.97
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-26, 2026-08-22
- **arXiv ID**：2608.21836
- **作者**：Hui Zeng, Pengfei Yang, Yanxin Chen, Fusong Ju, Xinran Wei
- **入选理由**：提出部署感知的闭环优化框架，抽取推理工作负载的优化任务并用经验引导的agent搜索、接受包含kernel补丁的优化，在A100/H100上带来几何平均3.91x/6.98x端到端延迟加速；满足A的LLM/agent直接性能优化。

**TL;DR**：提出LLM4LLM，一种部署感知的闭环优化框架，直接优化语言模型推理脚本，在真实工作负载上实现端到端延迟大幅加速，并弥合了内核基准与部署行为之间的差距。

**中文摘要**：大型语言模型已成为低层代码和内核优化的日益强大的智能体，但孤立的内核基准测试仅能代表语言模型推理中实际部署行为的近似。我们发现了一个从基准测试到部署的差距：在独立测试框架中看似正确且快速的候选内核，在集成到真实推理工作负载后，可能表现出不同的性能、安全性或阶段行为。我们引入了LLM4LLM，一个部署感知的闭环优化框架，它从目标推理脚本开始，提取阶段感知的优化任务，使用经验引导的 episodic 智能体进行搜索，并通过模型内验证接受补丁。在A100和H100 GPU上的十个语言模型推理工作负载中，LLM4LLM显著改善了每个评估模型的端到端延迟，在A100/H100上实现了3.91×/6.98×的几何平均加速；作为支持性的内核级证据，它还在KernelBench Level 2上达到了高达2.745×的几何平均加速。

**方法**：从目标推理脚本出发，提取阶段感知优化任务，使用经验引导的episodic智能体进行搜索，并通过模型内验证接受补丁，形成闭环优化。

**结果**：在A100和H100上的十个LM推理工作负载中，所有模型端到端延迟均改善，几何平均加速比分别为3.91×和6.98×；KernelBench Level 2上获得最高2.745×几何平均加速。

[返回索引](#快速索引)

---

<a id="2"></a>
## 2. [CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language](http://arxiv.org/abs/2609.00058v1)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-30
- **arXiv ID**：2609.00058
- **作者**：Qi Fan, An Zou, Yehan Ma
- **入选理由**：提出从自然语言生成并优化CUDA内核的agentic框架，包含中间结构化生成、合成验证与反馈自适应演化，在正确性与性能上同时优化；满足A的LLM/agent直接代码与kernel优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="3"></a>
## 3. [KernelArc: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2608.17071)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-20, 2026-08-17
- **arXiv ID**：2608.17071
- **作者**：Joyjit Kundu, Ben Stoffelen, Kaili Wang, Peter Vrancx, Ludovic Denoyer
- **入选理由**：核心是多智能体GPU内核自动优化框架，通过并行策略智能体、基准守卫和只读状态协调来优化真实硬件上的kernel，并在SOL-ExecBench取得领先；满足A的LLM/agent自动内核优化。

**TL;DR**：提出 KernelArc，一个多智能体 GPU 内核优化框架，在 H100/B200 上通过并行策略智能体与协调机制，在 SOL-ExecBench 多个任务上取得领先。

**中文摘要**：我们提出了 KernelArc，一个用于跨异构工作负载的自主 GPU 内核优化的多智能体框架。策略专化的智能体并行运行，并通过仅结论的共享内存、确定性基准测试守卫以及带有平台触发的草稿的只读跨智能体状态进行协调。我们在 NVIDIA H100 和 B200 GPU 上使用类别代表性的 SOL-ExecBench 工作负载评估了 KernelArc。生成的实现涵盖自定义 BF16 GEMM、静态 cuBLASLt Expert-API 配置表、融合的混合专家反向传播、形状门控的解码器层融合、原生 NVFP4 分组查询注意力以及分页预填充注意力。在 2026 年 7 月 30 日记录的公开 SOL-ExecBench 排行榜快照中，这些提交在代表性的 L1、L2、量化和 FlashInfer 任务上排名第一。轨迹支持论文的核心动机：共享的多智能体搜索可以在固定的候选预算内扩大探索并达到更强的现有方案，而各个协调特性的价值取决于内核和优化阶段。

**方法**：KernelArc 使用策略专化的多个智能体并行运行，通过仅结论的共享内存、确定性基准守卫和只读跨智能体状态（带平台触发的草稿）进行协调。

**结果**：在 NVIDIA H100 和 B200 上针对 6 类代表性负载生成优化实现，并在 SOL-ExecBench 排行榜快照（2026-07-30）的 L1、L2、量化和 FlashInfer 任务上排名第一。

[返回索引](#快速索引)

---

<a id="4"></a>
## 4. [RepoOMP: Repository-Aware Hotspot OpenMP Parallelization via Dependency-Aware Context Reduction](http://arxiv.org/abs/2608.05855v1)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-06
- **arXiv ID**：2608.05855
- **作者**：Yongjie Qian, Ke Gao, Zhibin Zhang, Shaohui Peng, Ling Li
- **入选理由**：RepoOMP对仓库热点进行OpenMP自动并行化，构造依赖上下文、区分规则与LLM agent，并在951个热点上做编译与负载检查且报告8-9倍平均加速，属于LLM/agent自动并行性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="5"></a>
## 5. [AgenticCANN: Automated Ascend C Operator Generation via Knowledge-Augmented Agentic Evolution](https://arxiv.org/abs/2607.26661)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-31, 2026-07-29
- **arXiv ID**：2607.26661
- **作者**：Junhao Qiu, Zidong Wang, Yansong Sun, Zhitong Ma, Ping Guo, Qingfu Zhang
- **入选理由**：核心是知识增强的智能体进化框架自动合成Ascend C算子，在NPU上优化内核性能，并报告可行性与6.65倍加速，满足A，且为LLM/agent直接自动优化代码。

**TL;DR**：提出了 AgenticCANN，一个知识增强的智能体进化框架，用于在低语料 NPU 环境中自动合成 Ascend C 算子，显著提升可行性并实现最高 6.65 倍加速。

**中文摘要**：Ascend C 算子优化对于 NPU（神经处理单元）推理性能至关重要，但需要深厚的硬件专业知识。尽管大语言模型（LLM）在自动生成 CUDA 内核方面显示出潜力，但 Ascend C 根本不同的编程模型引入了尚未探索的独特挑战。在本文中，我们提出了 AgenticCANN，一个专门为低语料 NPU 环境中自动合成 Ascend C 算子而设计的知识增强型智能体进化框架。为了克服在陌生硬件上严重的平台知识缺陷，AgenticCANN 整合了一个知识编排生成系统，该系统在开发生命周期中提供结构化的、多层次的领域洞察，以解决上游可行性瓶颈。在此基础之上，它采用了一种阶段自适应的智能体进化策略，该策略动态地将 LLM 交互模式与特定的生成和进化阶段对齐，平衡高探索性的候选发现与高收敛性的性能调优。在华为 Ascend 910B 上跨五个模式类别的六个算子进行的大量实验表明，我们的方法在逐元素和归一化算子上实现了 90% 到 100% 的可行性，在融合算子上实现了 56%，并在 1B Pangu 模型推理内核上实现了高达 6.65 倍的加速。进一步的分析表明，知识注入将逐元素算子的可行性从 57% 单调提升至 86%，证明了其通用性而非特定于算子的益处。

**方法**：设计知识编排生成系统以提供分层领域知识，解决上游可行性瓶颈；采用阶段自适应智能体进化策略，根据生成和优化阶段动态调整 LLM 交互模式，平衡探索与收敛。

**结果**：在华为 Ascend 910B 上，对六种算子（覆盖五类模式）实现元素级和归一化算子 90-100% 可行性、融合算子 56% 可行性，在 1B Pangu 模型推理内核上最高获得 6.65 倍加速；知识注入使元素级算子可行性从 57% 提升至 86%。

[返回索引](#快速索引)

---

<a id="6"></a>
## 6. [PerfAgent: Profiler-Guided Iterative Refinement for Repository-Level Code Optimization](https://arxiv.org/abs/2607.19653)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化、Profiling/程序分析
- **收录日期**：2026-07-24, 2026-07-22
- **arXiv ID**：2607.19653
- **作者**：Ryan Deng, Yuanzhe Liu, Bastian Lipka, Yao Ma, Xuhao Chen, Tim Kaler, Jatin Ganhotra
- **入选理由**：PerfAgent通过profiler引导与验证器循环，驱动LLM代理对仓库级代码进行迭代性能优化，相比baseline显著提升专家级补丁率，满足条件A。

**TL;DR**：PerfAgent通过性能分析引导和验证器循环，显著提升了LLM代理在仓库级代码优化上的表现，匹配专家速度的补丁率翻倍以上。

**中文摘要**：大型语言模型（LLM）代理现在在面向正确性的仓库级任务中表现良好，包括SWE-Bench问题解决和真实代码库中的功能实现。然而，它们在仓库级代码优化方面仍然存在困难，这需要在提高运行时性能的同时保持行为。在此场景中，仅通过测试是不够的；补丁必须保持行为、实现代码优化，并接近专家级加速。当前的代理常常忽略隐藏在抽象层和本地扩展背后的瓶颈，在实现浅层加速后就停止，或者未能充分测试代码补丁，从而可能静默破坏边缘情况。我们提出了PerfAgent，一个性能分析器引导、验证器在环的工作流程，为现成的编码代理提供所需的反馈，以发现真正的热点，超越第一个通过的补丁进行改进，并使用性能分析器证据（而非仅靠时间）来决定下一步优化什么。在两个具有挑战性的优化基准测试GSO和SWE-fficiency-Lite上，PerfAgent使匹配专家级补丁的比率比使用GPT-5.1的OpenHands提高了一倍以上，在GSO上从19.6%提升到39.2%，在SWE-fficiency-Lite上从26%提升到74%。它还以显著更低的成本超越了最优的五个采样基线，表明性能提升来自更好的反馈，而非额外的测试时采样。

**方法**：提出PerfAgent，一个包含性能分析器引导和验证器在环的工作流程，为代理提供热点发现、迭代优化和基于证据的决策反馈。

**结果**：在GSO和SWE-fficiency-Lite基准上，PerfAgent将匹配专家级补丁的比率分别从19.6%提升至39.2%和从26%提升至74%，并以更低成本超越最优采样基线。

[返回索引](#快速索引)

---

<a id="7"></a>
## 7. [Beyond the Need for Speed: Energy-Aware Code Generation via Simulation-Guided Reinforcement Learning](http://arxiv.org/abs/2607.04577v1)

- **相关度**：0.96
- **方向标签**：LLM/Agent 代码优化、Benchmark/评测
- **收录日期**：2026-07-06
- **arXiv ID**：2607.04577
- **作者**：Saurabhsingh Rajput, Tushar Sharma
- **入选理由**：用确定性架构仿真构建大规模语料并训练/RL能量感知代码生成模型，CARET显示真实能耗降低，直接面向自动节能代码生成，满足A且含B性质的数据与评估。

**TL;DR**：提出用确定性仿真代替硬件测量来训练节能代码模型，构建Green Tea数据集，通过监督微调和强化学习训练能量感知模型，并引入CARET指标评估，在保留问题上取得显著能效提升，同时揭示IPC作为能效代理的不可靠性。

**中文摘要**：代码模型严格优先考虑功能正确性，将软件能效视为未优化的副产品。训练模型生成节能代码需要可扩展的可重复反馈，而由于硬件测量的方差，物理硬件测量无法可靠提供。在本文中，我们用确定性架构模拟框架替代硬件性能分析，构建了Green Tea，一个包含来自1,474个C++问题的350万次评估的数据集。我们通过在能量对比对上进行监督微调，然后使用仿真在环反馈进行闭环强化学习（GRPO），训练了一个能量感知的代码模型。为了严格评估部署准备程度，我们引入了正确性调整后的能量总减少量（CARET），该指标明确惩罚牺牲功能以换取效率的代码。在143个保留问题上，我们的仿真在环流程实现了12.63%的CARET，几乎将单独微调的增益提高了三倍，并且在其有效输出中成功击败了人类专家参考的能效，占58.4%。此外，我们的分析揭示了IPC陷阱：像每周期指令数（IPC）这样的标准吞吐量代理在67.8%的问题上主动错误排序真实能效，证明了直接能量模拟的绝对必要性。通过发布我们的数据集和基础设施，我们绕过了重现所需的263,000 CPU小时，从根本上赋能社区部署固有节能的代码生成模型。

**方法**：使用确定性架构模拟构建Green Tea数据集（350万评估），通过能量对比对监督微调，再结合GRPO强化学习（仿真在环反馈）训练能量感知模型，并引入CARET指标惩罚牺牲功能换取效率的行为。

**结果**：在143个保留问题上取得12.63% CARET，比单独微调提升近三倍，58.4%有效输出超越人类专家能效；分析发现IPC在67.8%问题上错误排序能效。

[返回索引](#快速索引)

---

<a id="8"></a>
## 8. [Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory](http://arxiv.org/abs/2608.25570v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-08-26
- **arXiv ID**：2608.25570
- **作者**：Siyuan Chen, Runlin Hou, Shenxiu Wu, Yansong Sun, Junming Cao, Yiyu Zhang, Shudi Shao, Junhao Qiu, Zhichao Lu, Qingfu Zhang
- **入选理由**：LLM agents系统化地对硬件kernel进行编译、正确性测试、profile和修正确认，结合经验记忆持续优化并报告显著speedup，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="9"></a>
## 9. [FABRICA: Agentic CUDA-to-CSL Translation and Optimization for Wafer-Scale Systems](http://arxiv.org/abs/2608.25124v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优、Benchmark/评测
- **收录日期**：2026-08-25
- **arXiv ID**：2608.25124
- **作者**：Yuebo Luo, Eliu Huerta, Venkatram Vishwanath, Caiwen Ding, Rajeev Thakur, Le Chen
- **入选理由**：核心是FABRICA agentic框架，将CUDA kernel翻译并优化为Cerebras CSL，包含目标知识、失败修复和正确性门控优化，在WSE-3上几何平均速度提升3.47x。满足A类：跨架构自动翻译/kernel优化，明确改善性能并有硬件实测；也提供benchmark（FABRICA-Bench）。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="10"></a>
## 10. [Accelerated Genetic Programming Hyper-Heuristics for Simulation-Based Scheduling via Agentic AI](https://arxiv.org/abs/2608.19487)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-22, 2026-08-19
- **arXiv ID**：2608.19487
- **作者**：Heyang Thomas Li, Alexander Pletzer, Yuan Tian, Yi Mei, Mengjie Zhang
- **入选理由**：使用Claude agentic AI对Python项目调度仿真代码做系统性性能重构，以基准与正确性检查引导优化，运行时间从1298秒降至200秒以下并节省大量核心小时；满足A的LLM/agent直接代码性能优化。

**TL;DR**：使用 Claude 智能体 AI 对 Python 项目调度模拟进行系统重构，将测试运行时间从 1298 秒降至 200 秒以下，每年节省 400 万核心小时。

**中文摘要**：Python 因能够快速开发并提供丰富的数据分析、人工智能（AI）和机器学习生态系统，被广泛用于科学研究。然而，随着实验规模的扩大，定制化的研究代码可能会变得异常缓慢。这一挑战在离散事件项目调度模拟中尤为严峻，因为顺序状态更新、嵌套循环、条件评估和面向对象结构限制了编译型数值库和 GPU 加速库的优势。解决这些瓶颈通常需要迭代式性能分析、重构、测试和验证，但研究人员可能缺乏时间或专门的软件工程专业知识来进行底层优化。本文提出了一种在高级计算（HPC）环境中，使用 Claude 智能体 AI 对真实项目调度工作负载进行系统性重构的方法。在代表性基准测试和正确性检查的指导下，智能体识别瓶颈、实施有针对性的优化并评估其效果，而研究人员保留最终控制权。测试运行时间从 1,298 秒减少到 200 秒以下，且输出不变，每年节省 400 万核心小时（新西兰元 320,000）。

**方法**：提出一种使用 Claude 智能体 AI 的系统性重构方法，在 HPC 环境中，以代表性基准和正确性检查为指导，自动识别瓶颈、实施优化并评估效果，研究人员保留最终控制。

**结果**：测试运行时间从 1,298 秒降至 200 秒以下，输出不变，每年节省 400 万核心小时（NZ$320,000）。

[返回索引](#快速索引)

---

<a id="11"></a>
## 11. [AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification](http://arxiv.org/abs/2608.20711v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-08-21
- **arXiv ID**：2608.20711
- **作者**：Ji Liu, Puyuan Yang, Rongzhang Zheng, Fan Wang, Jinglin Wang, Muhammad A. Awad, Mortis Huang, Andy Chang, Zekai Li, Zeping Li, Zihao An, Yue Liu, Yuchen Yang, Jianghui Wang, Chushi Chen, Ziqiong Liu, Fuwei Yang, Dong Li, Wen Heng Chung, Shengcai Liu, Emad Barsoum
- **入选理由**：核心是AsmEvo，agentic assembly级优化AMD GPU kernel code object，通过功能等价验证和差分验证保全行为，在MI308X和MI300X上获得1.35x和1.09x-1.31x等速度提升。满足A类：直接在汇编层修改已编译kernel以优化性能并验证功能。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="12"></a>
## 12. [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](http://arxiv.org/abs/2608.12629v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、编译器优化、Kernel/自动调优
- **收录日期**：2026-08-12
- **arXiv ID**：2608.12629
- **作者**：Zihao Ye, Yingyi Huang, Hongyi Jin, Bohan Hou, Junru Shao, Zhongming Yu, Jinqi Chen, Meghan Cowan, Shiyi Cao, Shanli Xing, Hanfeng Chen, Vinod Grover, Tianqi Chen, Luis Ceze
- **入选理由**：核心是CAKE，编译器-agent协同设计：agent编写硬件明确的CAKE IR并迭代优化GPU kernel，配合验证、成本模型、诊断和演进式harness，在多个kernel上超越手调基线，并作为上游PR提交。满足A类：自动生成/优化GPU kernel并验证性能（如Flash-KMeans、Kimi Delta Attention的2.05x加速）。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="13"></a>
## 13. [Effect of Abstractions and Prompting Strategies on LLM-Guided High-Performance Optimizations](http://arxiv.org/abs/2608.08085v2)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-08
- **arXiv ID**：2608.08085
- **作者**：Jiří Klepl, Matyáš Brabec, Martin Kruliš
- **入选理由**：论文明确研究LLM指导的并行HPC代码自动优化，在PolyBench上生成优化C代码并报告正确率与实际测量性能提升，属于A类LLM直接代码性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="14"></a>
## 14. [HLSmith: An Expert-Guided Agentic Framework for C/C++-to-HLS Translation](http://arxiv.org/abs/2608.06791v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-07
- **arXiv ID**：2608.06791
- **作者**：Yuebo Luo, Ahmad Sedigh Baroughi, Philip Stachura, Le Chen, Venkatram Vishwanath, Zhenman Fang, Caiwen Ding
- **入选理由**：HLSmith由LLM/agent将C/C++翻译为优化的HLS/FPGA加速器，包含HLS专家规则、反馈式优化流程，并在PolyBench上验证功能正确性与几何平均4.24倍加速，满足A类HLS/agent自动性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="15"></a>
## 15. [SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System](http://arxiv.org/abs/2608.05033v2)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-08-05
- **arXiv ID**：2608.05033
- **作者**：Shiyang Li, Guangyan Sun, Jinwei Tang, Yanzhi Wang, Mingyi Hong, Caiwen Ding
- **入选理由**：用LLM agent系统针对不同稀疏模式自动构造/定制GPU kernel，并通过目标GPU上实测反馈迭代优化，获得显著加速，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="16"></a>
## 16. [Kernel Forge: An Agent Harness for LLM-based Generation and Optimization of CUDA Kernels](https://arxiv.org/abs/2607.24762)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-30
- **arXiv ID**：2607.24762
- **作者**：Joshua Brodsky, Dhravid Kumar, Savini Kashmira, Jayanaka Danatanarayana, Jason Mars, Krisztian Flautner, Lingjia Tang
- **入选理由**：核心任务是构建LLM智能体自动生成并优化CUDA kernel，以提升多个PyTorch模型的推理性能，并在真实GPU上给出加速比，满足A。

**TL;DR**：Kernel Forge是一个开源的端到端智能体框架，使用蒙特卡洛树搜索（MCTS）自动优化PyTorch模型的内核，在视觉、扩散和LLM模型上以少量迭代取得显著加速。

**中文摘要**：机器学习模型日益嵌入日常软件，其大部分运行时间消耗在一小组计算内核上，如矩阵乘法、卷积和归一化。优化这些内核是减少延迟和成本最直接的方式之一，但传统上需要专家工程师手工编写底层GPU代码。基于大型语言模型（LLM）的智能体系统现在可以以更少的人力生成和优化内核，然而现有工具大多在随机生成的张量和孤立内核上评估，生成独立的CUDA代码，开发者必须手动重新集成，主要仅针对LLM PyTorch模型，并且对检查和调试结果的支持有限。我们提出Kernel Forge，一个开源的端到端智能体框架，可以原地接受任何未经修改的PyTorch模型。Kernel Forge支持视觉、扩散和LLM工作负载，使用蒙特卡洛树搜索（MCTS）探索多个优化路径，而非单一线性优化链，并附带图形用户界面用于监控进度、检查候选内核和调试失败。我们在NVIDIA DGX Spark（配备GB10 GPU）上对涵盖视觉、扩散和LLM工作负载的四个PyTorch模型评估Kernel Forge。每个内核仅经过50次优化迭代，它优化了14个内核，使其性能优于PyTorch eager模式，在ResNet-50的adaptive_avgpool2d上达到1.52倍加速，在Stable Diffusion 3.5 Medium的group_norm上达到1.70倍，在Gemma 4 E2B的softmax上达到2.83倍，在Qwen 3.5 35B-A3B的softmax上达到1.54倍。

**方法**：提出Kernel Forge，一个开源的端到端智能体框架，接受任何未修改的PyTorch模型，使用蒙特卡洛树搜索（MCTS）并行探索多个优化路径，并提供图形用户界面进行监控和调试。

**结果**：在四个PyTorch模型上，每个内核仅50次优化迭代，优化了14个内核，相比PyTorch eager模式取得1.52×至2.83×的加速。

[返回索引](#快速索引)

---

<a id="17"></a>
## 17. [RLPF: Reinforcement Learning from Performance Feedback for Code Generation](http://arxiv.org/abs/2607.27271v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-29
- **arXiv ID**：2607.27271
- **作者**：Huihao Jing, Haozhe Cui, Wenbin Hu, Shaojin Chen, Haochen Shi, Changxuan Fan, Yuxuan Liu, Hanyu Yang, Sirui Zhang, Ziyi Chen, Haoran Li, Yangqiu Song
- **入选理由**：RLPF将运行时间性能反馈引入代码agent训练：先按执行进度排序失败程序，再按相对基准的加速对正确程序排名，最终提升可运行且高效的代码比例，真实性能验证明确，属于面向代码生成性能优化的LLM方法。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="18"></a>
## 18. [Multi-level Code Optimization via Mixture of Prompts](https://arxiv.org/abs/2607.23665)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Profiling/程序分析
- **收录日期**：2026-07-29, 2026-07-26
- **arXiv ID**：2607.23665
- **作者**：Yun Peng, Jun Wan, Jiakun Liu, Shuzheng Gao, David Lo, Xiaoxue Ren
- **入选理由**：Optimo基于差异profiling和多级提示混合架构，由LLM自动识别瓶颈并施加从算法到API的四级代码优化，在COFFE/Effibench上验证正确性与加速，满足A。

**TL;DR**：Optimo是一种基于LLM和混合提示架构的多级代码优化方法，通过差异分析识别瓶颈并应用多级优化，在两种基准上显著提升代码效率。

**中文摘要**：运行时效率是影响软件质量和用户满意度的关键因素。有许多方法被提出来优化代码以提高运行时效率。传统的代码优化方法在编译期间对静态语言的中级表示（IR）进行操作。它们是有效的，但难以处理不需要编译的动态语言。最近，大型语言模型（LLM）已被用于直接优化动态语言的源代码。然而，这些方法无法识别合适的优化目标，并且通常进行不全面的单级优化。为了解决这些挑战，我们提出了Optimo，一种基于新型混合提示（MoP）架构的多级LLM代码优化方法。在MoP架构中，Optimo通过差异分析识别时间关键的代码结构作为性能瓶颈。然后，这些结构被路由到一些优化策略，类似于MoE中的专家模型，每个策略针对特定代码模式进行优化。与传统方法仅关注语句级优化不同，Optimo在四个抽象级别上操作，从粗粒度的算法改进到细粒度的API使用优化。我们在两个代码效率基准测试COFFE和Effibench上评估了Optimo。我们的结果表明，Optimo实现了高达57.48%的opt%（即优化后程序正确且比原始程序快至少10%的百分比），在优化人类编写代码时实现了高达3.97倍的加速，并且在opt%上始终优于最佳基线，最高达96.51%。此外，Optimo在优化LLM生成的代码时实现了高达42.42%的opt%和高达13.51倍的加速。

**方法**：提出Optimo，基于Mixture-of-Prompts架构，通过差异分析识别时间关键代码结构，路由到不同优化策略，在四个抽象级别（算法改进到API优化）进行优化。

**结果**：在COFFE和Effibench上，对人类编写代码达到57.48% opt%和3.97倍加速，对LLM生成代码达到42.42% opt%和13.51倍加速，opt%优于基线最高96.51%。

[返回索引](#快速索引)

---

<a id="19"></a>
## 19. [VPR-Evolve: Multi-Agent-Driven Algorithm Evolution for FPGA Place and Route](http://arxiv.org/abs/2607.24998v1)

- **相关度**：0.95
- **方向标签**：搜索与进化优化
- **收录日期**：2026-07-27
- **arXiv ID**：2607.24998
- **作者**：Qihang Wu, Taizun Jafri, Aman Arora, Vidya A. Chhabria
- **入选理由**：VPR-Evolve用多智能体LLM直接在VPR源码级别提出、实现和评估修改，以时延、线长和工具运行时间复合指标为目标，并通过完整构建/运行验证效果，属于基于进化的自动源码性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="20"></a>
## 20. [Multi-Source and Cross-Scenario Strategy-Guided Code Optimization](https://arxiv.org/abs/2607.20353)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、优化策略检索
- **收录日期**：2026-07-24, 2026-07-22
- **arXiv ID**：2607.20353
- **作者**：Yuwei Zhao, Qianyu Xiao, Ye Cui, Yijun Yu, Yingfei Xiong
- **入选理由**：MoST跨场景整合多知识源，从历史优化提交提取优化策略并生成规则指导LLM进行代码优化，在真实项目上获得显著性能提升，满足A与C。

**TL;DR**：提出MoST框架，跨场景整合多知识源，通过聚类和示例转移生成静态分析规则，显著提升代码优化效果。

**中文摘要**：自动代码优化通过重构源代码来提升程序性能，近期研究使用大语言模型（LLM）生成优化补丁。最新的方法是策略引导型：它们从历史优化提交中提炼策略作为静态分析规则，并利用这些规则匹配代码位置供LLM优化。然而，这些方法存在两个局限性：（1）策略可能来自其他知识源，如教科书和网页，但现有方法无法利用它们；（2）某个策略可能适用于不同场景，例如不同编程语言，但现有方法只能将策略形式化为其源提交所属的场景。为解决这些局限性，我们提出了MoST，一个基于LLM的代码优化框架，它跨场景整合多个知识源。MoST将不同知识源中的项目统一表示为证据对象，以跨源和跨场景的方式聚类以识别策略，并在必要时将其转移到目标场景以生成静态分析规则。为实现这一过程，MoST采用了一种新颖的自平衡加权聚类算法来平衡来自不同知识源的证据对象，以及一种新颖的示例转移过程来确保跨场景转移时生成规则的质量。在一个包含151个C/C++、150个Python和50个Rust历史优化任务的基准测试中，与SemOpt相比，MoST产生的与开发者补丁完全相同或语义等价的补丁分别多出24.44%-180.00%和21.88%-37.50%。在优化15个真实世界项目时，MoST在项目性能测试中实现了19.72%-717.42%的最大改进和4.44%-258.17%的平均改进，显著优于SemOpt和Codex。

**方法**：MoST将不同知识源中的项目统一表示为证据对象，采用自平衡加权聚类算法跨源跨场景聚类以识别策略，并通过示例转移过程将策略迁移到目标场景生成静态分析规则。

**结果**：在151个C/C++、150个Python和50个Rust任务上，MoST比SemOpt多产生24.44%-180.00%的完全匹配补丁和21.88%-37.50%的语义等价补丁；在15个真实项目中，性能最大提升19.72%-717.42%，平均提升4.44%-258.17%。

[返回索引](#快速索引)

---

<a id="21"></a>
## 21. [Technical Report: AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell](http://arxiv.org/abs/2607.16831v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-18
- **arXiv ID**：2607.16831
- **作者**：Hyunjun Shin, Jiseung Jang, Jaewoo Maeng, Hyunjun Kim
- **入选理由**：核心是AI辅助GPU kernel优化，针对Gated DeltaNet在NVIDIA Blackwell上的解码和预填充，官方1.58x加速，并作为案例研究强调端到端系统问题。满足A类：自动修改/优化kernel代码以改善延迟和吞吐，且有实际性能验证。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="22"></a>
## 22. [From Custom-Fit to Portable: Bridging the Gap Between Synthesized and Engineered GPU Query Execution](http://arxiv.org/abs/2607.07632v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-08
- **arXiv ID**：2607.07632
- **作者**：Ivan Donchev Kabadzhov, Eugenio Marinelli, Raja Appuswamy
- **入选理由**：满足A：SHADB利用LLM在自动profile-guided优化循环中合成特化的CUDA/HIP内核，以逼近内存带宽、相对于引擎实现7.4倍加速，并有真实性能验证；随后把可泛化优化迁移到SYCLDB中。核心是为数据库查询自动生成并优化GPU代码。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="23"></a>
## 23. [Copper: Unifying Correctness and Performance Specification in Code Generation](http://arxiv.org/abs/2607.03130v1)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-03
- **arXiv ID**：2607.03130
- **作者**：André Lizardo, Raul Barbosa
- **入选理由**：Copper将性能规格与形式验证结合进AI代码生成，自动生成经真实运行时间/内存验证的高效正确代码，性能优化是核心目标，满足A。

**TL;DR**：Copper框架通过结合形式化验证与性能感知规范，生成了可证明正确且高效的代码，在多样任务上显著优于基线AI生成代码。

**中文摘要**：生成式人工智能在生成功能正确的代码方面取得了显著进展，但确保同时具备正确性和性能仍然是一个开放挑战。我们提出了Copper，一个结合了形式化验证与性能感知规范的框架，用于生成可证明正确且高效执行的代码。我们的方法将AI驱动的代码合成与形式化验证工具以及自动化性能分析循环相结合。在多种算法和实际编程任务上的评估表明，与基线AI生成的代码相比，Copper生成的解决方案在满足严格正确性保证的同时，在运行时和内存效率方面实现了显著改进。这项工作表明，在AI辅助编程中弥合可信性与性能之间的鸿沟是可行的，为可靠、高性能代码生成提供了一条实用路径。

**方法**：将AI驱动的代码合成与形式化验证工具及自动化性能分析循环相结合，构建Copper框架。

**结果**：在多种算法和实际编程任务上，Copper生成的代码在运行时和内存效率上显著优于基线AI代码，同时满足严格正确性保证。

[返回索引](#快速索引)

---

<a id="24"></a>
## 24. [Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation](http://arxiv.org/abs/2607.01590v2)

- **相关度**：0.95
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-02
- **arXiv ID**：2607.01590
- **作者**：Junyi Wen, Ruiyan Zhuang, Yongjia Xu, Pengtu Li, Rui Zou, Hongyi Chen, Chingman Wan, Puxu Yang, Wuhui Chen, Yanlin Wang
- **入选理由**：Hawk面向NPU内核生成，利用硬件感知知识表示、检索与蒸馏迭代生成可执行高性能内核，在真实负载上取得相较基线2.2x加速，满足A。

**TL;DR**：Hawk是一个无需训练的NPU内核生成框架，通过硬件感知知识表示和检索，将准确率从49.4%提升到80.0%，执行加速高达2.2倍。

**中文摘要**：为神经处理单元（NPU）开发高性能内核是一个关键的行业瓶颈，要求开发者手动处理隐式的硬件约束和严格的内存层次结构。虽然大语言模型提供了巨大的自动化潜力，但由于缺乏硬件特定的先验知识，它们在NPU上会彻底失败。天真地移植来自类似NPU内核的代码片段可能通过编译器，但会持续触发运行时崩溃和性能下降，因为盲目违反了潜在的硬件约束。为了克服这个问题，我们引入了Hawk，一个无需训练的框架，通过三个核心模块利用硬件感知知识：（1）运行时知识合成模块，采用三分可执行知识表示来将错误上下文与可执行语义固有地耦合；（2）瓶颈感知知识检索模块，实现2D检索范式将查询投影到正交的语法和硬件对齐的语义空间；（3）效果驱动知识蒸馏模块，利用LLM驱动的语义仲裁，通过基于经验执行反馈修剪错误和整合冗余来持续蒸馏知识。在真实NPU工作负载上的广泛评估表明，Hawk将生成准确率从49.4%提升到80.0%，并且相比最先进的基线实现了高达2.2倍的执行加速。

**方法**：提出Hawk框架，包含三个模块：运行时知识合成（三分可执行知识表示耦合错误上下文与语义）、瓶颈感知知识检索（2D检索范式投影查询到语法和硬件对齐空间）、效果驱动知识蒸馏（LLM仲裁根据执行反馈修剪冗余）。

**结果**：在真实NPU工作负载上，生成准确率从49.4%提升至80.0%，执行速度相比最先进基线最高提升2.2倍。

[返回索引](#快速索引)

---

<a id="25"></a>
## 25. [HIERA: Workload-Aware Planning Across Implementation Spaces for GPU Kernel Optimization](http://arxiv.org/abs/2608.21157v1)

- **相关度**：0.94
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-08-21
- **arXiv ID**：2608.21157
- **作者**：Jinghao Wang, Qiqi Gu, Chenpeng Wu, Jianguo Yao, Haibing Guan, Xijun Li
- **入选理由**：提出HIERA，利用LLM在PyTorch算子、CUDA库和自定义CUDA kernel间选择实现空间并迭代优化，显著提升性能和采样效率，满足A中LLM/agent自动优化GPU kernel的核心目标。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="26"></a>
## 26. [Compiler-Grounded Hierarchical Diagnosis for LLM-Based Triton Kernel Optimization](https://arxiv.org/abs/2607.23089)

- **相关度**：0.94
- **方向标签**：Kernel/自动调优、编译器优化、LLM/Agent 代码优化
- **收录日期**：2026-07-29, 2026-07-25
- **arXiv ID**：2607.23089
- **作者**：Dongjie Chen, Ping Zhao, Bohua Zhan, Yulong Wang, Shushu Chen, Liangjun Feng, Hao Zhou, Min Shen, Linmu Wang, Weijia Sheng, Xiangyu Wei, Weijie Ding, Jianhui Huang, Yaoqing Gao
- **入选理由**：系统针对Triton kernel采用编译器grounded的分层诊断，将运行时症状与IR结构和编译器行为关联后提出证据支持的源码重写，在Ascend NPU上获得平均4.35倍加速，满足A。

**TL;DR**：提出一个基于编译器的层次化优化框架，通过渐进式跨层诊断实现Triton内核自动优化，在Ascend NPU上获得平均4.35倍加速。

**中文摘要**：近期大语言模型（LLM）的进展使得自动内核生成和优化成为可能，但现有大多数方法依赖于编译反馈和性能分析度量等表面信号。这些信号表明内核运行缓慢，但并未揭示后端编译器为何未能实现有利的优化，尤其是在新兴加速器（如NPU）上。因此，我们将内核优化表述为一个渐进式跨层诊断问题，该问题将运行时症状与IR结构和编译器行为联系起来，然后再重写源代码。基于这一洞察，我们提出了我们的系统，一个针对Triton内核的、基于编译器的层次化优化框架。该系统从轻量级模式分诊和性能分析诊断升级到IR归因和基于编译器的分析，仅在需要更深层证据时进行，然后提出基于证据的源代码级重写。我们在面向Ascend NPU的Triton上实现了该系统，并在来自标准化NPUKernelBench的Ascend 950基准测试中的37个成功转换条目上进行了评估。在这些条目中，系统从初始Triton内核到优化后的内核实现了4.35倍的几何平均加速和2.73倍的中位数加速；22/37个条目加速超过2倍，13/37个条目加速超过5倍。完整的分布范围从接近基线的条目到大幅收益，这激励了我们透明地报告当前系统的范围和局限性。

**方法**：将内核优化视为渐进式跨层诊断问题，从轻量级模式分诊到编译器接地分析，再提出证据驱动的源代码重写。

**结果**：在37个NPU基准测试中，几何平均加速4.35倍，中位数加速2.73倍，其中22个超过2倍，13个超过5倍。

[返回索引](#快速索引)

---

<a id="27"></a>
## 27. [MKEvolve: A Modular Multi-Agent Framework for Kernel Code Generation](https://arxiv.org/abs/2607.20501)

- **相关度**：0.94
- **方向标签**：Kernel/自动调优、LLM/Agent 代码优化
- **收录日期**：2026-07-25, 2026-07-26
- **arXiv ID**：2607.20501
- **作者**：Jason Yoo, Rajarshi Saha, Shaowei Zhu, Tao Yu, Wei Tang, Youngsuk Park
- **入选理由**：MKEvolve用模块化多智能体框架对PyTorch模块进行分解，并为每子模块用LLM beam search迭代生成/优化Triton kernel，同时在正确性和加速比上超越baseline，满足A。

**TL;DR**：MKEvolve通过迭代分解和LLM驱动的子内核优化，提高了硬件内核生成的正确性和性能，并减少了LLM令牌消耗。

**中文摘要**：尽管基于LLM的代码生成取得了快速进展，但为硬件加速器编写正确且高性能的内核仍然是扩展现代机器学习工作负载的关键瓶颈。我们提出了MKEvolve（模块化内核进化），一个迭代地共同进化复杂PyTorch模块的模块化分解和每个子模块的LLM生成内核的框架，通过跨迭代的分裂和融合来细化分解，同时通过LLM驱动的束搜索独立改进每个子内核。生成的内核是独立验证的子内核的程序化组合，使其可配置（子内核实现可互换）、可解释（错误和加速可追溯到特定子内核），并且易于适应相关模型架构。在KernelBench L2和L3上使用Triton进行的实验，涵盖了多算子序列和完整模型架构，表明与端到端直接合成基线相比，MKEvolve提高了正确性和加速，同时将LLM令牌使用量减少了高达35%。

**方法**：提出MKEvolve框架，迭代共进化模块分解和子内核，通过分裂和融合调整分解，并使用LLM驱动的束搜索独立优化每个子内核。

**结果**：在KernelBench L2和L3上，MKEvolve相比端到端直接合成基线提高了正确性和加速，同时减少了高达35%的LLM令牌使用量。

[返回索引](#快速索引)

---

<a id="28"></a>
## 28. [FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications](http://arxiv.org/abs/2607.18171v2)

- **相关度**：0.94
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-20
- **arXiv ID**：2607.18171
- **作者**：Krish Agarwal, Zhuoming Chen, Yanyuan Qin, Zhenyu Gu, Atri Rudra, Beidi Chen
- **入选理由**：满足A：FlashRT引导通用编码agent将简单参考实现自动转换为优化后的多GPU部署，涉及代码变换、IR分析、并行化策略选择，并以延迟和吞吐为指标进行测量门控优化，在B200/MI355X上有明确性能验证。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="29"></a>
## 29. [PERFOPT-Bench: Evaluating Coding Agents on Software Performance Optimization](https://arxiv.org/abs/2607.07744)

- **相关度**：0.94
- **方向标签**：Benchmark/评测
- **收录日期**：2026-07-08, 2026-07-12
- **arXiv ID**：2607.07744
- **作者**：Yingyun Cui, Yi Xie, Piaohong Wang, Jiawei Ma, Bo Liu, Liangliang Cao
- **入选理由**：PERFOPT-Bench提供完整性能工程循环评测：隐藏正确性测试、验证speedup、轨迹审计，是专门面向coding agent软件性能优化的基准，满足B。

**TL;DR**：提出 PERFOPT-Bench 基准，评估代码智能体在性能优化任务上的能力，发现优化性能取决于工作负载，且原始速度提升可能不安全。

**中文摘要**：代码智能体基准测试在很大程度上衡量了智能体能否生成功能正确的补丁，但生产软件还要求在真实执行目标上实现可测量的速度提升。性能优化是一个独特的智能体任务：智能体必须分析性能剖析，诊断跨层瓶颈，在编辑代码时不破坏正确性，并验证增益是可复现的而非测量假象。我们引入了 PERFOPT-Bench，一个用于评估这一完整性能工程循环的基准。每个任务提供一个正确但故意次优的代码库，并要求智能体改进一个目标性能指标；评分需要隐藏的正确性测试、验证的速度提升测量以及轨迹级审计。我们评估了 7 个不同 LLM 和智能体框架的智能体栈在 7 个长时优化任务上的表现。结果显示，优化性能是工作负载相关的，而非仅由模型身份决定：没有单一栈占优，改变智能体框架会显著改变同一 LLM 在不同任务上的速度提升分布。我们进一步发现，原始速度提升作为基准分数是不安全的，因为一些大的增益来自于基准特定的捷径利用；一个探索性的中继试点表明，在初始会话停止后，从外部优化摘要重新开始可以恢复额外的改进空间。该基准及我们的评估可在 https://anonymous.4open.science/r/Dataset-D3CC 获取。

**方法**：构建 PERFOPT-Bench 基准，包含正确但次优的代码库，要求智能体改进性能指标；使用隐藏的正确性测试和验证的速度提升进行评分，并评估多种 LLM 和智能体框架。

**结果**：优化性能与工作负载相关，而非模型身份；无单一栈占优；原始速度提升可能因捷径利用而不安全；中继试点可恢复额外优化空间。

[返回索引](#快速索引)

---

<a id="30"></a>
## 30. [DSEffi-Bench: Demystifying Large Language Models' Capability in Efficient Data Science Code Generation](https://arxiv.org/abs/2608.30248)

- **相关度**：0.93
- **方向标签**：Benchmark/评测
- **收录日期**：2026-09-02, 2026-08-31
- **arXiv ID**：2608.30248
- **作者**：Zhihao Gong, Junzhe Yu, Dong Huang, Zeyu Sun, Jie M. Zhang, Dan Hao
- **入选理由**：首个专门针对LLM生成数据科学代码执行效率构建的基准，含1000个实例、压力测试和人工验证参考，并用效率分数与税类诊断刻画效率问题；满足B的专用性能benchmark。

**TL;DR**：提出 DSEffi-Bench 基准，发现正确性不能反映 DS 代码生成效率，并验证了效率诊断可用于优化。

**中文摘要**：当前数据科学（DS）代码生成基准将正确性等同于质量，忽略了正确解决方案之间可能相差数量级的执行时间差异。我们引入了 DSEffi-Bench，这是首个专门针对 LLM 生成的 DS 代码执行效率的基准，包含 10 多个 DS 库的 1,000 个实例，并配有压力测试框架和人工验证的参考实现。评估了 3 个层级共 16 个模型后，我们发现仅凭正确性无法表征效率：GPT-5.4 在正确性（Pass，66.9%）上领先，但其效率得分（B|P，71.7%）几乎与 GPT-5.4-mini（71.6%）持平，后者解决的任務少了 47 个；Kimi-K2.5 在前沿模型中正确性排名最低（40.2%），却在全部 16 个模型中取得了最高的效率得分（73.6%）。一个由人工标注的五类分类法显示，79.1% 的效率缺陷超出了算法复杂性范畴，根因与特定领域相关，且不同模型层级和库之间存在不同的失败模式。两项探索性实验提供了初步证据，表明这些诊断可以指导改进：通过分类法引导的优化实现了高达 +14.7% 的效率提升，并通过基于库的条件路由，以 13.0 倍更低的成本在效率上接近 Claude-Opus-4.6 的 Best@3。

**方法**：构建包含 1,000 个实例、覆盖 10+ DS 库的 DSEffi-Bench 基准，使用压力测试和人工验证参考，评估 16 个模型，并进行人工标注分类和探索性优化实验。

**结果**：正确性与效率不一致：效率最佳模型并非正确性最高；79.1% 的效率缺陷源于领域特定因素；分类引导优化可提升 +14.7% 效率，路由方法以 13.0 倍低成本接近顶尖模型。

[返回索引](#快速索引)

---

<a id="31"></a>
## 31. [DataKernelBench: Can LLMs Optimize Database Queries on GPUs?](http://arxiv.org/abs/2608.25061v2)

- **相关度**：0.93
- **方向标签**：Benchmark/评测、LLM/Agent 代码优化
- **收录日期**：2026-08-25
- **arXiv ID**：2608.25061
- **作者**：Gokul Karthik Kumar, Yotam Perlitz, Corey Lammie, Andrea Giovannini, Katja Hose
- **入选理由**：构建针对GPU数据库查询优化的LLM benchmark（DataKernelBench），要求LLM优化CUDA/Triton kernel并验证实际加速，满足B（专用自动程序优化benchmark）且直接面向LLM驱动的kernel自动优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="32"></a>
## 32. [RAGas: Retrieval-Augmented Gas Optimization for Smart Contracts with Continuous Knowledge Integration](https://arxiv.org/abs/2608.15857)

- **相关度**：0.93
- **方向标签**：LLM/Agent 代码优化、优化策略检索
- **收录日期**：2026-08-19
- **arXiv ID**：2608.15857
- **作者**：Yishun Wang, Wenjin Yi, Wenkai Li, Zongwei Li, Xiaoqi Li
- **入选理由**：核心是面向智能合约Gas消耗的自动识别与修复，即针对代码执行成本（Gas/能耗）的自动优化，并有实际合约的Gas降低验证，满足A。

**TL;DR**：提出RAGas，一个基于检索增强生成的三阶段框架，利用大型语言模型自动检测并修复以太坊智能合约中的Gas低效问题，在实际合约上最高减少11%的Gas消耗。

**中文摘要**：以太坊现已深度融入关键任务领域，包括金融、医疗保健和供应链管理。执行费用（通常称为Gas）随函数计算复杂度的增加而增长。以太坊上的智能合约会产生执行费用（称为Gas），其随计算复杂度增加而增加。因此，在保持功能等价的前提下优化Gas密集型代码可显著降低部署成本。现有系统尚未能持续利用不断演变的Gas消耗模式。我们系统性地分析了导致过度Gas消耗的语法和语义结构。这产生了六个高层次类别，涵盖十二个细粒度反模式，支撑了一个精选知识库。我们通过RAGas将这些见解付诸实践，这是一个三阶段检索增强生成框架，利用大型语言模型来定位并自动修复Gas低效问题。在已部署合约上的实验表明，RAGas可将Gas消耗最多降低11%，并在检测表现出Gas浪费的代码片段方面实现了高精确率和召回率。

**方法**：系统分析导致过度Gas消耗的语法和语义结构，归纳出6个高层次类别、12个细粒度反模式，构建知识库，并设计RAGas三阶段检索增强生成框架，用大模型定位和自动修复Gas低效。

**结果**：在已部署合约上，RAGas将Gas消耗最多降低11%，并在检测Gas浪费代码片段上实现高精确率和召回率。

[返回索引](#快速索引)

---

<a id="33"></a>
## 33. [T-LLM Compiler: Trusted LLM-based Code Optimization and Verification Framework](https://arxiv.org/abs/2608.14953)

- **相关度**：0.93
- **方向标签**：LLM/Agent 代码优化、编译器优化
- **收录日期**：2026-08-19, 2026-08-15
- **arXiv ID**：2608.14953
- **作者**：Zahra Fazel, Sunanda Gamage, Shayan Shirahmad Gale Bagi, Amir H. Ashouri, Tomasz S. Czajkowski, Bryan Chan, Reza Azimi, Yaoqing Gao
- **入选理由**：T-LLM Compiler利用LLM做高层代码变换，并结合传统编译器与验证工具迭代优化代码，在PolyBench/C上报告正确性与加速比，核心是自动代码优化，满足A。

**TL;DR**：T-LLM编译器结合LLM高级变换、传统编译器和验证工具，在PolyBench/C上实现高正确性和加速，最高准确率83.3%，最高加速16.1%，平均加速26.7%。

**中文摘要**：大型语言模型（LLM）的最新进展为将高级代码变换应用于代码优化领域带来了机遇，并且它已成为LLM执行的最基本任务之一；然而，目前LLM在广泛的代码优化任务中面临困难，这既源于代码的复杂性，也源于无法独立验证变换的正确性。在本文中，我们提出了可信LLM（T-LLM）编译器，它通过高级LLM代码变换、传统编译器和验证工具的协作，推动了编译器技术的进步。实验结果表明，在一组PolyBench/C基准测试上，它能够显著提高代码正确性。我们的方法通过验证策略促进迭代式代码优化工作，使纠正措施得以实施。通过这种方法，T-LLM编译器在PolyBench/C基准测试上实现了高达83.3%的代码优化准确率和高达16.1%的加速比，变换后的代码相对于标准基线平均达到26.7%的加速比。此外，我们将项目的源代码发布给开源社区。

**方法**：提出T-LLM编译器，将LLM高级代码变换与传统编译器及验证工具协同工作，通过验证策略支持迭代优化和纠正措施，确保变换的正确性并持续改进。

**结果**：在PolyBench/C基准上，代码优化准确率高达83.3%，最高加速比16.1%，变换后代码相对标准基线平均加速26.7%，并开源源代码。

[返回索引](#快速索引)

---

<a id="34"></a>
## 34. [PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX](http://arxiv.org/abs/2608.17379v2)

- **相关度**：0.93
- **方向标签**：Benchmark/评测、Kernel/自动调优、LLM/Agent 代码优化
- **收录日期**：2026-08-18
- **arXiv ID**：2608.17379
- **作者**：Genghan Zhang, Yixin Dong, Chengze Fan, Zhichen Zeng, Yueming Yuan, Shaowei Zhu, Kunle Olukotun
- **入选理由**：核心是PTXBench，一个面向LLM使用架构特定PTX优化GPU kernel的benchmark，测量正确性、目标指令执行和相对frontier库的加速，并在H100/B200上覆盖GEMM和attention。满足B类：专门面向自动程序性能优化的benchmark和评估方法；同时支持LLM kernel优化评估。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="35"></a>
## 35. [A Barrier-Free Synchronization Algorithm for Multi-Engine AI Accelerators](http://arxiv.org/abs/2608.13757v1)

- **相关度**：0.93
- **方向标签**：编译器优化
- **收录日期**：2026-08-13
- **arXiv ID**：2608.13757
- **作者**：Chungha Sung, Nikil V. Shyamsunder, Hanliang Zhang, Daniel Kroening, Joonwon Choi
- **入选理由**：提出AI加速器上的无屏障同步算法，作为编译后端pass在结构化控制流上精确执行依赖，相比基于屏障的基线降低10-45%延迟，满足A的编译器优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="36"></a>
## 36. [Enhancing SLMs for Sustainable Code Optimization in Radio-Astronomy](https://arxiv.org/abs/2607.21677)

- **相关度**：0.93
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-27, 2026-07-28, 2026-07-23
- **arXiv ID**：2607.21677
- **作者**：Elisa Chiarotto, Jingbo Li, P. Chris Broekema, Rob V. van Nieuwpoort
- **入选理由**：本文用增强SLM结合多采样生成和编译器反馈，对射电天文软件进行自动代码优化与加速器移植，目标是计算效率提升并验证性能，满足A。

**TL;DR**：本工作针对LOFAR望远镜升级带来的计算需求激增问题，提出使用增强的小型语言模型（SLM）通过多采样生成和编译器反馈来优化代码，在降低能源消耗的同时实现高性能优化，且该方法可扩展至其他大规模科学代码。

**中文摘要**：最近的大型语言模型（LLM）能够生成和优化复杂代码。我们研究了使用LLM为大规模科学（特别是射电天文学和可持续性）生成和优化代码。LOFAR望远镜目前正在进行升级，显著增加了观测的天空区域，同时更快地处理更多数据。然而，这预计将使计算需求增加40倍。因此，这一升级关键取决于现有软件的严格性能优化和加速器的广泛采用。代码库非常庞大，使得这项任务艰巨。因此，我们研究并展示了一种人工智能驱动的方法，旨在帮助开发人员评估和优化他们的代码，包括移植到硬件加速器。LOFAR社区致力于可持续解决方案，需要在不超过能源预算的情况下实现这些改进。因此，我们需要优化现有代码或将其移植到加速器，同时确保优化过程本身也是节能的。这带来了挑战，因为LLM是能源密集型的。因此，我们建议使用小型语言模型（SLM）来限制环境影响。在本文中，我们展示了如何通过使用智能体AI来增强SLM。我们通过两种方式扩展SLM以改善代码生成质量和性能：首先采用多采样生成策略，其次结合编译器反馈。我们证明，多采样SLM可以用更少的计算资源匹配或超越较大的单次生成模型，并且将编译器输出反馈回SLM会导致所有测试模型的一致改进。我们的方法是通用的，还可以在代码生成流程中使用检索增强生成（RAG）以及静态和动态分析工具。

**方法**：采用智能体AI增强SLM，通过两种方式提升代码生成质量：一是多采样生成策略，二是将编译器输出反馈给SLM以迭代改进；同时该方法支持集成检索增强生成（RAG）及静态/动态分析工具。

**结果**：多采样SLM在消耗更少计算资源的情况下，能匹配或超越较大单次生成模型的性能；将编译器输出反馈给SLM可一致性地提升所有测试模型的代码优化效果。

[返回索引](#快速索引)

---

<a id="37"></a>
## 37. [Harness Engineering for LLM-Driven GPU Kernel Generation](http://arxiv.org/abs/2607.17979v1)

- **相关度**：0.93
- **方向标签**：LLM/Agent 代码优化、Kernel/自动调优
- **收录日期**：2026-07-20
- **arXiv ID**：2607.17979
- **作者**：Yue Shui, Chenyu Ma, Hangfei Xu, Shengzhao Wen, Yanpeng Wang
- **入选理由**：面向LLM驱动的GPU kernel自动生成/优化构建了评测与profile驱动优化框架，在MLSys竞赛五个算子中对比baseline取得真实延迟加速，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="38"></a>
## 38. [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541)

- **相关度**：0.93
- **方向标签**：Benchmark/评测、Kernel/自动调优、LLM/Agent 代码优化
- **收录日期**：2026-07-18, 2026-07-19, 2026-07-20, 2026-07-16
- **arXiv ID**：2607.14541
- **作者**：Lingyun Yang, Yuxiao Wang, Shenghao Liang, Linfeng Yang, Daocheng Ying, Chunbo You, Rui Zhang, Luping Wang, Yinghao Yu, Guodong Yang, Liping Zhang
- **入选理由**：提出面向生产推理轨迹的GPU kernel基准Atrex-Bench，并发布profile-driven kernel优化智能体AKA，有真实kernel性能验证，满足A与B。

**TL;DR**：提出Atrex-Bench基准测试，从生产轨迹采样，并推出AKA代理优化内核，显著提升性能。

**中文摘要**：现有的GPU内核生成基准测试从合成或精选来源中提取问题，这些来源与部署的工作负载存在差异。我们提出了Atrex-Bench，这是一个基准测试，其30个算子和440种形状直接从全集群生产推理轨迹中采样，针对计算受限、内存丰富的GPU。每个问题都带有一个重要性权重，该权重源自其在观察到的GPU时间中的份额，按应用卡时加权，并根据其运行的推理阶段分别计算，同时每个问题还带有屋顶线上限，因此总体得分突出了消耗最多推理时间的内核。在Atrex-Bench上评估六种前沿编码代理显示，即使是最好的普通模型也只能达到生产算子上硬件屋顶线的大约10%；而仅靠正确性会高估能力，因为很大一部分表面通过率来自PyTorch回退而非模型编写的内核。为弥合这一差距，我们共同发布了Atrex-Kernel-Agent（AKA），这是一种基于性能分析的内核优化代理，结合了迭代测量-修订搜索、用于逃离停滞搜索上下文的优化丢弃，以及分层GPU优化知识库（298个参考内核文件和244个优化知识文档，外加用于API/ISA查找的外部上游参考项目）。在一个受控案例研究中，该代理将零FlyDSL回退转换为实际内核，这些内核达到或超过了手动调整的生产基线。

**方法**：从全集群生产推理轨迹中采样30个算子和440种形状，赋予重要性权重和屋顶线上限；提出AKA代理，结合迭代测量-修订搜索、优化丢弃和分层GPU优化知识库。

**结果**：最佳普通模型仅达~10%屋顶线；AKA代理将零FlyDSL回退转换为达到或超过手动调优基线的内核。

[返回索引](#快速索引)

---

<a id="39"></a>
## 39. [Nova: An End-to-End MLIR Compiler for Deep Learning](http://arxiv.org/abs/2608.00029v3)

- **相关度**：0.93
- **方向标签**：编译器优化
- **收录日期**：2026-07-15
- **arXiv ID**：2608.00029
- **作者**：Adwaid Suresh, Aparna A, Harshini V M, Jona Delcy C A, Killi Uma Maheswara Rao, Ram Charan Golla, Surendra Vendra
- **入选理由**：核心是Nova端到端MLIR编译器，自动合成细粒度kernel、跨算子融合、整图优化，训练GPT-2吞吐达441K tokens/s，优于torch.compile。满足A类：自动编译/代码生成与优化以提升深度学习模型端到端吞吐，并验证性能与数值一致性。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="40"></a>
## 40. [Every Kernel Is a Join: Automatic Multi-GPU Parallelism for AI Computations in Einsummable](http://arxiv.org/abs/2609.03905v1)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-09-03
- **arXiv ID**：2609.03905
- **作者**：Zhimin Ding, Chen-Kuan Liao, Chima Adiole, Brianna Barrow, Fangzhou Du, Yu Hsiao, Ge Huang, Yicheng Jin, Ismail Syed, Chris Jermaine
- **入选理由**：自动将PyTorch式计算分布到多GPU，通过join-agg分解搜索和编译期生成的exchange程序实现通信与聚合，无需人工设备/切分/通信标注，较手调PyTorch和vLLM更快；满足A的自动编译并行化优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="41"></a>
## 41. [Enhancing the Power of Polyhedral-Based Optimizations with Coordinate-Based Hill Climbing](http://arxiv.org/abs/2609.03114v1)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-09-02
- **arXiv ID**：2609.03114
- **作者**：Gaurav Verma, Michael Canesche, Fernando Magno Quintão Pereira
- **入选理由**：在Pluto多面体编译后加入坐标式爬山调优，调整tile size、线程块维度等参数，在x86/ARM和A100上相对原配置获得真实加速并接近AutoTVM；满足A的编译器/kern自动参数优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="42"></a>
## 42. [Semantics-Guided Automatic Tensorization for Multiobjective Evolutionary Algorithms: A Multi-Agent Framework](http://arxiv.org/abs/2609.02387v1)

- **相关度**：0.92
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-09-02
- **arXiv ID**：2609.02387
- **作者**：Zhenyu Liang, Beichen Huang, Bowen Zheng, Ran Cheng
- **入选理由**：用多agent LLM系统把CPU MOEA代码自动tensor化/向量化为GPU实现，保持优化语义并验证正确性，取得可扩展加速，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="43"></a>
## 43. [HyperCut: Fast Inter-Layer Scheduling via Directed Hypergraph and Early Filtering](http://arxiv.org/abs/2608.19296v1)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-08-19
- **arXiv ID**：2608.19296
- **作者**：Ziang Wei, Zirui Xu, Sufeng Guo, Chuanchao Gao, Yiyang Gao, Arvind Easwaran, Yuxiang Fu
- **入选理由**：提出面向DNN编译器的层间调度与资源分配框架HyperCut，通过超图划分早期筛选并耦合格网划分与映射，显著提升性能并减少搜索时间，满足A中编译调度优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="44"></a>
## 44. [ComFuse: Fusing Complex Memory-Intensive Subgraphs with Compute-Intensive Kernels For Modern GPU Architectures](http://arxiv.org/abs/2608.03537v1)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-08-04
- **arXiv ID**：2608.03537
- **作者**：Di Mu, Tengyuan Jin, Zhenkun Wang, Jialin Yang, Yusen Li, Mian Huo, Shusong Guo, Gang Wang, Xiaoguang Liu
- **入选理由**：ComFuse是一个自动GPU编译系统，将计算密集与记忆密集子图融合并自动生成高性能kernel，通过编译变换改善端到端性能和片上数据重用，满足A类编译器/kernel自动优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="45"></a>
## 45. [Rethinking Agentic Kernel Generation for Emerging Accelerators](http://arxiv.org/abs/2608.00894v2)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-08-01
- **arXiv ID**：2608.00894
- **作者**：Ruijie Gao, Jirong Yang, Barry Lyu, Haoran Jin, Nathan Bleier
- **入选理由**：Zomboss在编译器验证边界内用神经agent生成新兴加速器kernel，把机器语义编译为可复用接口，所有56个实例均获得正确验证kernel并取得几何平均加速，满足A类agent+编译器kernel自动优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="46"></a>
## 46. [CONQuER: Hardware-Aware Mixed-Precision Quantisation with Online-Calibrated Surrogates](https://arxiv.org/abs/2607.25884)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-07-30, 2026-07-28
- **arXiv ID**：2607.25884
- **作者**：Aidan Dakhama, Ajitha Rajan
- **入选理由**：将混合精度量化集成到编译器流水线，利用进化搜索自动寻找硬件感知的量化配置，目标是最大推理速度并验证精度，属于通过编译/后端配置自动优化程序执行性能，满足A。

**TL;DR**：CONQuER将混合精度量化集成到编译器流水线中，利用带代理预筛选的进化搜索找到硬件感知的Pareto最优配置，实现高达12.19倍的推理加速且准确率损失极小。

**中文摘要**：在资源受限硬件上部署深度神经网络依赖于混合精度量化（MPQ）。当前的部署工具链严重碎片化了这一过程。量化通常作为前端框架中的硬件无关预处理步骤发生，与生成物理机器代码的下游编译器脱节。这种分离导致配置次优，分配的位宽与目标硬件的异构执行块（如张量核心和可变宽度向量单元）映射不佳，从而产生严重的运行时执行惩罚。此外，通过穷举的硬件在环（HIL）测试评估这些配置由于指数级大的搜索空间而变得不可行。我们提出了CONQuER，一个统一的编译器集成的硬件感知MPQ基础设施。CONQuER将量化转移到编译器流水线中的TOSA级别，从而基于编译器支持实现智能配置处理。为了在实用的编译预算内评估模型不同层的组合搜索空间，CONQuER将NSGA-II进化算法与双代理预筛选引擎相结合。该引擎评估理论缓存内存界限和特征空间各向同性，以丢弃不可行的配置。然后，CONQuER仅通过IREE在硬件上执行最强的候选策略，并将执行指标反馈到在线校准器中。该校准器在NSGA-II进化搜索过程中将代理模型与真实硬件行为对齐。在移动和笔记本电脑CPU以及服务器GPU上的评估表明，最优量化策略是硬件相关的。通过将量化与编译器低化和物理执行耦合，CONQuER发现了Pareto最优配置，推理速度提升高达12.19倍，top-1准确率在未量化基线的1.44%以内。

**方法**：CONQuER将量化移至TOSA级别的编译器流水线，结合NSGA-II进化算法与双代理预筛选引擎（评估缓存界限和特征各向同性），剔除不可行配置，通过IREE在硬件上执行最强候选，并使用在线校准器使代理模型对齐硬件行为。

**结果**：在移动和笔记本电脑CPU及服务器GPU上，CONQuER发现的Pareto最优配置实现推理速度提升高达12.19倍，top-1准确率在未量化基线的1.44%以内。

[返回索引](#快速索引)

---

<a id="47"></a>
## 47. [JAXBench: Benchmarking Autonomous TPU Kernel Optimization](https://arxiv.org/abs/2607.20466)

- **相关度**：0.92
- **方向标签**：Benchmark/评测、Kernel/自动调优
- **收录日期**：2026-07-25, 2026-07-26
- **arXiv ID**：2607.20466
- **作者**：Arya Tschand, Charles Hong, Julian Walker, Nina Cai, Shangkun Wang, Suvinay Subramanian, Sundar Dev, Vijay Janapa Reddi, Amir Yazdanbakhsh, Sethu Sankaran
- **入选理由**：JAXBench提供面向TPU的AI生成内核优化基准与评测工具，包含生产算子与手调Pallas作为专家上界，并评测多种优化方法，满足B。

**TL;DR**：JAXBench是首个TPU原生AI内核优化基准测试套件，包含50个JAX工作负载，实验表明目标特定上下文比模型规模更重要，结合搜索结构可实现显著加速。

**中文摘要**：严格的基准测试通过建立共同的目标来推动自主GPU内核性能优化的进步，但TPU没有类似的基准测试。我们提出JAXBench，一个用于Google Cloud TPU上AI生成内核优化的TPU原生基准测试套件。JAXBench包含50个既相关又具有优化空间的JAX工作负载。我们从公开的MaxText库（如Llama-3.1、DeepSeek-V3、Mixtral、Mamba-2和AlphaFold2）的架构中提取了17个生产级ML算子，并从KernelBench翻译了33个算子，这些算子经过正确性验证，并设置了新的问题大小以实现高TPU v6e MXU利用率。17个生产算子中有8个附带了来自公开Tokamax库的手工优化Pallas内核，并调整了块大小以建立专家上限基线。我们评估了四种反馈驱动方法在JAXBench上生成候选Pallas内核的效果。在整个套件中使用Gemini 3 Flash，我们发现目标特定上下文比模型规模在像Pallas这样文档稀疏的DSL上更重要。基于精选的TPU文档进行条件设置，将每个样本的正确率从5.8%提升到37.3%，并解决了50个基准测试中的48个，几何平均加速比为1.28倍。一旦实现正确性，搜索结构会带来显著收益，Autocomp的波束搜索管道相对于XLA实现了1.36倍的几何平均加速比。在8个手工调优的内核上，Autocomp相对于XLA达到1.60倍的几何平均加速比，恢复了Tokamax上限2.08倍的大部分，但在专门的页面和 ragged 注意力算子方面落后。高质量的TPU内核优化仍然是一项具有挑战性的任务，我们发布JAXBench基准测试、评估工具和基线结果以支持开源贡献。

**方法**：构建JAXBench基准套件，包含50个JAX工作负载（17个来自生产模型，33个来自KernelBench）；评估四种反馈驱动方法（如Autocomp的波束搜索）生成Pallas内核，并对比使用TPU文档条件化的效果。

**结果**：TPU文档条件化将正确率从5.8%提升至37.3%，解决48/50基准，几何平均加速1.28倍；Autocomp达到1.36倍（整体）和1.60倍（手调内核）的几何平均加速比。

[返回索引](#快速索引)

---

<a id="48"></a>
## 48. [Ciphertext- and Polynomial-Level Optimization for Fully Homomorphic Encryption](http://arxiv.org/abs/2607.15750v2)

- **相关度**：0.92
- **方向标签**：编译器优化
- **收录日期**：2026-07-17
- **arXiv ID**：2607.15750
- **作者**：Seongho Kim, Heelim Choi, Jaemin Kim, Seonyoung Cheon, Dongkwan Kim, Jaeho Lee, Hoyun Youm, Dongyoon Lee, Hanjun Kim, Yongwoo Lee
- **入选理由**：核心是FHE编译器Recifhe在ciphertext和polynomial两个级别自动优化FHE程序，消除冗余多项式计算，实现1.25x加速。这是典型的编译器自动优化表现，满足A类，通过修改/编译优化改善运行时间并验证性能。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="49"></a>
## 49. [Breaking Database Lock-in: Agentic Regeneration of High Performance Storage Readers for Database Bypass](http://arxiv.org/abs/2607.07696v1)

- **相关度**：0.92
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-08
- **arXiv ID**：2607.07696
- **作者**：Victor Giannakouris, Immanuel Trummer
- **入选理由**：核心任务是用LLM/智能体自动生成绕过数据库驱动的高性能存储读取代码，目标是提升端到端分析吞吐量，并用TPC-H验证正确性和最高27x加速，满足条件A。

**TL;DR**：Jailbreak利用LLM直接从数据库存储文件中读取数据生成列式缓冲区，绕过数据库引擎，实现高达27倍的分析吞吐量提升。

**中文摘要**：对外部数据库系统中存储的数据进行操作的分析工作负载面临一个根本性的瓶颈：数据访问完全由数据库驱动程序（如JDBC或ODBC）保护，强制所有读取操作通过查询执行和其他非为批量列式分析设计的驱动层进行。我们提出Jailbreak，一种通过直接读取存储文件并物化数据为内存列式缓冲区来完全绕过数据库引擎的方法。Jailbreak的关键洞察在于，数据库文件格式尽管复杂，但其源代码和文档完全规定了这些格式，而大型语言模型（LLM）可以消化这些工件，无需人工设计的解析逻辑就能重新生成特定操作符的表读取组件。Jailbreak利用LLM辅助的代码合成为数据库存储解码，将传统不透明的格式转变为可直接查询的工件。我们在PostgreSQL和MySQL存储文件上评估Jailbreak，针对读取副本和离线处理管道中常见的分析快照场景。生成的读取器产生Apache Arrow缓冲区，可直接被大多数广泛使用的查询引擎使用，包括DuckDB、Apache Spark以及GPU加速框架如cuDF和Spark RAPIDS。我们使用TPC-H基准测试在所有查询结果上验证与基于JDBC/ODBC的基线的正确性，并展示了端到端分析吞吐量的显著性能提升，实现了高达27倍的加速。我们的结果表明，LLM辅助的存储读取器合成是一种可行且可泛化的方法，用于打破跨数据库系统的数据锁定，并且可应用于PostgreSQL和MySQL以外的任何系统，只要其文件格式可通过文档或源代码提供给LLM。

**方法**：提出Jailbreak方法，利用LLM根据数据库文件格式的源代码和文档自动生成读取器，直接读取存储文件并转换为内存列式缓冲区（如Apache Arrow）。

**结果**：在TPC-H基准测试中，Jailbreak正确性验证通过，端到端分析吞吐量提升高达27倍，支持多种查询引擎（如DuckDB、Spark、cuDF）。

[返回索引](#快速索引)

---

<a id="50"></a>
## 50. [QuTuner: Feature- and Learning-Guided Optimization Pass Tuning for Quantum Compilers](http://arxiv.org/abs/2607.04586v1)

- **相关度**：0.92
- **方向标签**：编译器优化、优化策略检索
- **收录日期**：2026-07-06
- **arXiv ID**：2607.04586
- **作者**：Ming Zhong, Xiangyu Ren, Jinglei Cheng, Shaohua Li, Zhiding Liang
- **入选理由**：QuTuner用静态电路特征与优化感知pass嵌入离线检索/排序并精调量子编译器优化pass序列，自动调优编译选项并验证电路指标下降，满足A/优化策略检索，是编译器性能优化。

**TL;DR**：QuTuner通过结合静态电路特征和优化感知嵌入来引导量子编译器优化序列调优，显著提升指标降幅并减少调优时间。

**中文摘要**：量子编译器在将量子电路转换为更低代价、更高执行保真度的实现中起着关键作用。这一过程通常由电路级指标（如门计数和电路深度）引导。尽管编译器优化序列调优在经典编译器中已被广泛研究，但直接将这些技术迁移到量子编译器面临挑战，因为量子程序以电路形式表达，并展现出受量子特定结构塑造的优化行为。先前的量子编译器调优方法已开始使用电路特征来指导优化序列选择，但仍存在两个局限：它们仅搜索优化序列空间的一小部分，并且主要依赖静态特征，这些特征并未明确反映电路对编译器优化的响应。我们提出QuTuner，一种特征引导的量子编译器优化序列调优框架，可跨编译器和调优目标泛化。QuTuner首先构建一个大型优化数据集。然后从两个互补视角表征每个电路：描述电路结构的静态电路特征，以及总结电路对各优化序列响应的优化感知优化序列嵌入。利用这些表示，QuTuner训练两个离线模型来对未见电路检索和排序候选优化序列，随后进行轻量级优化。我们在Qiskit和PyTKET上使用两个基准套件评估QuTuner。在Qiskit上，QuTuner比最强基线最多提升84.85%的评估指标降幅，同时减少73.59%的调优时间。在PyTKET上，它最多提升18.68%的指标降幅，并减少64.49%的调优时间。这些结果表明QuTuner为量子编译器提供了一种有效的自适应优化序列调优方法。

**方法**：构建大型优化数据集，从静态电路特征和优化感知嵌入两个视角表征电路，训练离线模型检索和排序候选优化序列，再轻量级微调。

**结果**：在Qiskit上指标降幅提升84.85%，调优时间减少73.59%；在PyTKET上指标降幅提升18.68%，调优时间减少64.49%。

[返回索引](#快速索引)

---

<a id="51"></a>
## 51. [Understanding Agent-Based Patching of Compiler Missed Optimizations](http://arxiv.org/abs/2607.02370v2)

- **相关度**：0.92
- **方向标签**：编译器优化、Benchmark/评测
- **收录日期**：2026-07-02
- **arXiv ID**：2607.02370
- **作者**：Batu Guan, Zirui Wang, Shaohua Li
- **入选理由**：构造真实LLVM missed optimization基准，研究并利用历史PR检索/蒸馏改进agent生成的编译器优化补丁的泛化范围，属于编译器自动优化与专属评估，满足B/A。

**TL;DR**：本文系统研究了代理修补编译器错过优化的能力，发现代理生成的补丁在优化范围上与开发者补丁存在差异，并提出了历史知识增强技术以改善泛化。

**中文摘要**：编译器错过的优化是指编译器未能优化某些代码的情况。实现或修补这些错过的优化需要许多编译器开发人员的努力。在本文中，我们系统性地研究了代理(agent)修补编译器错过的优化的能力。我们识别了一个重大挑战：修补错过的优化不仅仅需要修复报告的具体案例，还需要泛化到类似案例。我们构建了一个真实世界LLVM错过优化问题的基准，并从优化范围的角度比较了代理生成的补丁与开发人员生成的补丁。我们的结果表明，编码代理经常优化给定的示例，但许多生成的补丁要么只覆盖了开发人员预期范围的一部分，要么与之部分重叠；在某些情况下，它们甚至进一步泛化到参考补丁之外。我们进一步引入了历史知识增强技术，通过检索和蒸馏利用先前的LLVM优化拉取请求，表明这些技术改善了与开发人员对齐的泛化，并在应用于真实世界IR时产生了实际效益。

**方法**：构建了真实世界LLVM错过优化问题的基准，比较代理与开发者补丁的优化范围，并引入基于检索和蒸馏的历史知识增强技术。

**结果**：代理常优化给定示例，但补丁多只部分覆盖开发者意图范围，甚至过度泛化；历史知识增强技术提升了与开发者对齐的泛化能力。

[返回索引](#快速索引)

---

<a id="52"></a>
## 52. [Tensor Seeks Layout: Formalizing Layout Selection for ML Compilers](http://arxiv.org/abs/2608.21555v1)

- **相关度**：0.91
- **方向标签**：编译器优化
- **收录日期**：2026-08-21
- **arXiv ID**：2608.21555
- **作者**：Clemens Eisenhofer, Yuwen Jia, Daniel Kroening, Sergey Pupyrev
- **入选理由**：形式化并求解ML编译器张量布局选择问题，目标是减少算子执行代价与layout转换代价，并在生产编译器上验证执行时间改进，满足A的编译pass优化核心。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="53"></a>
## 53. [Portable to Efficient: Auto-Tuning Hardware-Agnostic GPU Kernels in Julia](http://arxiv.org/abs/2608.21227v1)

- **相关度**：0.91
- **方向标签**：Kernel/自动调优
- **收录日期**：2026-08-21
- **arXiv ID**：2608.21227
- **作者**：Floris-Jan Willemsen, Evelyne Ringoot, Alan Edelman
- **入选理由**：将自动调参（auto-tuning）集成到硬件无关GPU kernel，系统探索kernel配置并提升NVIDIA/AMD/Intel/Apple GPU上的性能（3x-7x），直接满足A的kernel自动优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="54"></a>
## 54. [Validation-Centric AI-Assisted GPU Porting of a 250,000+ Line Legacy Weather Simulation Code](http://arxiv.org/abs/2608.13122v1)

- **相关度**：0.91
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-08-13
- **arXiv ID**：2608.13122
- **作者**：Tetsuya Hoshino, Masaya Kato, Kazuhisa Tsuboki, Daichi Mukunoki, Takahiro Katagiri, Toshihiro Hanawa
- **入选理由**：AI agent辅助将25万行遗留科学Fortran代码GPU化，使用OpenACC变换，以dump数据验证数值一致性并取得5.1x加速，是面向大型程序的自动/半自动性能移植优化，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="55"></a>
## 55. [WarmTuner: Program-Specific Warm Starts for Compiler Autotuning via Offline-to-Online Reinforcement Learning](https://arxiv.org/abs/2607.25831)

- **相关度**：0.91
- **方向标签**：编译器优化
- **收录日期**：2026-07-30, 2026-07-28
- **arXiv ID**：2607.25831
- **作者**：Tianlu Qiao, Mingxuan Zhu, Zeyu Sun, Dan Hao
- **入选理由**：核心是自动搜索编译器优化flag组合以最大化目标程序运行性能，属于编译选项自动优化，有实测速度提升，满足A。

**TL;DR**：WarmTuner是一个离线到在线的强化学习框架，通过程序条件化策略和GRPO优化，在编译器自动调优中平均加速1.732倍，显著优于现有方法。

**中文摘要**：编译器是将高级程序转换为机器代码的基础软件工具。现代编译器提供了数百种优化，每种优化通过优化标志开启或关闭，以提高生成代码的性能。然而，可能的标志组合数量呈指数级增长，使得为给定目标程序找到合适的标志配置变得困难。现有的编译器自动调优技术通过剪枝搜索空间、注入搜索偏差或预测配置性能来降低调优成本。尽管有些技术利用了程序特征，但它们从历史数据中提取的知识在搜索开始后就固定不变；运行时反馈仅指导搜索本身，而从未指导先验知识。因此，当这种先验知识与目标程序不匹配时，这些方法在搜索到达良好配置之前会浪费大量有限的在线预算。我们提出了WarmTuner，一个离线到在线的强化学习框架，它将历史记录转化为程序条件化的策略，该策略预测完整标志空间中每个标志的设置，并在目标程序上保持可适应性。离线阶段，WarmTuner从历史良好配置中学习这个完整标志空间上的程序条件化策略。在线阶段，它使用真实的编译-运行反馈在目标程序上优化同一策略，使得策略由测量的加速比驱动，而不是局限于历史数据。我们使用组相对策略优化（GRPO）来实例化在线更新，该方法比较同一轮中的候选者，并避免单独的价值模型。我们在GCC 15.2.0上使用cBench和PolyBench评估了WarmTuner。结果表明，WarmTuner相比于GCC -O3实现了平均1.732倍的加速比，并在14/30个程序上获得了最佳结果，显著优于对比技术。

**方法**：提出WarmTuner，离线阶段从历史配置学习程序条件化策略，在线阶段使用GRPO根据实际编译-运行反馈优化策略。

**结果**：在GCC 15.2.0上，平均加速比1.732x，在14/30个程序上获得最佳结果。

[返回索引](#快速索引)

---

<a id="56"></a>
## 56. [CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits](https://arxiv.org/abs/2607.20518)

- **相关度**：0.91
- **方向标签**：Benchmark/评测、Kernel/自动调优
- **收录日期**：2026-07-25, 2026-07-26
- **arXiv ID**：2607.20518
- **作者**：Xue-Jian Gao, Deng Pan, Yueming Su, Jiasheng Li, Bin Du, Fengming Zhu, Chengdi Ma, Junyi Fan, Qichen Liao, Chengqiu Hu, Xinxian Chen, Lingchao Zheng, Jun Li, Jiwei Yang, Yuwei Fan
- **入选理由**：CANN Bench是面向华为Ascend NPU的AI算子代码生成基准，提供编译/正确性/性能三维评分与硬件锚定性能上限，满足B。

**TL;DR**：提出了CANN Bench，一个针对华为昇腾NPU的AI生成算子代码的开放基准测试，包含53个算子、1060个测试用例和三维加权复合评分。

**中文摘要**：AI代理现在能够在不同硬件平台上编写、编译和迭代优化底层算子内核。然而，现有的基准测试几乎完全专注于CUDA和Triton，使得编程模型不那么开放的硬件生态系统缺乏共同的评估基准。我们提出了CANN Bench，这是一个针对华为昇腾NPU上AI生成算子代码的开放基准测试。当前版本涵盖53个算子和1060个测试用例，分为四个难度等级——从简单的逐元素原语到MoE分发和FlashAttention内核——涵盖FP16、BF16、FP32和INT8精度格式。评估采用三维加权复合得分，将编译、功能正确性和性能视为独立维度，为内核生成代理提供原则性的奖励信号。性能根据开箱即用的PyTorch-on-Ascend基线和真实NPU硬件上每个案例的分析性硬件锚定性能（HAP）极限进行评分，确保分数反映真正的优化空间而非测量伪影。评估框架从设计上抵制奖励黑客行为。CANN Bench在官方CANN仓库中进行版本管理，并旨在长期社区共建，为昇腾生态系统提供一个定量、可重复且可持续维护的AI算子编写能力衡量标准。

**方法**：设计了一个包含53个算子和1060个测试用例的基准测试，分四个难度等级，覆盖多种精度；采用三维加权复合得分（编译、正确性、性能）进行评估，性能对照PyTorch基线和硬件锚定性能极限。

**结果**：当前版本提供了全面的基准测试，能够有效评估AI代理生成算子代码的能力，并抵抗奖励黑客行为。

[返回索引](#快速索引)

---

<a id="57"></a>
## 57. [Can Coding Agents Implement Missed Compiler Optimizations? Evaluating LLM Agents on LLVM Peephole Optimizations](http://arxiv.org/abs/2607.02684v1)

- **相关度**：0.91
- **方向标签**：Benchmark/评测、编译器优化
- **收录日期**：2026-07-02
- **arXiv ID**：2607.02684
- **作者**：Hongxu Xu, Chunhao Liao, Xintong Zhou, Chengnian Sun
- **入选理由**：PeepholeBench从真实LLVM InstCombine漏优化issue/PR构造任务，衡量agent补丁的正确性与收益性，是专门针对自动编译器优化开发的性能优化基准，满足B。

**TL;DR**：PeepholeBench是一个评估编码智能体修复LLVM编译器遗漏窥孔优化的基准，发现当前智能体在正确性和收益性上均无法匹敌人类开发者，主要失败模式为转换过窄和LLVM机制误用。

**中文摘要**：基于大型语言模型的编码智能体现在能够修补可观的真实世界代码库，但它们能否开发编译器优化仍然是一个开放问题。为了研究这一问题，我们引入了PeepholeBench，这是一个评估框架，其任务来源于针对LLVM的InstCombine pass报告的真实世界遗漏窥孔优化。由于遗漏的窥孔优化通常通过小而局部的补丁修复，它们为编码智能体提供了一个范围明确但要求严格的测试平台：正确的修复需要对程序语义进行严谨推理，并熟悉优化器特定的约定。PeepholeBench的任务源自21个已解决的LLVM问题和19个合并的拉取请求（PR），仅向智能体提供每个修复之前存在的issue上下文，并评估生成的补丁的正确性和收益性。通过PeepholeBench，我们评估了最先进的编码智能体修复LLVM InstCombine pass中遗漏窥孔优化的能力，并将其补丁与对应的人工编写的修复进行比较。我们观察到正确性与收益性之间存在张力，没有智能体能在两个维度上同时匹敌人类开发者。主要的失败模式是过于狭窄的转换以及对LLVM特定机制的误用，这些错误现有的测试套件很少能暴露。这些结果共同确立了PeepholeBench作为编码智能体的一个真实且具有挑战性的基准，并为构建能够更可靠地协助编译器优化开发的智能体指明了未来方向。

**方法**：构建PeepholeBench任务：从21个已解决LLVM问题和19个合并PR中提取真实窥孔优化案例，仅提供修复前的issue上下文，评估智能体生成的补丁的正确性和收益性，并与人类修复对比。

**结果**：无智能体能在正确性和收益性上同时匹敌人类；主要失败模式包括过于狭窄的转换和LLVM特定机制误用（如模式匹配错误），这些错误难以被现有测试套件捕获。

[返回索引](#快速索引)

---

<a id="58"></a>
## 58. [CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling](http://arxiv.org/abs/2609.01864v1)

- **相关度**：0.90
- **方向标签**：Kernel/自动调优
- **收录日期**：2026-09-01
- **arXiv ID**：2609.01864
- **作者**：Zhengxiong Li, Tsung-Wei Huang, Umit Ogras
- **入选理由**：面向GPU分布式共享存储器的代价制导优化框架，通过profiling特征识别DSMEM可获利模式、应用reduction-reuse变换并用成本模型预测收益，在RTX 5090/H100等上一致加速；满足A的GPU kernel自动优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="59"></a>
## 59. [Hierarchical Shared Memory-Aware Optimization for TRSM on GPU Platforms](http://arxiv.org/abs/2608.25469v1)

- **相关度**：0.90
- **方向标签**：Kernel/自动调优
- **收录日期**：2026-08-26
- **arXiv ID**：2608.25469
- **作者**：Xinzhe Chen, Haowei Li, Lijuan Hu, Wenjing Ma, Fangfang Liu
- **入选理由**：针对GPU上TRSM实现进行分层共享内存优化，设计流水线、对角块解耦和基于离线profiling的块大小自动选择，在不同GPU上超过cuBLAS/rocBLAS；满足A的直接GPU kernel性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="60"></a>
## 60. [XRFix: Exploring Performance Bug Repair of Extended Reality Applications with Large Language Models](http://arxiv.org/abs/2608.21718v1)

- **相关度**：0.90
- **方向标签**：LLM/Agent 代码优化、Profiling/程序分析
- **收录日期**：2026-08-22
- **arXiv ID**：2608.21718
- **作者**：Jingwen Wu, Hanyang Guo, Hong-Ning Dai, Xiapu Luo
- **入选理由**：核心是用LLM修复XR应用中的性能bug（渲染/计算低效），属于以性能改善为目标的自动代码修复，并配有检测工具和修复验证，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="61"></a>
## 61. [Can Large Language Models Recover Semantic Optimization Opportunities That Compilers Miss?](http://arxiv.org/abs/2608.03983v1)

- **相关度**：0.90
- **方向标签**：Benchmark/评测
- **收录日期**：2026-08-04
- **arXiv ID**：2608.03983
- **作者**：Hailong Jiang, Feng Yu, Emran Hossain, Jianfeng Zhu, Mengfei Ren, Qiang Guan, Chunwei Xia
- **入选理由**：论文提出SeGaBench可执行benchmark，专门测试LLM从语义层面恢复编译器错过的优化机会，包含正确性、语义验证和可复现加速比协议，属于面向自动程序性能优化的专用benchmark与LLM评估。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="62"></a>
## 62. [KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation](https://arxiv.org/abs/2607.27231)

- **相关度**：0.90
- **方向标签**：Benchmark/评测、Kernel/自动调优
- **收录日期**：2026-08-01, 2026-08-02, 2026-08-03, 2026-07-22
- **arXiv ID**：2607.27231
- **作者**：Peiyu Zang, Jian Tao, Jialing Zhang, Yichen Yuan, Wentao Zhang, Guang Liu, Yonghua Lin
- **入选理由**：提出专门评估LLM/智能体生成Triton内核性能的统一基准，支持多算子来源与多硬件平台，属于针对自动程序优化的benchmark，满足B。

**TL;DR**：提出了KernelGenBench基准，用于评估LLM和智能体生成的Triton内核在多来源算子与多硬件平台上的表现，发现基于智能体的方法更优但存在跨平台性能下降和高token消耗问题。

**中文摘要**：大语言模型（LLM）显著提升了对高效加速器内核的需求，但内核开发仍然是一项高度专业化和劳动密集型的任务。近年来LLM和智能体框架的兴起为自动内核生成提供了一条有前景的路径。然而，尽管进展迅速，目前仍缺乏一个全面的基准来严格评估LLM生成的内核在不同算子来源或异构硬件平台上的表现。我们提出KernelGenBench，一个统一的基准，用于系统评估LLM和智能体生成的Triton内核在不同算子来源和异构硬件平台上的表现。它包含两个互补的子基准：KernelGenBench-MS（多源），评估来自三个来源的210个算子，超越了标准的以PyTorch为中心的任务；以及KernelGenBench-MC（多芯片），使用110个算子的子集衡量跨六个异构硬件平台的性能可移植性。我们的大规模评估消耗了超过150亿个token，结果表明：（1）基于智能体的方法始终优于纯LLM采样方法，而cuBLAS算子在所有方法中都是最具挑战性的；（2）生成性能在不同硬件平台之间差异显著，即使是最近的内核专用智能体也经历了严重的跨平台性能下降（例如，AutoKernel从NVIDIA上的87%降至平台E上的25%）；（3）自主内核生成仍然成本高昂，专用智能体方法每个成功算子平均消耗511万个token（AKO4all达到519万个），比简单的LLM采样方法高出数个数量级。

**方法**：构建KernelGenBench，包含多源（210算子）和多芯片（110算子、6平台）两个子基准，并大规模评估多种LLM和智能体方法。

**结果**：基于智能体的方法优于纯LLM采样；cuBLAS算子最难；跨平台性能下降严重（AutoKernel从87%降至25%）；生成成本高，每算子平均需511万token。

[返回索引](#快速索引)

---

<a id="63"></a>
## 63. [Cross-Model Cross-Language AI Coding Agent Performance: Accuracy and Speed of Parallel CLRS Algorithms](https://arxiv.org/abs/2607.26083)

- **相关度**：0.90
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-31, 2026-07-26
- **arXiv ID**：2607.26083
- **作者**：Shiqi Cheng, Evelyne Ringoot, Rabab Alomairy, Alan Edelman
- **入选理由**：核心评估AI编码智能体将串行代码自动改写为并行实现以取得加速的能力，将性能作为主要指标并有实测加速比较，属于LLM代码性能优化的直接研究，满足A相关。

**TL;DR**：本文评估了AI编码智能体在并行代码生成上的能力，发现它们能生成正确代码但加速效果高度依赖算法和语言，Sonnet 4.6表现最佳，C++在图算法上并行化最一致，Python和Julia在搜索算法上加速最大。

**中文摘要**：AI编码智能体已迅速成为软件工程中无处不在的工具。它们的串行性能，无论是在准确性还是速度方面，都已被广泛覆盖。然而，最近初步结果表明它们的并行编程能力落后于串行编程能力。本文对三种编码智能体——Cursor的Composer 2.0、GPT 5.4和Claude Sonnet 4.6——在三种算法类别——排序、图遍历和搜索——中使用C++、Python和Julia进行并行代码生成进行了跨语言评估。对于每种算法和语言对，我们提示编码智能体从串行基线生成并行实现，追踪实现功能正确性和性能改进所需的提示努力，并测量相对于自定义串行基线和第三方库实现的加速比。我们发现编码智能体能够以适度的提示努力生成正确的并行实现，但实现有意义的加速比高度依赖算法和语言。Sonnet 4.6在整体性能提升方面表现最强，而GPT 5.4尽管始终保持正确性，但未产生可测量的加速比。C++在图算法方面的并行化最一致，而Python和Julia在搜索算法上实现了最大的加速比：没有一种语言在所有类别中占主导地位。Python和Julia在某些图算法上实现了加速，但在其他算法上出现了退化。这些发现强调了将运行时性能效率作为LLM主要性能指标（除了准确性之外）的影响，特别是对于并行实现。

**方法**：跨语言评估三种编码智能体（Composer 2.0、GPT 5.4、Sonnet 4.6）在C++、Python、Julia上生成排序、图遍历、搜索三类算法的并行实现，从串行基线出发，记录提示次数，测量功能正确性和加速比。

**结果**：编码智能体能以较少提示生成正确并行代码，但只有Sonnet 4.6显著提升性能，GPT 5.4无提速；C++在图算法上并行化最一致，Python和Julia在搜索算法上提速最大，但不同语言各有优劣。

[返回索引](#快速索引)

---

<a id="64"></a>
## 64. [Demonstrating GenDB: Instance-Optimized and Customized Query Processing Code Generation via LLM Agents](http://arxiv.org/abs/2607.20630v1)

- **相关度**：0.90
- **方向标签**：LLM/Agent 代码优化
- **收录日期**：2026-07-22
- **arXiv ID**：2607.20630
- **作者**：Jiale Lao, Immanuel Trummer
- **入选理由**：LLM agents自动生成实例级优化的query执行代码，针对特定数据/负载/硬件迭代获得正确且高效实现，并在TPC-H等基准上验证性能优势，符合A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="65"></a>
## 65. [Pattern-Guided Design Space Exploration for FPGA Accelerator Design](http://arxiv.org/abs/2607.15068v1)

- **相关度**：0.90
- **方向标签**：编译器优化、Kernel/自动调优、搜索与进化优化
- **收录日期**：2026-07-16
- **arXiv ID**：2607.15068
- **作者**：Jialiang Zhang, Weiman Yan, Yuelin Zou
- **入选理由**：核心是面向FPGA HLS的pattern-guided design space exploration框架PATTERNDSE，自动探索调度（pipeline、unroll、tiling等）以优化Vitis HLS latency。满足A类：自动修改编译调度/综合选项（HLS kernel）改善性能，并验证功能正确性与延迟。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="66"></a>
## 66. [Rethinking Code Performance Benchmarks for LLMs](http://arxiv.org/abs/2607.07619v1)

- **相关度**：0.90
- **方向标签**：Benchmark/评测
- **收录日期**：2026-07-08
- **arXiv ID**：2607.07619
- **作者**：Nhat Minh Le, Yisen Xu, Zhijie Wang, Tse-Hsun, Chen
- **入选理由**：指出函数级性能基准测试充分性不足，并提出多智能体生成/诊断/修复更能暴露运行时差异的性能测试，改进自动程序优化benchmark的可靠speedup测量方法，满足B。

**TL;DR**：当前函数级性能基准测试因测试充分性不足而无法有效暴露LLM生成代码的性能差异，本文提出多智能体框架生成的性能测试能显著提升检测效果。

**中文摘要**：许多函数级性能基准测试已被提出，用于评估大型语言模型（LLM）是否能生成高效的程序。然而，这些基准测试的结果通常表明，LLM生成的实现与规范解决方案在执行时间上几乎没有差异。在本文中，我们重新审视了四个流行的基准测试：EffiBench、Enamel、EvalPerf和Mercury。我们通过将每个任务运行30次并利用统计检验评估规范解决方案与基准测试提供的高性能实现之间的运行时间差异，在更严格的设置下评估了1,538个任务。使用基准测试提供的测试套件，只有6.11%的高性能实现显著快于规范解决方案。在对308个非显著任务的手动分析中，99个高性能实现不包含有意义的性能变化，而209个包含潜在的性能改进，但这些改进未被原始测试暴露出来。这些结果表明，主要限制不仅在于评估方法，还在于基准测试提供的性能测试的充分性有限。为了解决这一限制，我们提出了一种基于LLM的多智能体框架，用于生成比原始测试更有效地暴露运行时差异的性能导向测试。该框架使用三个独立的智能体来生成、诊断和修复确定性测试，这些测试在保留功能正确性的同时更好地暴露性能差异。在原始测试未发现显著性能差异的1,345个基准任务中，使用DeepSeek-v3.1和GPT-4o的框架生成的测试分别揭示了24.01%和25.43%的任务存在统计显著的改进，优于当前最先进的基于LLM的性能测试生成方法。

**方法**：提出基于LLM的多智能体框架，包含生成、诊断和修复三个智能体，用于创建确定性性能测试，更有效地暴露运行时差异。

**结果**：在1,345个任务中，使用DeepSeek-v3.1和GPT-4o的框架分别使24.01%和25.43%的任务显示出统计显著的性能改进，优于现有方法。

[返回索引](#快速索引)

---

<a id="67"></a>
## 67. [Optimus: A Generic Operator-Level PyTorch Model Transformation Framework](http://arxiv.org/abs/2607.02945v1)

- **相关度**：0.90
- **方向标签**：编译器优化、优化策略检索
- **收录日期**：2026-07-03
- **arXiv ID**：2607.02945
- **作者**：Menglu Yu, Jiaqi Xu, Yuzhen Huang, Yanbo Liang, Jia Liu, Shuai Yang, Jason Ansel, Elias Ellison, Edward Yang, Brian Hirsh, Jia Chen Ren, Will Feng, Oguz Ulgen, Xu Zhao, Daohang Shi, Huaqing Xiong, Quanyu Zhu, Mingming Ding, Junqing Zhou, Ruilin Chen, Yuhang Yang, Chi-Keung Luk
- **入选理由**：核心是Optimus，一个基于模式匹配的PyTorch FX/PT2图变换框架，自动替换模块级模式以加速推理/训练，实现高达63%加速和内存降低，并嵌入PyTorch 2.x编译器栈。满足A类：自动修改计算图/模型代码以改善性能，验证语义保持与性能。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="68"></a>
## 68. [Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?](http://arxiv.org/abs/2607.01211v2)

- **相关度**：0.90
- **方向标签**：Benchmark/评测
- **收录日期**：2026-07-01
- **arXiv ID**：2607.01211
- **作者**：Zhi Chen, Zhensu Sun, Yuling Shi, David Lo, Lingxiao Jiang
- **入选理由**：审计三个仓储级性能优化基准的跨机器可重复性、评分规则与覆盖度，直接研究可靠 speedup/排行榜测量方法，满足B。

**TL;DR**：对GSO、SWE-Perf和SWE-fficiency三个仓库级性能优化基准的审计表明，大多数参考补丁存在跨机器可重复性问题，评分规则导致排名不一致，且大量任务已被公开提交解决，揭示了排行榜分数的局限性。

**中文摘要**：仓库级性能优化基准测试（如GSO、SWE-Perf和SWE-fficiency）通过将补丁应用于真实仓库并对比运行时间与未优化基线和官方参考补丁来评估编码代理。其排行榜分数越来越多地被用作编码代理进展的证据，但这些分数可能混淆运行时不稳定、基准特定评分规则以及有多少任务已被至少一个公开提交解决。我们对这三个基准的这些问题进行了审计。首先，我们在四种常见的Google Cloud机器类型上重放了740个代码优化任务的官方参考补丁。大多数基准任务可以重放，但它们的参考补丁在所有跨机器重放中满足原始基准有效性规则的只有39/102的GSO任务、11/140的SWE-Perf任务和411/498的SWE-fficiency任务；SWE-Perf尤其脆弱，因为许多参考补丁产生的运行时变化几乎为零。其次，我们显示公开提交排名在很大程度上取决于基准评分规则。在GSO和SWE-fficiency共享的八个公开提交中，官方排名在28对提交比较中有9对不一致，而SWE-fficiency的排行榜评分规则将最差的十个任务的分数权重分配过高，达到58.5%-82.8%。第三，观察每个任务的10个公开提交，我们发现至少有1个提交在85.3%（384/450）的可重放GSO和SWE-fficiency任务中达到或超过了参考补丁，在99.8%（449/450）的任务中击败了未优化的基础代码。我们的研究通过识别具有更可靠性能信号的任务、量化每个任务的分数贡献以及揭示被聚合排名隐藏的剩余性能差距，补充了排行榜分数。

**方法**：重放740个任务的官方参考补丁于四种Google Cloud机器上验证有效性；对比八个公开提交的评分规则一致性；分析每个任务多个提交的表现。

**结果**：仅少数参考补丁跨机器有效（GSO 38%，SWE-Perf 8%，SWE-fficiency 83%）；评分规则导致排名分歧；大部分任务已有至少一个提交达到或超过参考补丁（85.3%）。

[返回索引](#快速索引)

---

<a id="69"></a>
## 69. [Integrating a Python Dynamical core into ICON](http://arxiv.org/abs/2608.21150v1)

- **相关度**：0.89
- **方向标签**：编译器优化
- **收录日期**：2026-08-21
- **arXiv ID**：2608.21150
- **作者**：Mauro Bianco, Till Ehrengruber, Enrique González Paredes, Andreas Jocksch, Christos Kotsalos, Ioannis Magkanaris, Philip Müller, Edoardo Paone, Mikael Simberg, Hannes Vogt, Jacopo Canton, Yilu Chen, Anurag Dipankar, Nicoletta Farabullini, Michael Jähn, Matthieu Leclair, Ong Chia Rui, Nathan Beech, Nicolas Gruber, Christoph Müller, Daniel Hupp, Xavier Lapillonne
- **入选理由**：通过GT4Py DSL与DaCe数据流优化将Python动力核集成到气候模拟，替代硬件专用指令并自动生成优化设备代码，获得20-30%性能提升，属于面向性能的编译/代码生成优化，满足A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="70"></a>
## 70. [Memory Allocation for Constant-Bounded Programs](http://arxiv.org/abs/2608.14471v1)

- **相关度**：0.88
- **方向标签**：编译器优化
- **收录日期**：2026-08-14
- **arXiv ID**：2608.14471
- **作者**：Vinícius Silva, Kael Soares, Márcio Costa e Fernando Magno Quintão Pereira
- **入选理由**：为常数界程序设计编译期内存分配策略，部署于eBPF编译器spiller和MLIR静态堆分配，显著降低栈空间和存储开销，满足A中减少内存/代碼大小的编译优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="71"></a>
## 71. [RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks](https://arxiv.org/abs/2608.12004)

- **相关度**：0.88
- **方向标签**：Benchmark/评测
- **收录日期**：2026-08-14, 2026-08-12
- **arXiv ID**：2608.12004
- **作者**：Jinjun Huang, Zhongzhen Wen, Tongtong Xu, Meng Yan, Xin Xia, Zhongxin Liu
- **入选理由**：提出RealisticTritonBench，从真实PR中提取Triton内核生成任务并集成到框架进行端到端评估，直接服务LLM内核生成/优化的benchmark，满足B。

**TL;DR**：提出RealisticTritonBench，一个从真实AI框架PR中提取Triton内核生成任务的基准，用于评估LLM在实际生产场景中的内核生成能力。

**中文摘要**：在现代AI框架中，GPU内核是整体系统性能的关键。结合可用性、可移植性和接近手写CUDA的性能，Triton被广泛用于实现GPU内核。最近的进展表明，大型语言模型（LLM）能够自动生成Triton内核，从而减少对专家内核开发人员的手动工作需求。有几个基准测试评估了LLM生成的Triton内核。然而，它们存在三个关键局限性：（1）它们将任务限制为PyTorch到Triton的转换，未能反映现实世界Triton任务的多样性和复杂性；（2）它们仅评估单个内核的性能，而非端到端性能，而端到端性能是AI框架中实际部署的核心标准；（3）它们依赖手动编写的单个内核评估脚本，这些脚本可能存在缺陷，模型可利用这些缺陷绕过正确性检查并获得虚高的分数。为了解决这些局限性，我们引入了RealisticTritonBench，这是第一个从流行AI框架中的真实拉取请求（PR）派生Triton内核生成任务的基准测试，从而实现逼真的、类似生产的评估。RealisticTritonBench系统地提取修改Triton内核的PR，并将其转化为具有具体工程上下文的生成任务。每个任务以自然语言需求作为输入，要求实现相应的Triton内核，并提供完整且可复现的评估环境。与先前专注于孤立内核性能的基准不同，RealisticTritonBench将生成的内核集成到其原始框架中，并使用端到端测试进行评估，从而实现更真实的评估。我们在RealisticTritonBench上评估了领先的LLM，发现它们仍然难以处理现实世界的Triton内核生成任务。

**方法**：从流行开源AI框架中系统提取修改Triton内核的PR，转化为带自然语言需求和工程上下文的生成任务，并提供完整可复现的评估环境；将生成的内核集成到原框架，用端到端测试评估。

**结果**：在RealisticTritonBench上评估领先LLM，发现它们仍难以处理现实世界的Triton内核生成任务。

[返回索引](#快速索引)

---

<a id="72"></a>
## 72. [GPU Offload in Rust: Portable, Safe, and Fast](http://arxiv.org/abs/2608.13759v1)

- **相关度**：0.88
- **方向标签**：编译器优化、Kernel/自动调优
- **收录日期**：2026-08-13
- **arXiv ID**：2608.13759
- **作者**：Manuel S. Drehwald, Marcelo Domínguez, Kevin Sala, Alán Aspuru-Guzik, Johannes Doerfert
- **入选理由**：在rustc与LLVM后端中实现多厂商GPU offload编译框架，利用所有权与noalias生成高效LLVM IR，性能与手写CUDA/HIP C++相当，属于面向GPU kernel的编译器自动优化，满足A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="73"></a>
## 73. [An eightfold equivalence-preserving speedup of the JUNO OMILREC vertex and energy reconstruction](http://arxiv.org/abs/2608.00461v1)

- **相关度**：0.88
- **方向标签**：LLM/Agent 代码优化、Profiling/程序分析
- **收录日期**：2026-08-01
- **arXiv ID**：2608.00461
- **作者**：Guangbao Sun, Qishan Liu, Wenjie Wu, Jun Cao, Xuefeng Ding, Wenxing Fang, Wuming Luo, Liangjian Wen, Zeyuan Yu, Xiang Zhou
- **入选理由**：以AI coding agent辅助对真实物理重建代码进行分阶段等价保持性能优化，并严格验证bit-identical/阈值一致性和8倍加速，满足A。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="74"></a>
## 74. [Compiling Bioinformatics Recurrences](http://arxiv.org/abs/2607.06225v1)

- **相关度**：0.88
- **方向标签**：编译器优化
- **收录日期**：2026-07-07
- **arXiv ID**：2607.06225
- **作者**：Bala Vinaithirthan, Shiv Sundram, Sneha Goenka, Fredrik Kjolstad
- **入选理由**：提出FILTR，一个生物信息学动态规划recurrence的DSL和编译器，分离recurrence规则与pruning和scheduling策略，编译为C++代码并匹配手调实现性能（0.95x-30x）。满足A类：编译/代码生成自动优化，核心目标是实现高效实现，验证功能与性能。虽领域是生物信息学，但研究产物是编译器/DSL，属于程序性能优化。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="75"></a>
## 75. [EffiHolmes: Differential Profiling-Guided Repository Level Time Inefficiency Fix Localization](https://arxiv.org/abs/2608.03558)

- **相关度**：0.87
- **方向标签**：Profiling/程序分析
- **收录日期**：2026-08-06
- **arXiv ID**：2608.03558
- **作者**：Haowen Yang, Yun Peng, Zishuo Ding
- **入选理由**：核心是针对仓库级时间低效修复定位，使用差异剖析与领域引导LLM推理找出低效热点及修复位置，并构建首个定位基准，属于为自动性能优化提供热点定位与profiling分析的关键子问题，满足C。

**TL;DR**：EffiHolmes是一个基于LLM的仓库级时间低效修复定位框架，通过差异剖析、紧凑执行路径提取和领域引导推理，有效定位性能热点；引入RepoEffi-Bench基准，在140个问题上显著超越现有基线。

**中文摘要**：大型软件系统常常遭受时间低效问题，导致尽管功能正确但执行时间过长。定位其修复位置很困难，因为与功能缺陷不同，它们既不产生测试失败，也不产生堆栈跟踪线索，使得传统和近期基于LLM的故障定位方法不适用。运行时剖析提供了替代证据，但在仓库级设置中面临三个挑战：单次运行剖析无法可靠地区分低效热点与执行噪声；现有剖析器难以从大量后台执行中提取相关执行路径；观察到的热点与实际修复位置之间仍存在语义鸿沟。我们提出EffiHolmes，一个基于LLM的仓库级时间低效修复定位框架。EffiHolmes在默认和缩放工作负载下使用差异剖析来识别低效热点，提取连接这些热点到所报告低效函数的紧凑执行路径，并采用领域引导的LLM推理来定位底层低效逻辑。我们还引入了RepoEffi-Bench，这是第一个仓库级低效定位基准，包含从流行Python仓库收集的140个高质量问题。实验表明，EffiHolmes始终优于最先进的基于检索、基于代理和基于剖析的基线，使用GPT-5.1将文件级Acc@3提高了4.29个百分点，使用qwen3-4b将函数级Acc@5提高了15.00个百分点。它在不同模型容量下也保持稳健。

**方法**：EffiHolmes采用默认与缩放工作负载下的差异剖析识别低效热点，提取连接热点与所报告低效函数的紧凑执行路径，再利用领域引导的LLM推理定位底层低效逻辑；同时构建了包含140个高质量Python仓库问题的RepoEffi-Bench基准。

**结果**：实验显示EffiHolmes一致优于最先进的检索、代理和剖析基线：使用GPT-5.1使文件级Acc@3提升4.29个百分点，使用qwen3-4b使函数级Acc@5提升15.00个百分点，且在不同模型容量下保持稳健。

[返回索引](#快速索引)

---

<a id="76"></a>
## 76. [Gaming Without an Attacker: Benchmark Fingerprinting in LLM-Driven Search Under Selection Pressure](http://arxiv.org/abs/2608.08722v1)

- **相关度**：0.86
- **方向标签**：Benchmark/评测
- **收录日期**：2026-08-09
- **arXiv ID**：2608.08722
- **作者**：Víctor Gallego
- **入选理由**：研究LLM驱动的GPU kernel自动优化系统中的benchmark game/fingerprinting问题，并给出面向自动优化评估的可靠度量设计与失效分类，属于自动程序性能优化测评方法(B)。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="77"></a>
## 77. [TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters](http://arxiv.org/abs/2607.22432v1)

- **相关度**：0.86
- **方向标签**：Profiling/程序分析
- **收录日期**：2026-07-24
- **arXiv ID**：2607.22432
- **作者**：Zhiwen Mo, Yu Cheng, Lei Wang, Zhengju Tang, Lei Xu, Guoyu Li, Yuqi Dong, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Luo Mai, Zhi Yang, Wayne Luk, Hongxiang Fan
- **入选理由**：TileSight是面向tile级GPU kernel的分析性性能模型/剖析工具，可解释compute-memory重叠、cache命中率和跨节点通信，并用于tile配置选择；为自动GPU kernel优化直接提供性能建模和剖析支持，满足C类关键子问题。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="78"></a>
## 78. [Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap in Modern Domain-Specific Languages](http://arxiv.org/abs/2607.04454v3)

- **相关度**：0.86
- **方向标签**：Profiling/程序分析、Kernel/自动调优
- **收录日期**：2026-07-05
- **arXiv ID**：2607.04454
- **作者**：Tingxi Li, Ravishka Rathnasuriya, Wei Yang
- **入选理由**：研究GPU DSL/Triton/TileLang内核的correctness-performance gap，提出库相对效率与roofline利用率两个轻量筛选/诊断准则，直接服务内核代码生成与调优的性能分析，属C。

**TL;DR**：基于正确性的评估可能遗漏性能极差的内核，两种轻量级检查（库相对效率与屋顶线利用率）能有效筛选出功能正确但低效的内核。

**中文摘要**：现代GPU领域特定语言（DSL），如Triton和TileLang，越来越多地用于实现专门的深度学习内核，并作为自动内核生成系统的目标语言。现有的DSL内核评估通过基于参考的数值验证来建立正确性——这是必要的，但未能说明替换质量：一个功能上有效的内核可能远低于其旨在替换的优化库算子的吞吐量。我们使用来自五个算子类别的22个Triton和TileLang内核，在NVIDIA A100和GH200 GPU上研究这种正确性-性能差距，询问基于正确性的评估是否能识别出不适合作为库替换的内核，为什么会发生这种失败，以及如何在没有全面基准测试覆盖的情况下检测它们。该研究得出三个结果。第一，基于正确性的评估可能容忍严重的速度下降：一个惯用的TileLang LayerNorm内核通过了KernelBench的正确性检查，但运行速度比PyTorch基线慢300倍以上。第二，不同内核家族的原因不同。TileLang归一化和归约的减速主要是可修复的编写缺陷，如串行归约和不必要的数据类型转换，而卷积和大型通用矩阵乘法（GEMM）在优化后仍存在残留差距，原因是代码生成和自动调优覆盖限制；供应商库算法选择贡献很小。第三，两个轻量级检查——库相对效率和屋顶线利用率——是互补的筛选标准：它们一起标记了我们集合中每一个功能有效但效率低下的内核，并将可修复的编写缺陷与结构性残留区分开来。

**方法**：使用22个Triton和TileLang内核（五个算子类别）在NVIDIA A100和GH200 GPU上实验，分析正确性-性能差距，并评估库相对效率与屋顶线利用率两种检查方法的有效性。

**结果**：1. 正确性评估可容忍超过300倍的减速；2. 减速原因因内核家族而异：部分可修复（如串行归约），部分为结构性残留（如代码生成限制）；3. 两种轻量级检查能互补地标记所有低效内核。

[返回索引](#快速索引)

---

<a id="79"></a>
## 79. [The Unseen Delta: Characterizing the Compiler Optimization Landscape via Top-Down Differential Analysis](http://arxiv.org/abs/2608.09530v1)

- **相关度**：0.85
- **方向标签**：Profiling/程序分析、优化策略检索
- **收录日期**：2026-08-10
- **arXiv ID**：2608.09530
- **作者**：Zhibo Liu, Huaijin Wang, Shuai Wang
- **入选理由**：提出自上而下的差分性能分析方法，用多层微架构指标定位编译器性能差异的根因，并开发二进制补丁框架移植更优代码序列，属于面向编译器优化缺陷的诊断与优化策略挖掘，满足C。

**TL;DR**：

**中文摘要**：

**方法**：

**结果**：

[返回索引](#快速索引)

---

<a id="80"></a>
## 80. [What Do AI Agents Actually Change? An Empirical Taxonomy of Mutation Patterns in Performance-Improving Pull Requests](http://arxiv.org/abs/2607.05666v1)

- **相关度**：0.85
- **方向标签**：搜索与进化优化、优化策略检索
- **收录日期**：2026-07-06
- **arXiv ID**：2607.05666
- **作者**：Illia Dovhoshliubnyi, Nima Soroush, Ashkan Sami, Alexander Brownlee
- **入选理由**：对216个AI代理性能改进PR的1254个diff hunks建立突变分类学，量化代理身份与优化策略对应的算子空间，为遗传改进等自动代码性能优化挖掘可复用突变模式，属C。

**TL;DR**：本文分析AI编码代理的性能优化PR，发现其突变操作符分布与传统遗传改进语料库显著不同，且代理身份和目标策略可缩小SBSE算子空间。

**中文摘要**：AI编码代理是黑盒：我们无法检查它们如何生成代码，但可以检查它们更改了什么。这一区别对于基于搜索的软件工程（SBSE）非常重要，其中诸如遗传改进（在我们研究的性能优化应用中）等技术依赖于反映代码实际转换方式的突变算子。在AIDev-pop中，33,596个代理PR中不到1%针对性能，使得每个案例成为进入原本不透明的代理行为的罕见窗口。我们使用双LLM交集管道，将来自216个这些PR的1,254个与性能相关的差异块（跨越五个代理系统）与Even-Mendoza等人（2025）的18类语法突变分类法进行分类。三个类别占主导地位：名称修改（37.0%）、对象创建（26.4%）和类型更改（22.7%），这一分布与之前的遗传改进语料库显著不同，后者中无变化占84%。每个代理部署的系统承诺一种独特的突变词汇，每个性能策略激活一个大多不相交的类别子集。因此，代理身份和目标策略是有信息量的先验，缩小了有效的SBSE算子空间。复制包：https://github.com/5uper6rain/ssbse-challenge-2026

**方法**：从AIDev-pop数据集中的33,596个PR中筛选出216个性能相关PR，提取1,254个差异块，使用Even-Mendoza等人（2025）的18类语法突变分类法，通过双LLM交集管道进行分类。

**结果**：三个类别占主导：名称修改（37.0%）、对象创建（26.4%）、类型更改（22.7%），与之前GI语料库（无变化占84%）显著不同；每个代理有独特突变词汇，性能策略激活几乎不相交的类别子集。

[返回索引](#快速索引)

---

<a id="81"></a>
## 81. [EvoMem: Memory-Augmented Evolution for Code Optimization](https://arxiv.org/abs/2608.10795)

- **相关度**：0.82
- **方向标签**：优化策略检索、搜索与进化优化
- **收录日期**：2026-08-13, 2026-08-11
- **arXiv ID**：2608.10795
- **作者**：Viktor Volkov, Valentin Khrulkov, Andrey V. Galichin, Danil Sivtsov, Nikita Glazkov, Olga Volkova, Konstantin Pchelin, Iaroslav Bespalov, Dmitry V. Dylov, Petr Anokhin, Ivan Oseledets
- **入选理由**：核心是LLM驱动的进化搜索中加入持久记忆以复用变异知识，覆盖GPU内核优化等任务，可作为自动程序优化策略挖掘/检索的基础设施，满足C的直接关键子问题。

**TL;DR**：EvoMem为LLM驱动的进化程序搜索引入持久记忆架构，通过捕获和复用成功变异策略，在多个基准上带来平均性能或搜索速度的提升。

**中文摘要**：成功的变异策略在进化代码搜索中可能包含超越单次运行的可复用知识，并且在某些情况下可以跨相关任务和领域迁移。然而，现有的基于LLM的进化框架大多丢弃此类知识，反复重新发现相似的想法，限制了跨运行和跨任务学习的机会。我们引入了EvoMem，一种用于基于LLM的进化程序搜索的持久记忆架构，用于捕获和复用候选变异知识。EvoMem将成功的变异事件转换为结构化的、任务感知的建议，供未来运行使用。它分两个阶段运作：每次运行后，提取并存储具有来源的有前景的想法；在后续进化过程中，基于当前任务和程序上下文检索一小部分相关指令以指导变异。在几何优化、多跳问答、GPU内核优化及相关基准测试中，我们的实验表明，在大多数评估设置下，目标指标或搜索速度均有正向平均改进，同时也揭示了任务间的差异性。总体而言，EvoMem提供了证据表明，持久记忆可以减少部分冗余探索，并改善LLM驱动的进化搜索中成功策略的复用与适应。

**方法**：EvoMem包含两个阶段：运行后提取成功变异事件并存储为带来源的结构化建议；后续进化中根据当前任务和程序上下文检索少量相关指令来引导变异。

**结果**：在几何优化、多跳问答、GPU内核优化及相关基准上，大部分设置的目标指标或搜索速度获得正平均改进，但不同任务间存在差异。

[返回索引](#快速索引)

---
