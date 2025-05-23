# Conn-Trans: Formal Architecture Specification

## Abstract

Conn-Trans is a novel transformer architecture that performs iterative semantic reasoning through learnable connections between fixed intermediate representation (IR) nodes and dynamic activation states.

---

## 1. Mathematical Formulation

### 1.1 Core Components

**Fixed IR Nodes (Base Knowledge)**

```
H ∈ ℝ^(N_ir × d)
```

- Fixed throughout training and inference
- Represents fundamental semantic concepts
- Initialized once, never updated

**Dynamic Activation State**

```
X^(t) ∈ ℝ^(B × N_ir × d)
```

- Input-dependent, dynamically updated
- Represents current reasoning state
- Updated through iterative reasoning process

**Connection Matrix**

```
C ∈ ℝ^(N_ir × N_ir)
```

- Primary learnable parameter for reasoning
- Encodes semantic relationships between IR nodes
- Shared across all reasoning steps

### 1.2 Forward Pass

**Input Processing**

```
X^(0) = Attention(Q_in, K_H, V_in)
```

where:

- Q_in = InputTokens × W_q^in
- K_H = H × W_k^H
- V_in = InputTokens × W_v^in

**Iterative Reasoning Process**

```
For t = 1, 2, ..., k:
    X^(t) = (C ⊗ H) + (I + C) ⊗ X^(t-1)
```

where ⊗ denotes matrix multiplication and I is the identity matrix.

**Output Generation**

```
H_eff = H + X^(k)
Output = Attention(Q_eff, K_out, V_out)
```

where:

- Q_eff = H_eff × W_q^out
- K_out = OutputTokens × W_k^out
- V_out = OutputTokens × W_v^out

### 1.3 Affine Transformation Analysis

The core update can be decomposed as:

```
X^(t) = Translation + LinearTransform
      = (C ⊗ H) + (I + C) ⊗ X^(t-1)
      = KnowledgeInjection + StateEvolution
```

This enables:

1. **Knowledge Injection**: C ⊗ H provides consistent semantic context
2. **State Preservation**: I ⊗ X^(t-1) maintains previous reasoning state
3. **Associative Propagation**: C ⊗ X^(t-1) spreads activation patterns

---

## 2. Architecture Variants

### 2.1 Base Architecture (Pure Connection)

```
Layer: X^(t) = (C ⊗ H) + (I + C) ⊗ X^(t-1)
Parameters: C ∈ ℝ^(N_ir × N_ir), Attention projections
```

### 2.2 With Feed-Forward Networks

**Variant A: FFN in Reasoning Loop**

```
For t = 1, 2, ..., k:
    X_temp = (C ⊗ H) + (I + C) ⊗ X^(t-1)
    X^(t) = X_temp + FFN(X_temp)
```

**Variant B: FFN in Input/Output**

```
X^(0) = X^(0) + FFN_in(X^(0))
For t = 1, 2, ..., k:
    X^(t) = (C ⊗ H) + (I + C) ⊗ X^(t-1)
Output = Output + FFN_out(Output)
```

**Variant C: Both**
Combination of Variant A and B.

---

## 3. Parameter Analysis

### 3.1 Parameter Count

**Base Conn-Trans:**

- Connection Matrix: N_ir²
- Input Attention: 3d²
- Output Attention: 3d²
- **Total: N_ir² + 6d²**

**Standard Transformer (L layers):**

- Attention per layer: 4d²
- FFN per layer: 8d² (assuming FFN_dim = 4d)
- **Total: L × 12d²**

**Efficiency Condition:**

```
N_ir² + 6d² < L × 12d²
```

For N_ir = 2d and L = k:

```
4d² + 6d² < k × 12d²
10d² < 12kd²
k > 10/12 ≈ 0.83
```

### 3.2 Computational Complexity

**Per Reasoning Step:**

- Connection computation: O(N_ir² × d)
- Batch processing: O(B × N_ir × d)
- **Total per step: O(B × N_ir × d + N_ir² × d)**

**Total Forward Pass:**

```
O(k × (B × N_ir × d + N_ir² × d) + B × seq_len × d)
```

---

## 4. Theoretical Properties

### 4.1 Convergence Analysis

The system X^(t) = (C ⊗ H) + (I + C) ⊗ X^(t-1) can be written as:

```
H_eff^(t) = (I + C) ⊗ H_eff^(t-1)
```

where H_eff^(t) = H + X^(t).

**Convergence Conditions:**

- Spectral radius ρ(I + C) determines stability
- For convergence: ρ(I + C) ≤ 1
- Eigenvalues of C should satisfy: λ_i ∈ [-2, 0]

### 4.2 Expressiveness

**Linear Expressiveness:**
The k-step reasoning can express any linear transformation of the form:

```
H_eff^(k) = Σ(i=0 to k) (I + C)^i ⊗ (C ⊗ H) + (I + C)^k ⊗ H
```

**With FFN:**
Addition of FFN introduces non-linearity, enabling more complex reasoning patterns.

---

## 5. Implementation Specification

### 5.1 Hyperparameters

**Architecture Parameters:**

- N_ir: Number of IR nodes (recommended: 2d)
- d: Model dimension
- k: Number of reasoning steps (recommended: 3-5)
- FFN_dim: FFN hidden dimension (if used)

**Training Parameters:**

- Learning rate: 1e-4 to 1e-3
- C initialization: Normal(0, 0.01)
- H initialization: Normal(0, 1) or pre-trained embeddings
- Gradient clipping: max_norm = 1.0

### 5.2 Regularization

**Connection Matrix Regularization:**

```
L_reg = λ_spec × spectral_norm(C) + λ_frob × ||C||_F
```

**Stability Enforcement:**

```
C = C - α × diag(C)  # Suppress diagonal elements
```

---

## 6. Experimental Protocol

### 6.1 Ablation Studies

1. **Core Mechanism**: Pure vs Standard Transformer
2. **Knowledge Injection**: With/without C ⊗ H term
3. **Reasoning Depth**: k = 1, 2, 3, 4, 5
4. **FFN Placement**: Pure, Input/Output FFN, Reasoning FFN, Both
5. **IR Node Count**: N_ir = d, 2d, 4d

### 6.2 Benchmark Tasks

**Reasoning Tasks:**

- Logical inference (syllogisms)
- Multi-hop question answering
- Commonsense reasoning

**Analysis Tasks:**

- Connection matrix visualization
- Reasoning state evolution tracking
- Convergence analysis

---

## 7. Expected Contributions

1. **Novel Architecture**: Structured reasoning through learnable connections
2. **Parameter Efficiency**: Fewer parameters for equivalent reasoning depth
3. **Interpretability**: Analyzable reasoning process through X^(t) evolution
4. **Extensibility**: Adaptable reasoning depth without parameter growth

---

## Notation Summary

| Symbol | Meaning                    | Dimensions   |
| ------ | -------------------------- | ------------ |
| H      | Fixed IR nodes             | N_ir × d     |
| X^(t)  | Activation state at step t | B × N_ir × d |
| C      | Connection matrix          | N_ir × N_ir  |
| k      | Number of reasoning steps  | scalar       |
| N_ir   | Number of IR nodes         | scalar       |
| d      | Model dimension            | scalar       |
| B      | Batch size                 | scalar       |
