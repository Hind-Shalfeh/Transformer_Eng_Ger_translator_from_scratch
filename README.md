# Transformer_Eng_Ger_translator_from_scratch

This is an implementation of a Transformer model that closely follows the original paper, ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The model includes self-attention, cross-attention, residual connections, and padding/future token masks. The task is to translate English sentences into German.

---

## Table of Contents
1. [Implementation](#implementation)
2. [Installation](#installation)

---

## Implementation

While this is an implementation "from scratch," PyTorch’s built-in modules are used where possible for optimization and reliability. Here’s a breakdown of the implementation:

### **Built-in PyTorch Modules Used**
- **`nn.Linear`**: For linear transformations.
- **`nn.LayerNorm`**: For layer normalization.
- **`nn.Dropout`**: For dropout regularization.
- **`nn.Embedding`**: For input embeddings.

### **Custom Components**
- **Positional Encoding**:
  - A custom class using sine and cosine functions to compute positional encodings.
- **Feedforward Block**:
  - A custom `PositionWiseFeedForward` class with a two-layer feedforward network and ReLU activation.
- **Multi-Head Attention**:
  - A custom `MultiHeadAttention` class implementing scaled dot-product attention. Heads are split and combined using `reshape` and `transpose`.
- **Residual Connections**:
  - Directly implemented in the encoder and decoder layers.
- **Encoder and Decoder Layers**:
  - Custom `EncoderLayer` and `DecoderLayer` classes combining self-attention, cross-attention (for the decoder), feedforward networks, and residual connections. Layer normalization and dropout are applied using `nn.LayerNorm` and `nn.Dropout`.
- **Transformer Model**:
  - A custom `Transformer` class that combines the encoder, decoder, embeddings, positional encoding, and final linear layer. Mask generation is implemented for source and target sequences.

### **Softmax Handling**
- During training, the **softmax is not explicitly applied** to the final layer of the decoder because `nn.CrossEntropyLoss` internally applies it to the logits before computing the loss. This improves efficiency.
- During **inference**, the softmax is applied to the logits to compute probabilities.
- The softmax is also applied in the **attention mechanism** to compute attention weights.

---

## Installation

The main Colab Notebook is self-contained and includes:
1. Downloading requirements.
2. Loading and preprocessing the data.
3. Training the model.
4. Testing and inference.

### **Steps**
1. Open the Notebook in Google Colab.
2. Run all cells sequentially.

---

## References
1. ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
2. [Machine Translation](https://github.com/bharadwaj1098/Machine_Translation/blob/main/README.md)
3. [transformer from scratch](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)
4. [pytorch-transformer](https://github.com/hkproj/pytorch-transformer/blob/main/README.md)
5. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

###### This code is based on the code transformer-from-scratch, Copyright (c) 2023 Shubham Prasad.
