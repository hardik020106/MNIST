# PyTorch MNIST Digit Classifier

An industrial-grade implementation of a fully connected neural network (Multi-Layer Perceptron) designed to classify handwritten digits using the MNIST dataset. This repository serves as a production baseline configuration for deep learning workflows, leveraging PyTorch for network architecture definition, optimization, and hardware acceleration.

---

## 🚀 Architectural Overview

The architecture follows a standard Multi-Layer Perceptron (MLP) pattern optimized for tabularized spatial data. The model treats every pixel of the input image as an independent feature.



### Network Parameters
- **Input Dimensions:** $28 \times 28 = 784$ features (Flattened spatial layout)
- **Hidden Layer 1:** 128 nodes with Rectified Linear Unit (ReLU) activation
- **Hidden Layer 2:** 64 nodes with Rectified Linear Unit (ReLU) activation
- **Output Layer:** 10 nodes (representing raw logits for classes 0–9)

---

## 🛠️ Pipeline Features

- **Dynamic Hardware Targeting:** Automated device fallback routine (`CUDA` $\rightarrow$ `CPU`) ensuring seamless GPU execution when available.
- **Standardized Ingestion Pipeline:** Integrates `torchvision.transforms` for deterministic tensor scaling ($[0.0, 1.0]$ normalization).
- **Mini-Batch Processing:** Efficient memory footprint management via `DataLoader` partitioning data into groups of 64 with stochastic shuffling on training samples.
- **Serialization Handling:** Decoupled structural saving via `state_dict` mapping for industry-standard production handoffs and inference deployments.

---

## 💻 Code Breakdown

### 1. Data Ingestion & Scaling
The pipeline leverages the native MNIST ingestion class, scaling structural values uniformly to prevent mathematical instabilities during gradient descent:
transform = transforms.ToTensor()
train_data = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

### 2. Network Topology
Inherits directly from nn.Module. Non-linearity is enforced sequentially via nn.ReLU to ensure the mathematical system maps complex decision boundaries:
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.net(x)

### 3. Optimization Strategy
Loss Function: nn.CrossEntropyLoss (Applies implicit Softmax and calculates categorical Log Loss).
Optimizer: optim.Adam (Employs adaptive learning rates to adjust system weights safely based on accumulated momentum).

### 4.Execution Mechanics (Training Phase)
During each training epoch, data is passed through the standard four-step execution loop:
Clear Gradients: optimizer.zero_grad() prevents corrupt cross-batch step accumulations.
Forward Pass: Model evaluates the flattened image batch.
Backward Pass: loss.backward() applies auto-differentiation across the computation graph.
Weight Updates: optimizer.step() shifts parameters relative to gradients.

### 5.Model Serialization
Upon completion, the system isolates the weight matrix states from the structural architecture, saving the weights natively to a deployment-ready asset:
torch.save(model.state_dict(), 'digit_classifier.pth')

### 6.Getting Started
Prerequisites
Ensure your workspace includes the following core deep learning dependencies:
pip install torch torchvision matplotlib

Project Structure

├── mnist_project.ipynb     # Main execution script

├── data/                   # Automatically generated dataset folder

└── digit_classifier.pth    # Serialized model weights (Generated post-training)

## Author

**Hardik Sikka**

## License

This project is open-source and available under the MIT License.
