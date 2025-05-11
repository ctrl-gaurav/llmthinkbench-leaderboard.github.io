# LLMThinkBench 🏆

<div align="center">
  <img src="https://raw.githubusercontent.com/ctrl-gaurav/LLMThinkBench/main/logo/LLMThinkBench_logo_small.png" alt="LLMThinkBench Logo" width="200">
  
  <p>A comprehensive framework for evaluating reasoning capabilities of Large Language Models</p>

  <a href="https://github.com/ctrl-gaurav/LLMThinkBench/stargazers"><img src="https://img.shields.io/github/stars/ctrl-gaurav/LLMThinkBench" alt="Stars Badge"/></a>
  <a href="https://github.com/ctrl-gaurav/LLMThinkBench/network/members"><img src="https://img.shields.io/github/forks/ctrl-gaurav/LLMThinkBench" alt="Forks Badge"/></a>
  <a href="https://github.com/ctrl-gaurav/LLMThinkBench/issues"><img src="https://img.shields.io/github/issues/ctrl-gaurav/LLMThinkBench" alt="Issues Badge"/></a>
  <a href="https://github.com/ctrl-gaurav/LLMThinkBench/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ctrl-gaurav/LLMThinkBench" alt="License Badge"/></a>
  <a href="https://pypi.org/project/llmthinkbench/"><img src="https://img.shields.io/pypi/v/llmthinkbench" alt="PyPI"/></a>
</div>

## 🔍 Overview

LLMThinkBench is a rigorous evaluation framework designed to measure and compare the reasoning abilities of Large Language Models (LLMs). Moving beyond traditional benchmarks, LLMThinkBench specifically targets mathematical reasoning, instruction following, and computational efficiency through a comprehensive suite of tests.

The framework enables researchers, developers, and practitioners to:
- Conduct standardized assessments of LLM reasoning capabilities
- Compare model performance across multiple dimensions
- Gain insights into the strengths and limitations of different models
- Track the evolution of reasoning abilities in the latest LLM developments

## ✨ Features

- **Comprehensive Assessment**: Evaluates basic math reasoning, instruction adherence, and token efficiency
- **Varied Complexity Levels**: Tests with progressively larger input sizes (8, 16, 32, 64 elements)
- **Interactive Dashboard**: Visualize and compare model performance with customizable plots
- **Standardized Metrics**: Consistent evaluation approach across models
- **Easy-to-Use API**: Simple integration for evaluating your own models
- **Extensible Framework**: Add new tasks and metrics as needed

## 🔧 Installation

### Via PyPI

```bash
pip install llmthinkbench
```

### From Source

```bash
git clone https://github.com/ctrl-gaurav/LLMThinkBench.git
cd LLMThinkBench
pip install -e .
```

## 🚀 Quick Start

```python
from llmthinkbench import ThinkBench
from llmthinkbench.models import OpenAIModel

# Initialize a model to evaluate
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Initialize the benchmark
benchmark = ThinkBench(model=model)

# Run evaluations
results = benchmark.evaluate(tasks=["basic_math", "instruction_following"])

# Save results
benchmark.save_results("my_model_results.csv")
```

## 📊 Interactive Dashboard

LLMThinkBench includes an interactive Streamlit dashboard for visualizing benchmark results:

```bash
# Navigate to the repository directory
cd LLMThinkBench

# Run the dashboard
streamlit run main.py
```

The dashboard provides:
- Key Performance Indicators for quick model comparison
- Interactive Plotting Arena to create customized visualizations
- Detailed Benchmark Results for in-depth analysis

## 📚 Benchmark Suite

LLMThinkBench evaluates models across multiple dimensions:

### Basic Mathematical Reasoning
- Arithmetic operations (addition, subtraction, multiplication, division)
- Statistical operations (mean, median, mode)
- Sequence operations (sorting, finding min/max values)
- Counting operations (even/odd counts)

### Instruction Following
- Ability to adhere to explicit instructions
- Format compliance
- Constraint adherence

### Computational Efficiency
- Output token usage
- Response length
- Response time

## 🏆 Leaderboard

Visit our [live leaderboard](https://llmthinkbench.example.com) to see up-to-date model rankings across all benchmarks.

Current top performers:

| Rank | Model | Math Reasoning | Instruction Following | Output Efficiency |
|------|-------|----------------|------------------------|-------------------|
| 1    | Model A | 95.2% | 97.8% | 32.4 |
| 2    | Model B | 93.7% | 96.5% | 45.1 |
| 3    | Model C | 88.9% | 94.2% | 38.7 |

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) for details on how to:
- Add new benchmark tasks
- Improve existing evaluation metrics
- Add model adapters
- Fix bugs and enhance features

## 📖 Research

If you're using LLMThinkBench for research, please see our [research documentation](docs/RESEARCH.md) for information on:
- Evaluation methodology
- Benchmark design principles
- Comparison with other benchmarks
- Interpretation of results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
