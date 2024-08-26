# PrivacyBot: Advanced LLM-Powered RAG Techniques to Enhance User Privacy Literacy
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Logo_Université_de_Lausanne.svg/1280px-Logo_Université_de_Lausanne.svg.png" width="250" height="100"/> <br>
 </p>

 **By** : Feryel Ben Rhaiem


## 🧐 Overview
PrivacyBot is a state-of-the-art system designed to explore and evaluate different Retrieval-Augmented Generation (RAG) techniques. It uses LlamaIndex as its core framework to optimize interactions between large language models (LLMs) and data. The project focuses on enhancing user privacy literacy through conversational AI. This repository includes the implementation of various RAG pipelines, from a basic Naive RAG model to more advanced techniques like Sentence Window Retrieval, Auto-Merging Retrieval, Re-ranking, and Prompt Engineering.

## 💡 Features
* **Naive RAG Implementation** : A baseline model that combines retrieval and generation.
* **Advanced RAG Techniques** : Includes Sentence Window Retrieval, Auto-Merging Retrieval, Re-ranking, and Prompt Engineering.
* **TruLens Evaluation** : Integration with TruLens framework for detailed evaluation metrics.
* **Configurable Pipelines** : Easily switch between different RAG pipelines.

## ⚙️ Setup
### Requirements
* Python 3.8 or above
* OpenAI API Key for LLM integration
* Required Python libraries are listed in requirements.txt.

### Installation
1. Clone the repository:
```console
git clone https://github.com/Feryel-Ben-Rhaiem/privacybot.git
cd privacybot
```
2.  Install the dependencies:
```console
pip install -r requirements.txt
```
3. Configure your API key and models in config.yaml.

## 📁 Folder Structure
* **src** : Contains the implementation and evaluation of all RAG pipelines in the files ragpipelines.py, app.ipynb, and evaluation.ipynb.
* **files** : Contains the files needed for the training and evaluation of the RAG pipelines, namely; privacy-pmi.pdf, pmi_privacy_policy_qa.txt, and eval_questions.xlsx.
* **config.yaml** : Configuration file for setting up API keys and models.
* **requirements.txt** : List of required Python libraries.

## 👩‍💻 Usage
1. **Running the Pipelines** :
   * You can run the RAG pipelines using the notebook app.ipynb and the Python file ragpipelines.py.
   * Select the desired pipeline and run queries to get responses.

2. **Evaluating Pipelines** :
   * The evaluation is conducted through the evaluation.ipynb notebook.
   * Metrics such as Answer Relevance, Context Relevance, Groundedness, and Answer Correctness are computed using TruLens.

## 🤖 RAG Pipelines & Enhancing Techniques
* **Naive RAG Pipeline**
* **Sentence Window Retrieval**
* **Re-ranking**
* **Prompt Engineering**

## 🎯 Evaluation and Results
The evaluation is conducted using TruLens, a feedback-driven evaluation framework. The results highlight the strengths and weaknesses of each pipeline, focusing on the following metrics; Answer Correctness, Context Relevance, Groundedness, and Answer Relevance.

## 📱 UI



![alt text](https://github.com/Feryel-Ben-Rhaiem/privacybot/files/PrivacyBot - Display 1.png)

