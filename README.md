# Desconstrução de Títulos: Identificação de Papéis Latentes 
através de Clustering de Competências.

Esta pesquisa 
contribui para a Inteligência do Mercado de Trabalho (LMI) ao fornecer um 
framework para identificação dinâmica de papéis, oferecendo insights para a 
atualização de currículos acadêmicos e estratégias de recrutamento industrial

## 📋 Sobre o Projeto
O objetivo central deste estudo é mapear como as tecnologias se organizam em 
ecossistemas interdependentes e identificar perfis profissionais ocultos sob 
nomenclaturas genéricas. Através dessa análise, busca-se oferecer uma visão mais 
granular e precisa sobre o mercado de Engenharia de Software, construindo para a área 
de Inteligência do Mercado de Trabalho. 

## 🚀 Como Executar

Este projeto foi desenvolvido para ser **reprodutível**. O download dos dados (aprox. 500MB) é feito automaticamente via API.

### 1. Pré-requisitos
* Python 3.10 ou superior.
* Uma conexão com a internet para o primeiro download do dataset.

### 2. Instalação
Clone o repositório e instale as dependências necessárias:
```bash
git clone https://github.com/santoskarolina/projeto-tech-jobs

cd projeto-tech-jobs

pip install pandas matplotlib kagglehub sentence-transformers torch
```

### 2. Execução
Tems três scriots

```bash
python src/levels.py
```

### 🛠️ Tecnologias Utilizadas
- Python: Processamento de dados.
- Pandas: Manipulação de DataFrames e limpeza de dados.
- Matplotlib: Geração de visualizações gráficas de alta resolução (300 DPI).
- KaggleHub: Gerenciamento dinâmico do dataset, evitando o armazenamento de arquivos pesados no GitHub.
- Hugging Face / Sentence-Transformers: Implementação do modelo BERT (all-MiniLM-L6-v2) para normalização semântica e classificação dos clusters.

