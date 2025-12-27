# Análise de Competências em Vagas de Engenharia de Software

Este projeto realiza uma análise quantitativa das linguagens de programação, tecnologias e habilidades interpessoais (*soft skills*) mais requisitadas no mercado de tecnologia atual. Os dados são processados a partir de um dataset de anúncios de vagas coletado via Kaggle.

## 📋 Sobre o Projeto
O objetivo desta análise é identificar tendências de mercado para auxiliar na formação acadêmica e profissional. O script realiza a extração, limpeza (limpeza de caracteres especiais e correção de encoding) e a padronização de termos técnicos (ex: agrupando 'Nodejs' e 'Node.js').

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

pip install pandas matplotlib kagglehub
```

### 2. Execução
Execute o script principal:
```bash
python src/index.py
```

### 🛠️ Tecnologias Utilizadas
- Python: Processamento de dados.
- Pandas: Manipulação de DataFrames e limpeza de dados.
- Matplotlib: Geração de visualizações gráficas de alta resolução (300 DPI).
- KaggleHub: Gerenciamento dinâmico do dataset, evitando o armazenamento de arquivos pesados no GitHub.
