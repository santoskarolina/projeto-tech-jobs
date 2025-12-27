import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import kagglehub

import re

dicionario = {
    'mid senior': 'senior', 'mid-senior': 'senior', 'sr': 'senior', 'sênior': 'senior', 'senior': 'senior',
    'mid': 'mid-level', 'pleno': 'mid-level', 'associate': 'mid-level', 'mid-level': 'mid-level',
    'junior': 'junior', 'jr': 'junior', 'entry-level': 'junior', 'estágio': 'junior', 'estagiário': 'junior',
    'lead': 'lead', 'tech lead': 'lead', 'tech-lead': 'lead', 'team lead': 'lead','team-lead': 'lead', 'chief': 'lead', 'chefe': 'lead','manager': 'lead',
    'diretor': 'director', 'director': 'director',
    'principal': 'staff', 'staff': 'staff',
    'specialist': 'specialist',
    'vp': 'director',
}

TERMOS_NIVEL_BUSCA = set(dicionario.keys())

path = kagglehub.dataset_download("yazeedfares/software-engineering-jobs-dataset")
csv_file = os.path.join(path, "postings2.csv")
try:
    df = pd.read_csv(csv_file)
except:
    df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')

df['job_title_lower'] = df['job_title'].astype(str).str.lower().fillna('')
df['job_level_lower'] = df['job level'].astype(str).str.lower().fillna('')

def obter_nivel_normalizado(row):
        titulo = row['job_title_lower']
        nivel_coluna = row['job_level_lower']
        
        for termo in TERMOS_NIVEL_BUSCA:
            if re.search(r'\b' + re.escape(termo) + r'\b', titulo):
                return dicionario[termo]
        
        if nivel_coluna and nivel_coluna != 'nan':
            return dicionario.get(nivel_coluna, nivel_coluna)

        return 'não especificado'

df['nivel_final_normalizado'] = df.apply(obter_nivel_normalizado, axis=1)

contador_normalizado = Counter(df['nivel_final_normalizado'])

ranking_niveis_df = pd.DataFrame(contador_normalizado.items(), columns=['Nível', 'Frequencia'])
ranking_niveis_df = ranking_niveis_df.sort_values(by='Frequencia', ascending=False).reset_index(drop=True)

top_niveis_df = ranking_niveis_df.head(5).iloc[::-1] 
plt.figure(figsize=(10, 6))
plt.barh(top_niveis_df['Nível'].str.title(), top_niveis_df['Frequencia'], color='maroon') 
plt.title(f'Níveis de experiência mais solicitados nos anuncios', fontsize=14)
plt.xlabel('Frequencia Total', fontsize=11)
plt.ylabel('Nível', fontsize=11)
for index, value in enumerate(top_niveis_df['Frequencia']):
        plt.text(value, index, f' {value}', va='center')
plt.tight_layout()
output_filename = 'niveis.png'
plt.savefig(output_filename)
plt.close()