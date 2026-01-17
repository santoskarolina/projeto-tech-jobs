import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import kagglehub

import re

dicionario = {
    'mid senior': 'Sênior', 'mid-senior': 'Sênior', 'sr': 'Sênior', 'Sênior': 'Sênior', 'senior': 'Sênior',
    'mid': 'Pleno', 'pleno': 'Pleno', 'associate': 'Pleno', 'mid-level': 'Pleno',
    'junior': 'Júnior', 'jr': 'Júnior', 'entry-level': 'Júnior', 'estágio': 'Júnior', 'estagiário': 'Júnior',
    'lead': 'Lead', 'tech lead': 'Lead', 'tech-lead': 'Lead', 'team lead': 'Lead','team-lead': 'Lead', 'chief': 'Lead', 'chefe': 'Lead','manager': 'leLeadad',
    'diretor': 'director', 'director': 'director',
    'principal': 'staff', 'staff': 'staff',
    'specialist': 'specialist',
    'vp': 'director',
}

TERMOS_NIVEL_BUSCA = set(dicionario.keys())
PADRAO_DIPLOMA = r"(bachelor|degree|university|graduate|master|phd|education|engenh|superior|formação)"

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

def verificar_escolaridade(texto):
    if not isinstance(texto, str): return "Não Mencionado"
    if re.search(PADRAO_DIPLOMA, texto.lower()):
        return "Exige Diploma"
    return "Não Mencionado"

coluna_texto = 'job_summary' if 'job_summary' in df.columns else 'job_description'
df['exige_diploma'] = df[coluna_texto].apply(verificar_escolaridade)

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

niveis_principais = ['Júnior', 'Pleno', 'Sênior', 'Lead']
df_filtrado = df[df['nivel_final_normalizado'].isin(niveis_principais)].copy()

# Criar tabela de porcentagens
analise_esc = df_filtrado.groupby(['nivel_final_normalizado', 'exige_diploma']).size().unstack(fill_value=0)
analise_pct = analise_esc.div(analise_esc.sum(axis=1), axis=0) * 100

# Ordenar logicamente
ordem_logica = [n for n in niveis_principais if n in analise_pct.index]
analise_pct = analise_pct.reindex(ordem_logica)

# 6. Geração do Gráfico
plt.figure(figsize=(10, 6))
# Vermelho para 'Não Mencionado', Verde para 'Exige Diploma'
ax = analise_pct.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], figsize=(10, 6))

plt.title('Exigência de Escolaridade por Nível de Experiência', fontsize=14)
plt.ylabel('Porcentagem de Vagas (%)')
plt.xlabel('Nível Normalizado')
plt.xticks(rotation=0)
plt.legend(title='Escolaridade', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adicionar rótulos de porcentagem dentro das barras
for p in ax.patches:
    h = p.get_height()
    if h > 5: # Só escreve se a fatia for maior que 5%
        ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width()/2, p.get_y() + h/2), 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('escolaridade_por_nivel.png', dpi=300)
plt.show()