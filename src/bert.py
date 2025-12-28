import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import kagglehub
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

DICIONARIO_LINGUAGENS = {
    'python': 'Python', 
    'java': 'Java', 
    'javascript': 'JavaScript', 
    'js': 'JavaScript',
    'typescript': 'TypeScript', 
    'ts': 'TypeScript',
    'TypeScript': 'TypeScript',
    'c++': 'C++', 
    'cpp': 'C++',
    'c#': 'C#', 
    'csharp': 'C#',
    'c': 'C', 
    'php': 'PHP', 
    'ruby': 'Ruby', 
    'go': 'Go', 
    'golang': 'Go',
    'swift': 'Swift', 
    'kotlin': 'Kotlin', 
    'r': 'R', 
    'matlab': 'MATLAB', 
    'scala': 'Scala', 
    'perl': 'Perl',
    'shell': 'Shell', 
    'bash': 'Bash', 
    'haskell': 'Haskell', 
    'lua': 'Lua', 
    'dart': 'Dart', 
    'vba': 'VBA', 
    'groovy': 'Groovy',
    'rust': 'Rust', 
    'sql': 'SQL', 
    'pl/sql': 'PL/SQL', 
    'nosql': 'NoSQL', 
    'assembly': 'Assembly', 
    'cobol': 'COBOL',
    'fortran': 'Fortran', 
    'elixir': 'Elixir', 
    'erlang': 'Erlang', 
    'pascal': 'Pascal', 
    'delphi': 'Delphi', 
    'node': 'Node.js', 
    'node.js': 'Node.js', 
    'nodejs': 'Node.js',
    'node js': 'Node.js',
}

SINONIMOS_TECNOLOGIA = {
    'amazon web services': 'AWS',
    'amazon cloud': 'AWS',
    'aws': 'AWS',
    'google cloud platform': 'GCP',
    'gcp': 'GCP',
    'google cloud': 'GCP',
    'microsoft azure': 'Azure',
    'azure': 'Azure',
    'react.js': 'React',
    'reactjs': 'React',
    'react js': 'React',
    'react': 'React',
    'k8s': 'Kubernetes',
    'kubernetes': 'Kubernetes',
    'github': 'Git',
    'gitlab': 'Git',
    'git': 'Git'
}

ANCORAS = {
    "Tecnologias": model.encode("software infrastructure cloud platform framework tool library database api docker aws"),
    "Habilidades": model.encode("soft skill human behavior social interaction management mindset adaptability communication teamwork")
}

EXCLUSAO = {'software engineering', 'software development', 'engineering', 'development', 'design', 'architecture'}

def classificar_hibrido(skill):
    s_lower = skill.lower().strip()
    
    if s_lower in EXCLUSAO or len(s_lower) < 2:
        return None, None

    if s_lower in DICIONARIO_LINGUAGENS:
        return "Linguagens", DICIONARIO_LINGUAGENS[s_lower]

    embedding = model.encode(s_lower)
    sim_tec = util.cos_sim(embedding, ANCORAS["Tecnologias"]).item()
    sim_hab = util.cos_sim(embedding, ANCORAS["Habilidades"]).item()
    
    if sim_tec > sim_hab and sim_tec > 0.30:
        nome_padronizado = SINONIMOS_TECNOLOGIA.get(s_lower, skill.capitalize())
        return "Tecnologias", nome_padronizado
    elif sim_hab > sim_tec and sim_hab > 0.30:
        return "Habilidades", skill.capitalize()
    
    return None, None

path = kagglehub.dataset_download("yazeedfares/software-engineering-jobs-dataset")
df = pd.read_csv(os.path.join(path, "postings2.csv"))

df['skills_list'] = df['job_skills'].apply(lambda x: [s.lower().strip() for s in ast.literal_eval(x)] if pd.notna(x) else [])
contador_bruto = Counter([s for sublist in df['skills_list'] for s in sublist])

stats = {"Linguagens": Counter(), "Tecnologias": Counter(), "Habilidades": Counter()}

for skill, freq in contador_bruto.most_common(600):
    cat, nome = classificar_hibrido(skill)
    if cat:
        stats[cat][nome] += freq

config_graficos = {
    "Linguagens": {"cor": "#6a1b9a", "arquivo": "linguagens.png", "nomeGrafico": 'Linguagens mais Solicitadas no Anuncios'},
    "Tecnologias": {"cor": "#2e7d32", "arquivo": "tecnologias.png", "nomeGrafico": 'Tecnologias mais Solicitadas no Anuncios'},
    "Habilidades": {"cor": "#ef6c00", "arquivo": "habilidades.png", "nomeGrafico": 'Habilidades mais Solicitadas no Anuncios'}
}

for cat, dados in stats.items():
    plt.figure(figsize=(10, 8))
    
    df_plot = pd.DataFrame(dados.most_common(10), columns=['Skill', 'Freq']).iloc[::-1]
    
    if not df_plot.empty:
        plt.barh(df_plot['Skill'], df_plot['Freq'], color=config_graficos[cat]["cor"])
        plt.title(f'{config_graficos[cat]['nomeGrafico']}', fontsize=14)
        plt.xlabel('Frequência (Número de menções)')
        plt.tight_layout()
        
        plt.savefig(config_graficos[cat]["arquivo"], dpi=300)
    plt.close()