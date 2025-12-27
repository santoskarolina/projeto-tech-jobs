import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import kagglehub

MAPEAMENTO = {
       "linguagens": {
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
        'powershell': 'PowerShell', 
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
       },
      "habilidades": {
          'team work': 'teamwork', 'teamwork': 'teamwork', 'collaboration': 'collaboration',
            'communication': 'communication', 'leadership': 'leadership',
            'problem solving': 'problem solving', 'adaptability': 'adaptability',
            'proactive': 'proactivity', 'proactivity': 'proactivity', 'creativity': 'creativity',
            'organization': 'organization', 'time management': 'time management',
            'ethics': 'ethics', 'negotiation': 'negotiation', 'feedback': 'feedback',
            'growth mindset': 'growth mindset', 'customer focus': 'customer focus',
            'sales': 'sales', 'learning': 'learning', 'autonomy': 'autonomy',
            'resilience': 'resilience', 'english': 'english', 'spanish': 'spanish', 'german': 'german'
      },
      "tecnologias": {
          'agile': 'agile', 'scrum': 'scrum', 'kanban': 'kanban', 'design thinking': 'design thinking', 
            'devops': 'devops', 'ci/cd': 'ci/cd', 'ci cd': 'CI/CD',
            'git': 'git', 'github': 'git', 'gitlab': 'git', 'bitbucket': 'git',
            'cloud': 'cloud', 'aws': 'aws', 'azure': 'azure', 'gcp': 'gcp', 'google cloud': 'cloud',  'google-cloud': 'cloud', 'amazon web services': 'aws',
            'docker': 'docker', 'kubernetes': 'kubernetes',
            'data analysis': 'data analysis', 'machine learning': 'machine learning', 'ia': 'ai', 'ai': 'ai', 
            'database': 'database', 'tableau': 'tableau', 'power bi': 'power bi', 'excel': 'excel', 
            'jira': 'jira', 'confluence': 'confluence'
    }
}

habilidades = set(MAPEAMENTO['habilidades'].keys())
tecnologias = set(MAPEAMENTO['tecnologias'].keys())
linguagens = set(MAPEAMENTO['linguagens'].keys())

def safe_literal_eval(x):
    if pd.isna(x):
        return None
    try:
        result = ast.literal_eval(x)
        if isinstance(result, list):
            return result
        return None
    except (ValueError, TypeError, SyntaxError):
        return None
    
path = kagglehub.dataset_download("yazeedfares/software-engineering-jobs-dataset")
csv_file = os.path.join(path, "postings2.csv")
try:
    df = pd.read_csv(csv_file)
except:
    df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')


df['skills_list'] = df['job_skills'].apply(safe_literal_eval)

valid_skills_lists = df['skills_list'].dropna()
todas_as_skills = [skill.strip().lower() for sublist in valid_skills_lists for skill in sublist]

contador_skills = Counter(todas_as_skills)
skills_linguagens = []
skills_tecnologias = []
skills_habilidades= []

for skill, count in contador_skills.items():
    if skill in linguagens:
        skills_linguagens.append({'Linguagem': skill.capitalize(), 'Frequencia': count})
    elif skill in tecnologias:
        skills_tecnologias.append({'Tecnologia': skill.capitalize(), 'Frequencia': count})
    elif skill in habilidades:
        skills_habilidades.append({'Habilidades': skill.capitalize(), 'Frequencia': count})

ranking_linguagens_df = pd.DataFrame(skills_linguagens).sort_values(
    by='Frequencia', ascending=False
).reset_index(drop=True)

if not ranking_linguagens_df.empty:
    top_linguagens_df = ranking_linguagens_df.head(10).iloc[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_linguagens_df['Linguagem'], top_linguagens_df['Frequencia'], color='purple')
    
    plt.title(f'Linguagens de programação mais solicitadas nos anuncios', fontsize=14)
    plt.xlabel('Frequencia', fontsize=11)
    plt.tight_layout()
    plt.savefig('linguagens.png')
    plt.close() 

ranking_tecnologias_df = pd.DataFrame(skills_tecnologias).sort_values(
    by='Frequencia', ascending=False
).reset_index(drop=True)

if not ranking_tecnologias_df.empty:
    top_tecnologias_df = ranking_tecnologias_df.head(10).iloc[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_tecnologias_df['Tecnologia'], top_tecnologias_df['Frequencia'], color='darkgreen')
    
    plt.title(f'Tecnologias mais solicitadas nos anuncios', fontsize=14)
    plt.xlabel('Frequencia', fontsize=11)
    plt.tight_layout()
    plt.savefig('tecnologias.png')
    plt.close()

ranking_habilidades_df = pd.DataFrame(skills_habilidades).sort_values(
    by='Frequencia', ascending=False
).reset_index(drop=True)

if not ranking_habilidades_df.empty:
    top_habilidades_df = ranking_habilidades_df.head(10).iloc[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_habilidades_df['Habilidades'], top_habilidades_df['Frequencia'], color='orange')
    
    plt.title(f'Habilidades mais solicitadas nos anuncios', fontsize=14)
    plt.xlabel('Frequencia', fontsize=11)
    plt.tight_layout()
    plt.savefig('habilidades.png')
    plt.close()