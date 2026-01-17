import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
import kagglehub
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

DICIONARIO_LINGUAGENS = {
    'python': 'Python', 'java': 'Java', 'javascript': 'JavaScript', 'js': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript', 'node': 'Node.js', 'sql': 'SQL',
    'c#': 'C#', 'csharp': 'C#', 'go': 'Go', 'golang': 'Go', 'ruby': 'Ruby', 'c++': 'C++'
}

SINONIMOS_TECNOLOGIA = {
    'aws': 'AWS', 'amazon web services': 'AWS', 'gcp': 'GCP', 'google cloud': 'GCP',
    'azure': 'Azure', 'react': 'React', 'reactjs': 'React', 'k8s': 'Kubernetes',
    'kubernetes': 'Kubernetes', 'docker': 'Docker', 'git': 'Git', 'spark': 'Spark'
}

ANCORAS = {
    "Tecnologias": model.encode("software infrastructure cloud platform framework tool library database api docker aws"),
    "Habilidades": model.encode("soft skill human behavior social interaction management mindset adaptability communication teamwork")
}

def classificar_hibrido(skill):
    s_lower = skill.lower().strip()
    if len(s_lower) < 2: return None, None
    if s_lower in DICIONARIO_LINGUAGENS: return "Linguagens", DICIONARIO_LINGUAGENS[s_lower]
    
    embedding = model.encode(s_lower)
    sim_tec = util.cos_sim(embedding, ANCORAS["Tecnologias"]).item()
    sim_hab = util.cos_sim(embedding, ANCORAS["Habilidades"]).item()
    
    if sim_tec > sim_hab and sim_tec > 0.30:
        nome_padronizado = SINONIMOS_TECNOLOGIA.get(s_lower, skill.capitalize())
        return "Tecnologias", nome_padronizado
    return None, None

path = kagglehub.dataset_download("yazeedfares/software-engineering-jobs-dataset")
df = pd.read_csv(os.path.join(path, "postings2.csv"))

def processar_vaga(job_skills_str):
    if pd.isna(job_skills_str): return []
    try:
        skills = ast.literal_eval(job_skills_str)
        limpas = []
        for s in skills:
            cat, nome = classificar_hibrido(s)
            if cat in ["Linguagens", "Tecnologias"]:
                limpas.append(nome)
        return list(set(limpas))
    except: return []

df['tech_profile'] = df['job_skills'].apply(processar_vaga)
df_filtered = df[df['tech_profile'].map(len) >= 2].copy().reset_index(drop=True)

all_techs = [item for sublist in df_filtered['tech_profile'] for item in sublist]
top_techs = [t[0] for t in pd.Series(all_techs).value_counts().head(100).items()]
df_filtered['tech_profile_filtered'] = df_filtered['tech_profile'].apply(lambda x: [s for s in x if s in top_techs])

mlb = MultiLabelBinarizer(classes=top_techs)
X = mlb.fit_transform(df_filtered['tech_profile_filtered'])
df_matrix = pd.DataFrame(X, columns=mlb.classes_)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_filtered['cluster'] = kmeans.fit_predict(df_matrix)

pca = PCA(n_components=2)
coords = pca.fit_transform(df_matrix)
df_filtered['pca_x'], df_filtered['pca_y'] = coords[:, 0], coords[:, 1]

df_filtered['distancia_centro'] = np.sqrt(df_filtered['pca_x']**2 + df_filtered['pca_y']**2)
vaga_extrema = df_filtered.loc[df_filtered['distancia_centro'].idxmax()]

print(f"\nSilhouette Score: {silhouette_score(df_matrix, df_filtered['cluster']):.3f}")

print("\n--- ANÁLISE DE PAPÉIS LATENTES ---")
for i in range(n_clusters):
    is_in_cluster = (df_filtered['cluster'] == i)
    top_skills = df_matrix.loc[is_in_cluster].sum().sort_values(ascending=False).head(5).index.tolist()
    top_titles = df_filtered.loc[is_in_cluster, 'job_title'].value_counts().head(3).index.tolist()
    print(f"\nCluster {i}:")
    print(f" > Skills Dominantes: {', '.join(top_skills)}")
    print(f" > Títulos comuns: {', '.join(top_titles)}")

print("\n--- EXEMPLO DE VAGA DE ESPECIALIZAÇÃO EXTREMA (OUTLIER) ---")
print(f"Título: {vaga_extrema['job_title']}")
print(f"Empresa: {vaga_extrema['company']}")
print(f"Skills Técnicas: {vaga_extrema['tech_profile_filtered']}")
print(f"Cluster: {vaga_extrema['cluster']}")
print(f"Coordenadas PCA: ({vaga_extrema['pca_x']:.2f}, {vaga_extrema['pca_y']:.2f})")

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_filtered, x='pca_x', y='pca_y', hue='cluster', palette='viridis', alpha=0.5)

plt.scatter(vaga_extrema['pca_x'], vaga_extrema['pca_y'], color='red', marker='X', s=200, label='Outlier Extremo')
plt.title('Mapa de Papéis Latentes com Identificação de Outlier')
plt.legend()
plt.savefig('cluster_map_outlier.png', dpi=300)
print("\nGráfico 'cluster_map_outlier.png' salvo com sucesso.")