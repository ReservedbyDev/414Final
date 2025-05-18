import os
import requests
import psycopg2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dotenv import load_dotenv
from auth import refresh_access_token
from collections import defaultdict

load_dotenv('db.env')
DB_URL = os.getenv('DATABASE_URL')

def get_users():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute('SELECT user_id, access_token FROM users')
    users = cur.fetchall()
    conn.close()
    return users

def get_user_genres(user_id, token):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.spotify.com/v1/me/top/artists?limit=50&time_range=medium_term', headers=headers)

    if response.status_code == 401:
        print(f'\u26a0\ufe0f Token expired for {user_id}. Attempting to refresh')
        token = refresh_access_token(user_id)
        if not token:
            print(f'\u274c Failed to refresh token for {user_id}')
            return []
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get('https://api.spotify.com/v1/me/top/artists?limit=50&time_range=medium_term', headers=headers)

    if response.status_code != 200:
        print(f'Failed to fetch for {user_id}: {response.text}')
        return []

    data = response.json().get('items', [])
    genres = []
    for artist in data:
        genres.extend(artist['genres'])
    return genres

def build_genre_matrix(user_tokens):
    genre_counts = {}

    for user_id, token in user_tokens:
        genres = get_user_genres(user_id, token)
        print(f'🎧 {user_id}: {len(genres)} genres fetched')
        genre_series = pd.Series(genres)
        counts = genre_series.value_counts().to_dict()
        genre_counts[user_id] = counts

    real_user_ids = list(genre_counts.keys())

    if real_user_ids:
        for i in range(95):
            base_user_id = np.random.choice(real_user_ids)
            base_vector = pd.Series(genre_counts[base_user_id]).fillna(0)
            noise = np.random.normal(0, 0.5, size=base_vector.shape)
            noisy_vector = (base_vector + noise).round().clip(lower=0)
    
            genre_counts[f'Virtual User {i+1}'] = noisy_vector.to_dict()
        else:
            print('No real users')

    return pd.DataFrame.from_dict(genre_counts, orient='index').fillna(0)

def analyze_users():
    users = get_users()
    if not users:
        print('No users found.')
        return

    print(f'Real users fetched from DB ({len(users)}):')
    for user_id, _ in users:
        print(f'- {user_id}')
    genre_df = build_genre_matrix(users)
    normalized = pd.DataFrame(StandardScaler().fit_transform(genre_df), columns=genre_df.columns, index=genre_df.index)
    z_scores = np.abs(zscore(normalized))
    outliers = (z_scores > 5.5).any(axis=1)

    cleaned = normalized[~outliers].copy()
    print(f'Remaining users after outlier filtering: {len(cleaned)}')

    similarity_matrix = 1 - cosine_distances(cleaned)
    G = nx.Graph()
    user_ids = cleaned.index.tolist()
    G.add_nodes_from(user_ids)

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            sim = similarity_matrix[i][j]
            if sim > 0.3:
                G.add_edge(user_ids[i], user_ids[j], weight=sim)

    centrality = nx.degree_centrality(G)
    cleaned.loc[:, 'centrality'] = cleaned.index.map(centrality).fillna(0)
    print('Top 5 Most Central Users:')
    print(cleaned['centrality'].sort_values(ascending=False).head(5))

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(cleaned) - 1))
    tsne_components = tsne.fit_transform(cleaned.drop(columns='centrality'))
    cleaned['tsne1'] = tsne_components[:, 0]
    cleaned['tsne2'] = tsne_components[:, 1]

    clustering_data = normalize(cleaned.drop(columns=['centrality', 'tsne1', 'tsne2']))
    distance_matrix = pairwise_distances(clustering_data, metric='cosine')

    best_k, best_score = 2, -1
    for k in range(2, min(12, len(cleaned))):
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        if score > best_score:
            best_score, best_k = score, k

    model = AgglomerativeClustering(n_clusters=best_k, metric='precomputed', linkage='average')
    cleaned['cluster'] = model.fit_predict(distance_matrix)
    print(f'k = {best_k}')

    original_genres = genre_df.loc[cleaned.index]
    original_genres['cluster'] = cleaned['cluster']

    print('Top Genres per Cluster:')
    for cluster_id, group in original_genres.groupby('cluster'):
        top_genres = group.drop(columns='cluster').sum().sort_values(ascending=False).head(5)
        genre_list = [f"{genre} ({int(count)} plays)" for genre, count in top_genres.items()]
        genre_string = ", ".join(genre_list)
        print(f"Cluster {cluster_id}: {genre_string}\n")

    print('Top 5 Genres for Each Real User:')
    for user_id in genre_df.index:
        if not user_id.startswith('Virtual') and user_id in cleaned.index:
            top_genres = genre_df.loc[user_id].sort_values(ascending=False).head(5)
            genre_list = [f"{genre} ({int(count)} plays)" for genre, count in top_genres.items()]
            genre_string = ", ".join(genre_list)
            print(f'{user_id}:') 
            print(f'{genre_string}\n')

    cleaned.to_csv('genre_network_results.csv')

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=cleaned, x='tsne1', y='tsne2', hue='cluster', palette='deep', legend='full')
    plt.title('User Clusters Based on Genre Preferences')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()

    G_force = nx.Graph()
    genre_popularity = {}
    genre_connections = defaultdict(set)

    for user in original_genres.index:
        for genre, count in original_genres.loc[user].items():
            if genre == 'cluster' or count == 0:
                continue
            G_force.add_edge(user, genre, weight=int(count))
            genre_popularity[genre] = genre_popularity.get(genre, 0) + int(count)
            genre_connections[genre].add(user)

    node_sizes = []
    for n in G_force.nodes():
        if n in cleaned.index:  
            node_sizes.append(400)
        elif n in genre_connections: 
            size = 100 + len(genre_connections[n]) * 50  
            node_sizes.append(size)
        else:
            node_sizes.append(100)  

    node_colors = []
    for n in G_force.nodes():
        if n in cleaned.index:
            if n.startswith('Virtual'):
                node_colors.append('skyblue')
            else:
                node_colors.append('orange')
        else:
            node_colors.append('lightgreen')

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_force, seed=42, k=0.4)
    nx.draw_networkx_nodes(G_force, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G_force, pos, width=[G_force[u][v]['weight'] * 0.2 for u, v in G_force.edges()], alpha=0.4)

    labels = {}
    for n in G_force.nodes():
        if n not in cleaned.index:
            labels[n] = n 
        elif not n.startswith('Virtual'):
            labels[n] = n 

    nx.draw_networkx_labels(G_force, pos, labels=labels, font_size=8)

    plt.title('User–Genre Network (Real = Orange, Virtual = Skyblue, Genres = Green)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    similarity_threshold = 0.3
    G_user = nx.Graph()
    user_ids = cleaned.index.tolist()
    G_user.add_nodes_from(user_ids)

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            sim = similarity_matrix[i][j]
            if sim > similarity_threshold:
                G_user.add_edge(user_ids[i], user_ids[j], weight=sim)


    centrality_user_graph = nx.degree_centrality(G_user)
    nx.set_node_attributes(G_user, centrality_user_graph, 'centrality')
    node_sizes = [centrality_user_graph.get(n, 0) * 2000 for n in G_user.nodes]
    node_colors = ["orange" if not n.startswith("Virtual") else "skyblue"
        for n in G_user.nodes()]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_user, seed=42, k=0.4)
    nx.draw_networkx_nodes(G_user, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G_user, pos, width=[G_user[u][v]['weight'] * 2 for u, v in G_user.edges()], alpha=0.4)
    labels = {n: n for n in G_user.nodes() if not n.startswith('Virtual')}
    
    nx.draw_networkx_labels(G_user, pos, labels=labels, font_size=8)
    plt.title('User Similarity Network (Real = Orange, Virtual = Skyblue)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    genre_user_counts = {genre: len(users) for genre, users in genre_connections.items()}
    top5_genres_all_users = sorted(genre_user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    genres, user_counts = zip(*top5_genres_all_users)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=user_counts, y=genres, color='steelblue')
    plt.xlabel('Number of Users Connected')
    plt.ylabel('Genre')
    plt.title('Top 5 Genres Connected to the Most Users')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    analyze_users()