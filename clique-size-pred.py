#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:35:59 2025

@author: niloyroot
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from networkx.algorithms.core import core_number
from networkx.linalg.spectrum import adjacency_spectrum
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from decision_tree_gpt import DecisionTree
from random_forest import RandomForest
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import warnings

def generate_barabasi_albert_graph(n, m):
    G = nx.barabasi_albert_graph(n, m)
    clique_size = nx.graph_clique_number(G)
    return G, clique_size

def generate_graph_and_clique_size(n_nodes, p):
    G = nx.erdos_renyi_graph(n_nodes, p)
    max_clique = nx.graph_clique_number(G)  
    return G, max_clique

def generate_watts_strogatz_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    clique_size = nx.graph_clique_number(G)
    return G, clique_size

def generate_random_regular_graph(n, d):
    G = nx.random_regular_graph(d, n)
    clique_size = nx.graph_clique_number(G)
    return G, clique_size

def generate_powerlaw_cluster_graph(n, m, p):
    G = nx.powerlaw_cluster_graph(n, m, p)
    clique_size = nx.graph_clique_number(G)
    return G, clique_size



def safe_feature(f, default=np.nan):
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return default
    return wrapped

def compute_graph_features(G):
    features = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "max_degree": max(dict(G.degree()).values(), default=0),
        "clustering_coeff": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }
    
    
    features['assortativity'] = safe_feature(nx.degree_assortativity_coefficient)(G)

    
    if nx.is_connected(G):
        features['avg_shortest_path_length'] = nx.average_shortest_path_length(G)
        features['diameter'] = nx.diameter(G)
        ecc = nx.eccentricity(G)
    else:
        
        GCC = max(nx.connected_components(G), key=len)
        subG = G.subgraph(GCC)
        features['avg_shortest_path_length'] = safe_feature(nx.average_shortest_path_length)(subG)
        features['diameter'] = safe_feature(nx.diameter)(subG)
        ecc = safe_feature(nx.eccentricity)(subG)


    if ecc:
        features['avg_eccentricity'] = np.mean(list(ecc.values()))
        features['radius'] = min(ecc.values())
    else:
        features['avg_eccentricity'] = np.nan
        features['radius'] = np.nan

    cores = core_number(G)
    core_vals = list(cores.values())
    features['max_kcore'] = max(core_vals)
    features['avg_kcore'] = np.mean(core_vals)

    tri_dict = nx.triangles(G)
    tri_counts = list(tri_dict.values())
    features['total_triangles'] = sum(tri_counts) // 3
    features['avg_triangles'] = np.mean(tri_counts)

    try:
        eigenvals = adjacency_spectrum(G)
        features['spectral_radius'] = max(np.abs(eigenvals))
    except:
        features['spectral_radius'] = np.nan

    return features

def generate_dataset(n_samples, n_nodes_range=(30, 40), p_range=(0.5, 0.8)):
    X = []
    y = []
    for _ in tqdm(range(n_samples)):
        n = np.random.randint(*n_nodes_range)
        p = np.random.uniform(*p_range)
        G, clique_size = generate_graph_and_clique_size(n, p)
        features = compute_graph_features(G)
        X.append(features)
        y.append(clique_size)
    return pd.DataFrame(X), np.array(y)

def generate_mixed_dataset(n_samples):
    X, y = [], []
    for _ in tqdm(range(n_samples)):
        graph_type = np.random.choice(["ER", "ER", "ER", "BA", "WS", "RR", "PL"])
        if graph_type == "ER":
            n = np.random.randint(50, 100)
            p = np.random.uniform(0.3, 0.7)
            G, clique = generate_graph_and_clique_size(n, p)
        elif graph_type == "BA":
            n = np.random.randint(50, 60)
            m = np.random.randint(1, min(10, n))
            G, clique = generate_barabasi_albert_graph(n, m)
        elif graph_type == "WS":
            n = np.random.randint(50, 60)
            k = np.random.randint(2, 10)
            p = np.random.uniform(0.1, 0.5)
            G, clique = generate_watts_strogatz_graph(n, k, p)
        elif graph_type == "RR":
            n = np.random.randint(30, 50)
            d = np.random.randint(2, min(10, n))
            try:
                G, clique = generate_random_regular_graph(n, d)
            except:
                continue
        elif graph_type == "PL":
            n = np.random.randint(30, 50)
            m = np.random.randint(1, min(10, n))
            p = np.random.uniform(0.1, 0.5)
            G, clique = generate_powerlaw_cluster_graph(n, m, p)

        features = compute_graph_features(G)
        X.append(features)
        y.append(clique)
    return pd.DataFrame(X), np.array(y)


def train_mlp(X_train, y_train, X_test, y_test):
    mlpr = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,  alpha=1e-2)
    mlpr.fit(X_train, y_train)
    y_pred = mlpr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MLP Regressor Test MSE: {mse:.4f}")
    y_pred_ = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_)
    print("MLP Regressor Accuracy:", accuracy)
    
    mlpc = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=1e-2)
    mlpc.fit(X_train, y_train)
    y_pred = mlpc.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MLP Classifier Test MSE: {mse:.4f}")
    accuracy = accuracy_score(y_test, y_pred)
    print("MLP Classifier Accuracy:", accuracy)
    return mlpr

def train_linear_regressor(X_train, y_train, X_test, y_test):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression Test MSE: {mse:.4f}")
    
    y_pred_ = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_)
    print("Linear Regression Accuracy:", accuracy)
    return linreg

def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=100,  
        max_depth=None,
        n_jobs=-1          
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest Regressor Test MSE: {mse:.4f}")
    
    y_pred_ = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_)
    print("Random Forest Regressor Accuracy:", accuracy)
    
    rf2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1)
    rf2.fit(X_train, y_train)
    y_pred = rf2.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest Classifier Test MSE: {mse:.4f}")
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Classifier Accuracy:", accuracy)
    return rf

def train_random_forest_class(X_train, y_train, X_test, y_test):
    l = [i for i in range(1,101)]
    f = [i for i in range(1,18)]
    rf = RandomForest(
        n_trees=100,
        n_bootstrap=len(y_train)+len(y_test),
        max_depth=10,
        min_samples=2,
        min_gain=1e-3,
        n_split='sqrt',
        quantiles=[25, 50, 75],
    )
    rf2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10)
    rf.fit(X_train, y_train, l, f)
    rf2.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred2 = rf2.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse2 = mean_squared_error(y_test, y_pred2)
    print(f"Random Forest Implementation Classifier MSE: {mse:.4f}")
    print(f"Random Forest sklearn Classifier MSE: {mse2:.4f}")
    return rf

def train_decision_tree(X_train, y_train, X_test, y_Test):
    dt = DecisionTree(max_depth=5)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest Test MSE: {mse:.4f}")
    return dt

def evaluate_mlp_hyperparams(X_train, y_train, X_test, y_test, param_name, param_values):
    mse_scores = []

    for val in tqdm(param_values, desc=f"Testing {param_name}"):
        kwargs = {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 500,
            "random_state": 42,
            "alpha": 0,  # default
        }

        if param_name == "hidden_layer_sizes":
            kwargs["hidden_layer_sizes"] = val
        elif param_name == "max_iter":
            kwargs["max_iter"] = val
        elif param_name == "random_state":
            kwargs["random_state"] = val
        elif param_name == "alpha":
            kwargs["alpha"] = val
        else:
            raise ValueError(f"Unsupported param: {param_name}")

        mlp = MLPRegressor(**kwargs)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    # Plotting
    plt.figure(figsize=(10, 6))
    if param_name == "hidden_layer_sizes":
        plt.plot(range(len(param_values)), mse_scores, marker='o')
        plt.xticks(range(len(param_values)), [str(p) for p in param_values])
    else:
        plt.plot(param_values, mse_scores, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Test MSE")
    plt.title(f"MLP: {param_name} vs. Test MSE")
    plt.grid(True)
    plt.xscale("log") if param_name == "alpha" else None
    plt.show()


if __name__ == "__main__":
    print("Generating dataset...")
    X, y = generate_mixed_dataset(n_samples=5000)

    print("Training model...")
    X_df = pd.DataFrame(X)
    imputer = SimpleImputer(strategy='mean')  
    X_imputed = imputer.fit_transform(X_df)
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X_imputed), y, test_size=0.66, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Run and plot different hyperparameter evaluations
    #evaluate_mlp_hyperparams(X_train.values, y_train, X_test.values, y_test, 
                         #param_name="hidden_layer_sizes", 
                         #param_values=[(64,), (128,), (32, 32, 16), (32, 16), (64, 32), (128, 64), (64, 32, 16), (64, 64, 32), (64, 64), (32, 32)])

    #evaluate_mlp_hyperparams(X_train.values, y_train, X_test.values, y_test, 
                         #param_name="max_iter", 
                         #param_values=[100, 200, 300, 500, 700, 1000])

    #evaluate_mlp_hyperparams(X_train.values, y_train, X_test.values, y_test, 
                         #param_name="random_state", 
                         #param_values=[0,12,24,32,46,54,60,78,83,99,100])
    #evaluate_mlp_hyperparams(X_train.values, y_train, X_test.values, y_test, 
                         #param_name="alpha", 
                         #param_values=[0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1])

    
    model = train_mlp(X_train.values, y_train, X_test.values, y_test)
    model2 = train_linear_regressor(X_train.values, y_train, X_test.values, y_test)
    model3 = train_random_forest(X_train.values, y_train, X_test.values, y_test)
    #model4 = train_random_forest_class(X_train.values, y_train, X_test.values, y_test)
    #model5 = train_decision_tree(X_train.values, y_train, X_test.values, y_test)
    #result = permutation_importance(model, X_test.values, y_test)
    
    #print("\n=== Random Forest Feature Importance ===")
    feature_importances = pd.DataFrame({
        "Feature": X_df.columns,
        "Importance": model3.feature_importances_
    }).sort_values("Importance", ascending=False)
    #print(feature_importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"])
    plt.title("Random Forest Feature Importance")
    plt.show()
    
    estimator = model3.estimators_[0]

    plt.figure(figsize=(20, 10))
    plot_tree(estimator, 
              feature_names=list(X_df.columns), 
              filled=True, 
              rounded=True,
              max_depth=3)
    plt.title("Visualization of One Decision Tree from the Random Forest")
    plt.show()