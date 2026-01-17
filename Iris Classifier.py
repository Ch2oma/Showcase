# EEE485 Final Project: Iris Flower Classification
# Muhammed Yusuf Yaman (22002086)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ---preprocessing metrics & validation ---

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

def standard_scaler(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1 #prevent division by zero
    return (X_train - mean) / std, (X_test - mean) / std

def get_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def manual_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def classification_report(y_true, y_pred, target_names):
    """
    Generates a text report showing the main classification metrics.
    """
    classes = np.unique(y_true)

    report = f"{'':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    #classmetrics
    for i, class_label in enumerate(classes):
        tp = np.sum((y_pred == class_label) & (y_true == class_label))
        fp = np.sum((y_pred == class_label) & (y_true != class_label))
        fn = np.sum((y_pred != class_label) & (y_true == class_label))
        
        support = np.sum(y_true == class_label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        class_name = target_names[i] if i < len(target_names) else str(class_label)
        report += f"{class_name:<15} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}\n"
    
    report += "\n"
    
    #acc
    accuracy = np.mean(y_true == y_pred)
    total_support = np.sum(supports)
    report += f"{'accuracy':<15} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10}\n"
    
    #macro avg
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    report += f"{'macro avg':<15} {macro_prec:>10.2f} {macro_rec:>10.2f} {macro_f1:>10.2f} {total_support:>10}\n"
    
    #weighted avg by support
    supports = np.array(supports)
    weighted_prec = np.average(precisions, weights=supports)
    weighted_rec = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)
    report += f"{'weighted avg':<15} {weighted_prec:>10.2f} {weighted_rec:>10.2f} {weighted_f1:>10.2f} {total_support:>10}\n"
    
    return report

def k_fold_cross_validation(model, X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.seed(42) 
    np.random.shuffle(indices)
    scores = []
    
    for i in range(k):
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(get_accuracy(y[val_idx], preds))
        
    return np.mean(scores)

# ---algorithms ---

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _dist(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        return 0

    def predict(self, X_test):
        preds = []
        for x in X_test:
            distances = [self._dist(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            preds.append(np.bincount(k_labels).argmax())
        return np.array(preds)

class LogisticRegression:
    def __init__(self, lr=0.1, iterations=1000, lambd=0.0):
        self.lr = lr
        self.iters = iterations
        self.lambd = lambd
        self.models = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.models = []
        
        for cls in self.classes:
            y_bin = np.where(y == cls, 1, 0)
            weights = np.zeros(n_features)
            bias = 0
            
            for _ in range(self.iters):
                linear = np.dot(X, weights) + bias
                y_pred = self.sigmoid(linear)
                # gradient with L2 regularization term (Lambda)
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y_bin)) + (self.lambd/n_samples) * weights
                db = (1/n_samples) * np.sum(y_pred - y_bin)
                weights -= self.lr * dw
                bias -= self.lr * db
            self.models.append((weights, bias))

    def predict(self, X):
        probs = [self.sigmoid(np.dot(X, w) + b) for w, b in self.models]
        return np.argmax(np.array(probs).T, axis=1)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=3, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1: return None, None
        
        best_gini = self._gini(y)
        best_idx, best_thr = None, None
        feat_indices = np.arange(n)
        if self.max_features:
            feat_indices = np.random.choice(n, self.max_features, replace=False)

        for idx in feat_indices:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_mask = X[:, idx] < thr
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0: continue
                gini = (np.sum(left_mask)/m) * self._gini(y[left_mask]) + \
                       (np.sum(~left_mask)/m) * self._gini(y[~left_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_labels = X.shape[0], len(np.unique(y))
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < 2):
            return Node(value=np.bincount(y).argmax())

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=np.bincount(y).argmax())

        left_mask = X[:, feat_idx] < threshold
        return Node(feat_idx, threshold, 
                    self._grow_tree(X[left_mask], y[left_mask], depth + 1),
                    self._grow_tree(X[~left_mask], y[~left_mask], depth + 1))

    def _predict_one(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] < node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, max_features=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            final_preds.append(np.bincount(tree_preds[:, i]).argmax())
        return np.array(final_preds)

# --- main function ---
def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    names = iris.target_names
    
    #dataFrame for Visualization
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = [names[i] for i in y]

    #preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)
    X_train_s, X_test_s = standard_scaler(X_train, X_test)


    # [Visual 1] Feature Pair Plot
    print("\nGenerating Feature Pair Plots...")
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    plt.suptitle("Iris Feature Pair Plots", y=1.02)
    plt.savefig('feature_pairplot.png')
    
    #KNN test
    print("\n[analysis 1] KNN distance metrics")
    print(f"{'Metric':<15} {'Mean CV Accuracy':<20}")
    print("-" * 35)
    for m in ['euclidean', 'manhattan', 'chebyshev']:
        knn = KNN(k=5, metric=m)
        score = k_fold_cross_validation(knn, X_train_s, y_train, k=5)
        print(f"{m:<15} {score:.4f}")

    #Logistic Regression lambdas test
    print("\n[analysis 2] Logistic Regression regularization")
    print(f"{'Lambda':<15} {'Mean CV Accuracy':<20}")
    print("-" * 35)
    for lam in [0.01, 0.1, 1.0]:
        lr = LogisticRegression(lambd=lam)
        score = k_fold_cross_validation(lr, X_train_s, y_train, k=5)
        print(f"{lam:<15} {score:.4f}")

    #model comparisons
    print("\n[analysis 3] final model comparison (CV + Test)")
    
    models = {
        "KNN (k=5)": KNN(k=5, metric='euclidean'),
        "Logistic Reg (L=0.1)": LogisticRegression(lambd=0.1),
        "Decision Tree (d=3)": DecisionTree(max_depth=3),
        "Random Forest (10 Trees)": RandomForest(n_trees=10, max_depth=3, max_features=2)
    }
    
    #save predictions for conf matrices
    test_predictions = {}
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        print("-" * 30)
        
        #CV score
        cv_acc = k_fold_cross_validation(model, X_train_s, y_train, k=5)
        print(f"Mean CV Accuracy: {cv_acc:.4f}")
        
        #last train/test
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        test_predictions[name] = preds
        
        print(classification_report(y_test, preds, names))

    # [Visual 2] confusion matrices
    print("\n[Visuals] Generating Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, preds) in enumerate(test_predictions.items()):
        cm = manual_confusion_matrix(y_test, preds, n_classes=3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=names, yticklabels=names)
        axes[idx].set_title(name)
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')

    # [Visual 3] decision degions
    print("\n[Visuals] Generating Decision Regions...")
    #Petal Length & Petal Width for visualization since they provide best separability
    X_2d = X_train_s[:, [2, 3]] 
    
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    plot_models = [
        ("KNN (k=5)",KNN(k=5)),
        ("Logistic Regression", LogisticRegression(lambd=0.1)),
        ("Decision Tree", DecisionTree(max_depth=3)),
        ("Random Forest", RandomForest(n_trees=10))
    ]
    
    plt.figure(figsize=(12, 10))
    for i, (name, model) in enumerate(plot_models):
        model.fit(X_2d, y_train)
        
        #predict on meshgrid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.subplot(2, 2, i + 1)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, s=20, edgecolor='k')
        plt.title(f"{name} Boundaries")
        plt.xlabel("Petal Length (scaled)")
        plt.ylabel("Petal Width (scaled)")
    
    plt.tight_layout()
    plt.savefig('decision_regions.png')

if __name__ == "__main__":
    main()