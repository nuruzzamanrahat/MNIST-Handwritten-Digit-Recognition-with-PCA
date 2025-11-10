"""
MNIST Handwritten Digit Recognition with Dimensionality Reduction
Author: [Your Name]
Course: Mathematical Models for Machine Learning

This project demonstrates:
- Principal Component Analysis (PCA) for dimensionality reduction on real handwritten digits
- Supervised learning with multiple classifiers
- Model comparison and performance analysis
- Visualization of high-dimensional data (784 dimensions -> reduced)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MNISTRecognitionPCA:
    """
    A class for MNIST handwritten digit recognition using PCA and classifiers.
    """

    def __init__(self, n_components=None, sample_size=10000):
        """
        Initialize the digit recognition model.

        Parameters:
        -----------
        n_components : int or None
            Number of principal components to retain
        sample_size : int
            Number of samples to use (MNIST has 70k images, we can use subset for speed)
        """
        self.n_components = n_components
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.pca = None
        self.classifier = None

    def load_data(self):
        """Load and split the MNIST dataset."""
        print("Downloading MNIST dataset (this may take a minute on first run)...")

        # Load MNIST from OpenML
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.to_numpy() if hasattr(mnist.data, 'to_numpy') else np.array(mnist.data)
        y = mnist.target.to_numpy() if hasattr(mnist.target, 'to_numpy') else np.array(mnist.target)
        y = y.astype(int)

        # Use a subset for faster processing (optional)
        if self.sample_size and self.sample_size < len(X):
            np.random.seed(42)
            indices = np.random.choice(len(X), self.sample_size, replace=False)
            X = X[indices]
            y = y[indices]

        self.X = X
        self.y = y
        self.images = X.reshape(-1, 28, 28)  # Reshape to 28x28 images

        print(f"\nDataset loaded: {self.X.shape[0]} samples")
        print(f"Features per sample: {self.X.shape[1]} (28x28 pixel images)")
        print(f"Classes: {np.unique(self.y)}")
        print(f"Image dimensions: {self.images.shape[1]}x{self.images.shape[2]} pixels")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self):
        """Standardize the features."""
        print("\nStandardizing pixel values...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Data standardized (mean=0, std=1)")

    def apply_pca(self, n_components=None):
        """
        Apply PCA for dimensionality reduction.

        Parameters:
        -----------
        n_components : int or float
            Number of components or variance ratio to retain
        """
        if n_components is not None:
            self.n_components = n_components

        print(f"\nApplying PCA with {self.n_components} components...")
        self.pca = PCA(n_components=self.n_components)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)

        print(f"\nPCA Results:")
        print(f"Original dimensions: {self.X_train_scaled.shape[1]}")
        print(f"Reduced dimensions: {self.X_train_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return self.X_train_pca, self.X_test_pca

    def train_classifier(self, C=1.0, use_pca=True):
        """
        Train logistic regression classifier.

        Parameters:
        -----------
        C : float
            Inverse of regularization strength
        use_pca : bool
            Whether to use PCA-transformed data
        """
        if use_pca:
            X_train = self.X_train_pca
            X_test = self.X_test_pca
        else:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled

        print("\nTraining Logistic Regression classifier...")
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42,
            verbose=0
        )

        self.classifier.fit(X_train, self.y_train)

        # Predictions
        self.y_train_pred = self.classifier.predict(X_train)
        self.y_test_pred = self.classifier.predict(X_test)

        # Accuracy
        train_acc = accuracy_score(self.y_train, self.y_train_pred)
        test_acc = accuracy_score(self.y_test, self.y_test_pred)

        print(f"\nClassifier Performance:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")

        return train_acc, test_acc

    def plot_sample_digits(self, n_samples=20, save_path=None):
        """Plot sample digits from the dataset."""
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        for i in range(n_samples):
            axes[i].imshow(self.images[i], cmap='gray')
            axes[i].set_title(f'Label: {self.y[i]}', fontsize=12)
            axes[i].axis('off')

        plt.suptitle('Sample MNIST Handwritten Digits', fontsize=16, y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_variance_explained(self, max_components=100, save_path=None):
        """Plot cumulative explained variance by components."""
        if self.pca is None:
            print("Run apply_pca() first!")
            return

        # Limit to max_components for visualization
        n_show = min(max_components, len(self.pca.explained_variance_ratio_))
        variance_ratio = self.pca.explained_variance_ratio_[:n_show]
        cumsum = np.cumsum(variance_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Individual variance
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained Ratio', fontsize=12)
        ax1.set_title('Variance Explained by Each Component', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Cumulative variance
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=4)
        ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% Variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% Variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Find components for different variance thresholds
        full_cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        n_90 = np.argmax(full_cumsum >= 0.90) + 1
        n_95 = np.argmax(full_cumsum >= 0.95) + 1
        print(f"\nComponents needed for 90% variance: {n_90}")
        print(f"Components needed for 95% variance: {n_95}")

    def plot_pca_components(self, n_components=16, save_path=None):
        """Visualize the first n principal components as 28x28 images."""
        if self.pca is None:
            print("Run apply_pca() first!")
            return

        n_show = min(n_components, len(self.pca.components_))
        rows = int(np.sqrt(n_show))
        cols = int(np.ceil(n_show / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.ravel()

        for i in range(n_show):
            component = self.pca.components_[i].reshape(28, 28)
            axes[i].imshow(component, cmap='RdBu_r')
            axes[i].set_title(f'PC{i+1}\n({self.pca.explained_variance_ratio_[i]:.2%})',
                            fontsize=10)
            axes[i].axis('off')

        # Hide extra subplots
        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Principal Components Visualization (28x28)', fontsize=16, y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_2d_projection(self, save_path=None):
        """Plot 2D PCA projection of the data."""
        print("\nCreating 2D projection for visualization...")
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(self.X_train_scaled)

        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                            c=self.y_train, cmap='tab10',
                            alpha=0.5, s=20)
        plt.colorbar(scatter, label='Digit', ticks=range(10))
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('2D PCA Projection of MNIST Dataset', fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix for test predictions."""
        cm = confusion_matrix(self.y_test, self.y_test_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10),
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - MNIST Digit Recognition', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_test_pred))

    def compare_dimensions(self, component_range=[10, 30, 50, 100, 150, 200, 300],
                          save_path=None):
        """
        Compare model performance across different numbers of components.
        """
        train_scores = []
        test_scores = []
        components_list = []

        print("\nComparing different numbers of components...")
        for n_comp in component_range:
            if n_comp > self.X_train_scaled.shape[1]:
                continue

            print(f"  Testing {n_comp} components...")
            # Apply PCA
            pca_temp = PCA(n_components=n_comp)
            X_train_temp = pca_temp.fit_transform(self.X_train_scaled)
            X_test_temp = pca_temp.transform(self.X_test_scaled)

            # Train classifier
            clf_temp = LogisticRegression(max_iter=1000, random_state=42, verbose=0)
            clf_temp.fit(X_train_temp, self.y_train)

            train_scores.append(clf_temp.score(X_train_temp, self.y_train))
            test_scores.append(clf_temp.score(X_test_temp, self.y_test))
            components_list.append(n_comp)

        # Plot results
        plt.figure(figsize=(12, 7))
        plt.plot(components_list, train_scores, 'o-', label='Training Accuracy',
                linewidth=2, markersize=8)
        plt.plot(components_list, test_scores, 's-', label='Testing Accuracy',
                linewidth=2, markersize=8)
        plt.xlabel('Number of Principal Components', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance vs. Number of Components', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return components_list, train_scores, test_scores

    def plot_misclassified(self, n_samples=20, save_path=None):
        """Plot misclassified examples."""
        # Find misclassified indices
        misclassified_idx = np.where(self.y_test != self.y_test_pred)[0]

        if len(misclassified_idx) == 0:
            print("No misclassified samples!")
            return

        n_samples = min(n_samples, len(misclassified_idx))

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        for i in range(n_samples):
            idx = misclassified_idx[i]
            img = self.X_test[idx].reshape(28, 28)

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {self.y_test[idx]} | Pred: {self.y_test_pred[idx]}',
                            color='red', fontsize=11)
            axes[i].axis('off')

        plt.suptitle('Misclassified Examples', fontsize=16, y=0.995, color='red')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("MNIST Handwritten Digit Recognition with PCA")
    print("="*70)

    # Initialize model (using 10k samples for faster processing)
    # Set sample_size=None to use full 70k MNIST dataset
    model = MNISTRecognitionPCA(sample_size=10000)

    # Load and prepare data
    print("\n[1] Loading MNIST Data...")
    model.load_data()

    # Plot sample digits
    print("\n[2] Visualizing Sample Digits...")
    model.plot_sample_digits(save_path='results/mnist_sample_digits.png')

    # Preprocess data
    print("\n[3] Preprocessing Data...")
    model.preprocess()

    # Explore PCA - start with fewer components for speed
    print("\n[4] Analyzing Principal Components...")
    model.apply_pca(n_components=200)
    model.plot_variance_explained(max_components=100,
                                 save_path='results/mnist_variance_explained.png')
    model.plot_pca_components(n_components=16,
                             save_path='results/mnist_pca_components.png')

    # 2D visualization
    print("\n[5] Creating 2D Projection...")
    model.plot_2d_projection(save_path='results/mnist_2d_projection.png')

    # Compare different numbers of components
    print("\n[6] Comparing Different Dimensions...")
    model.compare_dimensions(component_range=[10, 30, 50, 100, 150],
                           save_path='results/mnist_dimension_comparison.png')

    # Train with optimal components
    print("\n[7] Training Final Model with Optimal Components...")
    model.apply_pca(n_components=100)  # Good balance of performance and speed
    model.train_classifier(use_pca=True)

    # Evaluate model
    print("\n[8] Evaluating Model Performance...")
    model.plot_confusion_matrix(save_path='results/mnist_confusion_matrix.png')
    model.plot_misclassified(save_path='results/mnist_misclassified.png')

    print("\n" + "="*70)
    print("Analysis Complete! Check the results folder for visualizations.")
    print("="*70)


if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)

    main()
