import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
#import seaborn as sns

class ProteinRankingModel:
    """
    Random Forest model to predict protein structure sample rankings (0-49)
    based on 1D features from Boltz2 model outputs.
    
    Dataset structure:
    - 40 proteins total
    - 50 samples per protein (2000 total samples)
    - Each sample has features and a rank (0-49) based on RMSD
    """
    
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        """
        Initialize the Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.feature_names = None
        self.is_fitted = False
        
    def prepare_data(self, features, ranks, protein_ids=None):
        """
        Prepare data for training/testing.
        
        Parameters:
        -----------
        features : array-like, shape (n_samples, n_features)
            Feature matrix (2000 samples x n_features)
        ranks : array-like, shape (n_samples,)
            Ranking values (0-49) for each sample
        protein_ids : array-like, shape (n_samples,), optional
            Protein identifiers for each sample
            
        Returns:
        --------
        X : numpy array
            Feature matrix
        y : numpy array
            Target rankings
        """
        X = np.array(features)
        y = np.array(ranks)
        
        # Validate input shapes
        assert X.shape[0] == y.shape[0], "Features and ranks must have same number of samples"
        assert y.min() >= 0 and y.max() <= 49, "Ranks must be in range [0, 49]"
        
        if protein_ids is not None:
            assert len(protein_ids) == X.shape[0], "protein_ids must match number of samples"
        
        return X, y
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the Random Forest model.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Training rankings
        feature_names : list, optional
            Names of features for interpretability
        """
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate training score
        train_score = self.model.score(X_train, y_train)
        print(f"Training R² score: {train_score:.4f}")
        
    def predict(self, X):
        """
        Predict rankings for new samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features for prediction
            
        Returns:
        --------
        predictions : numpy array
            Predicted rankings (clipped to [0, 49])
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        # Clip predictions to valid rank range
        predictions = np.clip(predictions, 0, 49)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            True test rankings
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mean_rank_error': np.mean(y_test - y_pred)
        }
        
        print("\n=== Model Evaluation ===")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f} ranks")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f} ranks")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Mean Rank Error: {metrics['mean_rank_error']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, plot=True):
        """
        Get and optionally plot feature importances.
        
        Parameters:
        -----------
        plot : bool
            Whether to plot feature importances
            
        Returns:
        --------
        importance_df : pandas DataFrame
            Feature importances sorted by value
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 6))
            top_n = min(20, len(importance_df))  # Plot top 20 features
            print(importance_df.head(top_n))
            #sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
            #plt.title('Top Feature Importances')
            #plt.xlabel('Importance')
            #plt.tight_layout()
            #plt.show()
        
        return importance_df
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Rankings
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        cv_scores : dict
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_scores = -scores
        
        cv_scores = {
            'cv_mae_mean': mae_scores.mean(),
            'cv_mae_std': mae_scores.std(),
            'cv_scores': mae_scores
        }
        
        print(f"\nCross-Validation (k={cv}):")
        print(f"Mean MAE: {cv_scores['cv_mae_mean']:.4f} ± {cv_scores['cv_mae_std']:.4f}")
        
        return cv_scores


# Example usage
def example_usage():
    """
    Example of how to use the ProteinRankingModel with synthetic data.
    """
    # Generate synthetic data
    # 40 proteins, 50 samples each = 2000 total samples
    n_proteins = 40
    n_samples_per_protein = 50
    n_features = 128  # Example: 128-dimensional features
    
    np.random.seed(42)
    
    # Create synthetic features and rankings
    all_features = []
    all_ranks = []
    all_protein_ids = []
    
    for protein_id in range(n_proteins):
        # Generate features for this protein's 50 samples
        protein_features = np.random.randn(n_samples_per_protein, n_features)
        
        # Generate rankings 0-49 for this protein
        protein_ranks = np.arange(n_samples_per_protein)
        
        all_features.append(protein_features)
        all_ranks.extend(protein_ranks)
        all_protein_ids.extend([protein_id] * n_samples_per_protein)
    
    X = np.vstack(all_features)
    y = np.array(all_ranks)
    protein_ids = np.array(all_protein_ids)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of proteins: {n_proteins}")
    print(f"Samples per protein: {n_samples_per_protein}")
    print(f"Total samples: {len(y)}")
    
    # Split data (stratify by protein to ensure each protein is represented)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = ProteinRankingModel(n_estimators=200, max_depth=20)
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Cross-validation
    model.cross_validate(X_train, y_train, cv=5)
    
    # Feature importance
    importance_df = model.get_feature_importance(plot=False)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Make predictions
    sample_predictions = model.predict(X_test[:])
    # Convert predictions (RMSD) → ranking (0 = best)
    import numpy as np
    predicted_ranks = np.argsort(np.argsort(sample_predictions))

    print(f"\nSample predictions (RMSD): {sample_predictions}")
    print(f"Predicted ranks: {predicted_ranks}")
    print(f"True ranks: {y_test[:]}")


if __name__ == "__main__":
    example_usage()
