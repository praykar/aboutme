from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix


class MatrixFactorization:
    def __init__(self, n_factors: int = 20, 
                 learning_rate: float = 0.01,
                 regularization: float = 0.02):
        """Initialize Matrix Factorization model
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: L2 regularization parameter
        """
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings: csr_matrix, n_epochs: int = 20):
        """Train MF model on sparse rating matrix
        
        Args:
            ratings: User-item rating matrix (sparse)
            n_epochs: Number of training epochs
        """
        n_users, n_items = ratings.shape
        
        # Initialize latent factors
        if self.user_factors is None:
            self.user_factors = np.random.normal(
                0, 0.1, (n_users, self.n_factors))
        if self.item_factors is None:
            self.item_factors = np.random.normal(
                0, 0.1, (n_items, self.n_factors))
        
        # Train using SGD
        for epoch in range(n_epochs):
            epoch_loss = 0
            update_count = 0
            
            # Iterate over observed ratings
            for u, i in zip(*ratings.nonzero()):
                # Compute current prediction
                r_ui = ratings[u, i]
                pred = np.dot(self.user_factors[u], self.item_factors[i])
                error = r_ui - pred
                
                # Update user factors
                u_factors = self.user_factors[u].copy()
                i_factors = self.item_factors[i].copy()
                
                self.user_factors[u] += self.lr * (error * i_factors - 
                                                  self.reg * u_factors)
                self.item_factors[i] += self.lr * (error * u_factors - 
                                                  self.reg * i_factors)
                
                # Accumulate loss
                epoch_loss += error ** 2
                update_count += 1
            
            # Compute average loss
            avg_loss = np.sqrt(epoch_loss / update_count)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: RMSE = {avg_loss:.4f}")
                
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for given user-item pairs
        
        Args:
            user_id: User ID
            item_ids: List of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if self.user_factors is None:
            raise ValueError("Model must be trained before making predictions")
            
        return np.dot(self.user_factors[user_id], 
                     self.item_factors[item_ids].T)
        
    def recommend(self, user_id: int, n_items: int = 5,
                 exclude_rated: bool = True) -> List[int]:
        """Get top-N recommendations for user
        
        Args:
            user_id: User ID
            n_items: Number of items to recommend
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of recommended item IDs
        """
        if self.user_factors is None:
            raise ValueError("Model must be trained before recommending")
            
        # Get all predictions for user
        predictions = self.predict(
            user_id, 
            list(range(self.item_factors.shape[0]))
        )
        
        # Sort by predicted rating
        item_scores = list(enumerate(predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N items
        return [item for item, _ in item_scores[:n_items]]
