import unittest
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer


class TestGraphQA(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.graph_qa = GraphQA(model_name="bert-base-uncased")
        
        # Create test knowledge graph
        self.test_triples = [
            ("Python", "created_by", "Guido van Rossum"),
            ("Guido van Rossum", "works_at", "Microsoft"),
            ("Python", "is_a", "Programming Language"),
            ("Programming Language", "used_in", "Software Development")
        ]
        self.graph_qa.build_knowledge_graph(self.test_triples)

    def test_knowledge_graph_construction(self):
        """Test if knowledge graph is properly constructed"""
        self.assertEqual(len(self.graph_qa.kg.nodes()), 6)
        self.assertEqual(len(self.graph_qa.kg.edges()), 4)
        
        # Test node existence
        self.assertTrue("Python" in self.graph_qa.kg)
        self.assertTrue("Guido van Rossum" in self.graph_qa.kg)
        
        # Test edge existence
        self.assertTrue(self.graph_qa.kg.has_edge("Python", "Guido van Rossum"))

    def test_question_decomposition(self):
        """Test complex question decomposition"""
        question = "Who created Python and where do they work?"
        sub_questions = self.graph_qa.decompose_question(question)
        
        self.assertEqual(len(sub_questions), 2)
        self.assertIn("Who created Python?", sub_questions)
        self.assertIn("Where does that person work?", sub_questions)

    def test_multi_hop_reasoning(self):
        """Test multi-hop reasoning capabilities"""
        question = "Where does Python's creator work?"
        answer = self.graph_qa.answer_question(question)
        
        self.assertIn("Microsoft", answer)
        self.assertIn("Guido van Rossum", answer)


class TestMatrixFactorization(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.n_users = 100
        self.n_responses = 50
        self.n_factors = 10
        self.mf = MatrixFactorization(n_factors=self.n_factors)
        
        # Create synthetic test data
        self.ratings = self._create_test_data()

    def _create_test_data(self) -> csr_matrix:
        """Create synthetic user-response interaction data"""
        # Create sparse rating matrix with some known patterns
        data = []
        rows = []
        cols = []
        
        # Add some consistent user preferences
        for user in range(self.n_users):
            # Each user likes responses in their preferred category
            preferred_category = user % 5
            for response in range(self.n_responses):
                if response % 5 == preferred_category:
                    rating = np.random.normal(4.5, 0.5)  # High ratings for preferred
                elif abs(response % 5 - preferred_category) <= 1:
                    rating = np.random.normal(3.0, 0.5)  # Medium for similar
                else:
                    if np.random.random() < 0.1:  # Sparse matrix
                        rating = np.random.normal(2.0, 0.5)  # Low for others
                        
                if 'rating' in locals():
                    data.append(max(1, min(5, rating)))  # Clip to [1, 5]
                    rows.append(user)
                    cols.append(response)
                    
        return csr_matrix((data, (rows, cols)), 
                         shape=(self.n_users, self.n_responses))

    def test_matrix_factorization_training(self):
        """Test if MF training improves predictions"""
        # Get initial predictions
        initial_rmse = self._calculate_rmse()
        
        # Train model
        self.mf.fit(self.ratings, n_epochs=20)
        
        # Get final predictions
        final_rmse = self._calculate_rmse()
        
        # Assert that training improved predictions
        self.assertLess(final_rmse, initial_rmse)
        
    def test_cold_start_handling(self):
        """Test handling of new users/responses"""
        # Train on subset of users
        train_users = self.ratings[:80]
        self.mf.fit(train_users, n_epochs=20)
        
        # Test predictions for new user
        new_user_ratings = self.ratings[90].toarray()[0]
        active_items = np.where(new_user_ratings > 0)[0]
        
        predictions = self.mf.predict(90, active_items)
        
        # Assert predictions are reasonable (between 1 and 5)
        self.assertTrue(np.all(predictions >= 1))
        self.assertTrue(np.all(predictions <= 5))

    def _calculate_rmse(self) -> float:
        """Calculate RMSE for current predictions"""
        errors = []
        for u, i in zip(*self.ratings.nonzero()):
            pred = self.mf.predict(u, [i])[0]
            true = self.ratings[u, i]
            errors.append((pred - true) ** 2)
        return np.sqrt(np.mean(errors))


class TestIntegratedSystem(unittest.TestCase):
    def setUp(self):
        """Initialize integrated test environment"""
        self.graph_qa = GraphQA(model_name="bert-base-uncased")
        self.mf = MatrixFactorization(n_factors=10)
        
        # Setup knowledge graph
        self.graph_qa.build_knowledge_graph([
            ("Python", "created_by", "Guido van Rossum"),
            ("JavaScript", "created_by", "Brendan Eich"),
            ("Java", "created_by", "James Gosling")
        ])
        
        # Setup user preferences
        self.user_preferences = self._create_user_preferences()
        self.mf.fit(self.user_preferences, n_epochs=20)

    def _create_user_preferences(self) -> csr_matrix:
        """Create synthetic user preferences for different response types"""
        n_users = 10
        n_response_types = 5  # Different types of responses
        
        data = []
        rows = []
        cols = []
        
        # Create preferences based on user "personality"
        for user in range(n_users):
            prefers_technical = user % 2 == 0
            for response in range(n_response_types):
                if (prefers_technical and response < 3) or \
                   (not prefers_technical and response >= 3):
                    rating = np.random.normal(4.5, 0.5)
                else:
                    rating = np.random.normal(2.5, 0.5)
                    
                data.append(max(1, min(5, rating)))
                rows.append(user)
                cols.append(response)
                
        return csr_matrix((data, (rows, cols)), 
                         shape=(n_users, n_response_types))

    def test_personalized_response_generation(self):
        """Test if system generates personalized responses"""
        question = "Who created Python?"
        user_id = 0  # Technical user
        
        # Get base response from GraphQA
        base_response = self.graph_qa.answer_question(question)
        
        # Get user's preferred response style
        preferred_style = self._get_preferred_style(user_id)
        
        # Generate personalized response
        personalized_response = self._personalize_response(
            base_response, preferred_style)
        
        # Assert response contains key information
        self.assertIn("Guido van Rossum", personalized_response)
        
        # Assert response matches user preference
        if user_id % 2 == 0:  # Technical user
            self.assertIn("programming language", personalized_response.lower())
        else:  # Non-technical user
            self.assertIn("created", personalized_response.lower())

    def _get_preferred_style(self, user_id: int) -> int:
        """Get user's preferred response style based on MF predictions"""
        predictions = self.mf.predict(user_id, list(range(5)))
        return np.argmax(predictions)

    def _personalize_response(self, 
                            base_response: str, 
                            style: int) -> str:
        """Adjust response based on preferred style"""
        templates = {
            0: "{} is a technical detail about programming languages.",
            1: "{} involves software development concepts.",
            2: "{} is related to computer science.",
            3: "{} is an interesting fact about software.",
            4: "{} is something you might want to know about coding."
        }
        return templates[style].format(base_response)


if __name__ == '__main__':
    unittest.main()
