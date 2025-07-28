1. `integrated_rag_test.py`: Contains comprehensive test cases that verify:
   - GraphQA functionality
   - Matrix Factorization functionality
   - Integration of both systems for personalized responses
   - Cold-start problem handling
   - Multi-hop reasoning capabilities

2. `graph_qa.py`: Implements the GraphQA system with:
   - Knowledge graph construction and management
   - Question decomposition
   - Path-based reasoning
   - BERT embeddings for semantic matching
   - Natural language answer generation

3. `matrix_factorization.py`: Implements the Matrix Factorization system with:
   - Latent factor model training
   - User-item prediction
   - Recommendation generation
   - Cold-start handling
   - Regularization to prevent overfitting

Key test scenarios covered:

1. Knowledge Graph Construction and Querying:
```python
def test_knowledge_graph_construction(self):
    """Test if knowledge graph is properly constructed"""
    self.assertEqual(len(self.graph_qa.kg.nodes()), 6)
    self.assertEqual(len(self.graph_qa.kg.edges()), 4)
```

2. Question Decomposition:
```python
def test_question_decomposition(self):
    """Test complex question decomposition"""
    question = "Who created Python and where do they work?"
    sub_questions = self.graph_qa.decompose_question(question)
    self.assertEqual(len(sub_questions), 2)
```

3. Matrix Factorization Training:
```python
def test_matrix_factorization_training(self):
    """Test if MF training improves predictions"""
    initial_rmse = self._calculate_rmse()
    self.mf.fit(self.ratings, n_epochs=20)
    final_rmse = self._calculate_rmse()
    self.assertLess(final_rmse, initial_rmse)
```

4. Integrated System Testing:
```python
def test_personalized_response_generation(self):
    """Test if system generates personalized responses"""
    question = "Who created Python?"
    user_id = 0  # Technical user
    base_response = self.graph_qa.answer_question(question)
    preferred_style = self._get_preferred_style(user_id)
    personalized_response = self._personalize_response(
        base_response, preferred_style)
```

To run the tests:
1. Install dependencies:
```bash
pip install torch torch_geometric transformers networkx numpy scipy
```

2. Run tests:
```bash
python -m unittest integrated_rag_test.py
```

These tests validate the key concepts from the article:
- Multi-hop reasoning in GraphQA
- Personalization through Matrix Factorization
- Cold-start problem handling
- Integration of structured knowledge with user preferences
