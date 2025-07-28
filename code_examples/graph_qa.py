from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer


class GraphQA:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize GraphQA with a pre-trained language model
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.kg = nx.Graph()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def build_knowledge_graph(self, triples: List[tuple]):
        """Build knowledge graph from (subject, relation, object) triples
        
        Args:
            triples: List of (subject, relation, object) tuples
        """
        for s, r, o in triples:
            self.kg.add_edge(s, o, relation=r)
            
    def get_bert_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for a text string
        
        Args:
            text: Input text to embed
            
        Returns:
            torch.Tensor: BERT embedding
        """
        inputs = self.tokenizer(text, return_tensors="pt", 
                              padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
            
    def decompose_question(self, question: str) -> List[str]:
        """Break complex question into simpler sub-questions
        
        Args:
            question: Complex input question
            
        Returns:
            List of simpler sub-questions
        """
        # Simple rule-based decomposition for demo
        if "and" in question.lower():
            parts = question.split(" and ")
            return [p.strip() + "?" for p in parts]
        return [question]
        
    def identify_entities(self, text: str) -> List[str]:
        """Identify entities in text that exist in knowledge graph
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of identified entity strings
        """
        entities = []
        for node in self.kg.nodes():
            if str(node).lower() in text.lower():
                entities.append(node)
        return entities
        
    def rank_paths(self, paths: List[List[str]], 
                  question: str) -> List[str]:
        """Rank paths by relevance to question
        
        Args:
            paths: List of possible paths through knowledge graph
            question: Original question for relevance scoring
            
        Returns:
            Most relevant path
        """
        if not paths:
            return []
            
        # Get question embedding
        q_emb = self.get_bert_embedding(question)
        
        # Score paths
        scores = []
        for path in paths:
            # Get mean embedding of path
            path_emb = torch.stack([
                self.get_bert_embedding(str(node)) 
                for node in path
            ]).mean(dim=0)
            
            # Score is cosine similarity
            score = torch.cosine_similarity(q_emb, path_emb, dim=0)
            scores.append(score.item())
            
        # Return path with highest score
        return paths[np.argmax(scores)]
        
    def extract_answer(self, path: List[str]) -> str:
        """Convert path to natural language answer
        
        Args:
            path: Knowledge graph path containing answer
            
        Returns:
            Natural language answer
        """
        if not path:
            return "I could not find an answer to this question."
            
        # Simple template-based answer generation
        answer = ""
        for i in range(len(path)-1):
            s = path[i]
            o = path[i+1]
            r = self.kg[s][o]["relation"]
            answer += f"{s} {r} {o}. "
            
        return answer.strip()
        
    def answer_question(self, question: str) -> str:
        """Answer a question using the knowledge graph
        
        Args:
            question: Question to answer
            
        Returns:
            Natural language answer
        """
        # 1. Decompose complex question
        sub_questions = self.decompose_question(question)
        
        # 2. Answer each sub-question
        answers = []
        for sub_q in sub_questions:
            # Identify start nodes
            start_entities = self.identify_entities(sub_q)
            
            # Find paths in KG
            paths = []
            for entity in start_entities:
                paths.extend([
                    list(p) for p in nx.single_source_shortest_path(
                        self.kg, entity, cutoff=2).values()
                ])
            
            # Score and select best path
            best_path = self.rank_paths(paths, sub_q)
            answers.append(self.extract_answer(best_path))
            
        return " ".join(answers)
