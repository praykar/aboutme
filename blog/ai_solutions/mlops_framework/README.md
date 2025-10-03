# MLOps Framework Demo (Streamlit)

This directory contains a minimal MLOps pipeline walkthrough demonstrating the key stages of an ML lifecycle: data versioning, model training, experiment tracking, deployment, and monitoring.

## Quick start

* **Python 3.9+**
* **Install deps:**

```bash
pip install -U streamlit pandas numpy scikit-learn
streamlit run app.py
```

## Architecture

```
Data Versioning --> Model Training --> Experiment Tracking
       |                 |                     |
       v                 v                     v
   [Session State] <-- [Pipeline Controller] --> [Metrics Storage]
                              |
                              v
                    Model Deployment --> Monitoring
```

* **Session-based state**: Simulates pipeline state across stages
* **5 stages**: Data versioning, training, tracking, deployment, monitoring
* **Simple models**: LogisticRegression and RandomForest for demo

## Technical Summary

### Pipeline Stages

1. **Data Versioning**: Create and track dataset versions
2. **Model Training**: Train models with different algorithms/hyperparameters
3. **Experiment Tracking**: Compare experiments, select best model
4. **Model Deployment**: Deploy selected model to "production"
5. **Monitoring**: Track predictions, confidence, and latency

## Files

* **app.py**: Streamlit app with 5-stage MLOps pipeline walkthrough
* **README.md**: This file

## Demo Behavior

1. Navigate through 5 stages using sidebar
2. Create data versions with synthetic datasets
3. Train models and view metrics (accuracy, precision, recall)
4. Compare experiments in tracking stage
5. Deploy best model
6. Generate and view monitoring metrics

## References

* **MLflow**: [https://mlflow.org/](https://mlflow.org/)
* **Kubeflow**: [https://www.kubeflow.org/](https://www.kubeflow.org/)
* **DVC (Data Version Control)**: [https://dvc.org/](https://dvc.org/)
* **ML Engineering Book**: Andriy Burkov. [http://www.mlebook.com/](http://www.mlebook.com/)

## Notes for Hugging Face Space Integration

* **Add requirements.txt**: streamlit, pandas, numpy, scikit-learn
* **Lightweight**: CPU-only, no external dependencies
* **Multi-demo integration**: Expose as tab in unified launcher
* **Production alternatives**: MLflow, Kubeflow, AWS SageMaker Pipelines, Vertex AI
