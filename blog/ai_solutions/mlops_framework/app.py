import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

st.set_page_config(page_title="MLOps Framework Demo", layout="wide")

st.title("⚙️ MLOps Pipeline Walkthrough")
st.markdown(
    """A minimal MLOps framework demonstration showing the key stages of an ML lifecycle: 
    data versioning, model training, experiment tracking, deployment, and monitoring."""
)

# Initialize session state for pipeline tracking
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'data_version': None,
        'experiments': [],
        'deployed_model': None,
        'monitoring_logs': []
    }

# Sidebar navigation
st.sidebar.header("Pipeline Stages")
stage = st.sidebar.radio(
    "Select Stage",
    ["1. Data Versioning", "2. Model Training", "3. Experiment Tracking", 
     "4. Model Deployment", "5. Monitoring"]
)

# Helper functions
def generate_sample_data(n_samples=1000, version="v1.0"):
    """Generate synthetic dataset"""
    np.random.seed(hash(version) % (2**32))
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3) > 0
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y.astype(int)
    return df

def train_model(data, model_type="LogisticRegression"):
    """Simulate model training"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    X = data[[c for c in data.columns if c != 'target']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

# Stage 1: Data Versioning
if stage == "1. Data Versioning":
    st.header("🗂️ Data Versioning")
    st.markdown("Track and manage different versions of training data for reproducibility.")
    
    col1, col2 = st.columns(2)
    with col1:
        data_version = st.text_input("Data Version", value="v1.0")
        n_samples = st.slider("Number of Samples", 100, 5000, 1000)
    
    if st.button("Create Data Version", type="primary"):
        df = generate_sample_data(n_samples, data_version)
        st.session_state.pipeline_state['data_version'] = {
            'version': data_version,
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'data': df
        }
        st.success(f"Data version {data_version} created with {n_samples} samples")
    
    if st.session_state.pipeline_state['data_version']:
        st.subheader("Current Data Version")
        dv = st.session_state.pipeline_state['data_version']
        col1, col2, col3 = st.columns(3)
        col1.metric("Version", dv['version'])
        col2.metric("Samples", dv['n_samples'])
        col3.metric("Features", len(dv['data'].columns) - 1)
        
        if st.checkbox("Show data sample"):
            st.dataframe(dv['data'].head(20), use_container_width=True)

# Stage 2: Model Training
elif stage == "2. Model Training":
    st.header("🏋️ Model Training")
    st.markdown("Train models with different hyperparameters and algorithms.")
    
    if not st.session_state.pipeline_state['data_version']:
        st.warning("Please create a data version first (Stage 1)")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", ["LogisticRegression", "RandomForest"])
            experiment_name = st.text_input("Experiment Name", value=f"exp_{len(st.session_state.pipeline_state['experiments']) + 1}")
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                data = st.session_state.pipeline_state['data_version']['data']
                results = train_model(data, model_type)
                
                experiment = {
                    'name': experiment_name,
                    'model_type': model_type,
                    'data_version': st.session_state.pipeline_state['data_version']['version'],
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'accuracy': results['accuracy'],
                        'precision': results['precision'],
                        'recall': results['recall']
                    },
                    'model': results['model']
                }
                st.session_state.pipeline_state['experiments'].append(experiment)
                st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.3f}")
        
        if st.session_state.pipeline_state['experiments']:
            st.subheader("Training Results")
            latest = st.session_state.pipeline_state['experiments'][-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{latest['metrics']['accuracy']:.3f}")
            col2.metric("Precision", f"{latest['metrics']['precision']:.3f}")
            col3.metric("Recall", f"{latest['metrics']['recall']:.3f}")

# Stage 3: Experiment Tracking
elif stage == "3. Experiment Tracking":
    st.header("📋 Experiment Tracking")
    st.markdown("Compare multiple experiments to select the best model.")
    
    if not st.session_state.pipeline_state['experiments']:
        st.warning("No experiments yet. Please train a model first (Stage 2)")
    else:
        experiments_df = pd.DataFrame([{
            'Name': exp['name'],
            'Model Type': exp['model_type'],
            'Data Version': exp['data_version'],
            'Accuracy': exp['metrics']['accuracy'],
            'Precision': exp['metrics']['precision'],
            'Recall': exp['metrics']['recall'],
            'Timestamp': exp['timestamp']
        } for exp in st.session_state.pipeline_state['experiments']])
        
        st.dataframe(experiments_df, use_container_width=True)
        
        # Best model
        best_idx = experiments_df['Accuracy'].idxmax()
        st.info(f"Best model: {experiments_df.loc[best_idx, 'Name']} (Accuracy: {experiments_df.loc[best_idx, 'Accuracy']:.3f})")

# Stage 4: Model Deployment
elif stage == "4. Model Deployment":
    st.header("🚀 Model Deployment")
    st.markdown("Deploy the selected model to a production environment.")
    
    if not st.session_state.pipeline_state['experiments']:
        st.warning("No trained models available. Please train a model first (Stage 2)")
    else:
        model_names = [exp['name'] for exp in st.session_state.pipeline_state['experiments']]
        selected_model = st.selectbox("Select Model to Deploy", model_names)
        
        if st.button("Deploy Model", type="primary"):
            model_idx = model_names.index(selected_model)
            deployed_exp = st.session_state.pipeline_state['experiments'][model_idx]
            st.session_state.pipeline_state['deployed_model'] = {
                'experiment': deployed_exp,
                'deployment_time': datetime.now().isoformat(),
                'status': 'active'
            }
            st.success(f"Model {selected_model} deployed successfully!")
        
        if st.session_state.pipeline_state['deployed_model']:
            st.subheader("Deployed Model Info")
            deployed = st.session_state.pipeline_state['deployed_model']
            col1, col2, col3 = st.columns(3)
            col1.metric("Model", deployed['experiment']['name'])
            col2.metric("Status", deployed['status'])
            col3.metric("Accuracy", f"{deployed['experiment']['metrics']['accuracy']:.3f}")

# Stage 5: Monitoring
elif stage == "5. Monitoring":
    st.header("📡 Model Monitoring")
    st.markdown("Track model performance and detect drift in production.")
    
    if not st.session_state.pipeline_state['deployed_model']:
        st.warning("No deployed model. Please deploy a model first (Stage 4)")
    else:
        st.subheader("Deployed Model Status")
        deployed = st.session_state.pipeline_state['deployed_model']
        col1, col2 = st.columns(2)
        col1.metric("Model Name", deployed['experiment']['name'])
        col2.metric("Status", deployed['status'])
        
        # Simulate predictions
        if st.button("Generate Monitoring Data"):
            # Generate synthetic monitoring log
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'predictions': np.random.randint(50, 200),
                'avg_confidence': np.random.uniform(0.7, 0.95),
                'latency_ms': np.random.uniform(10, 50)
            }
            st.session_state.pipeline_state['monitoring_logs'].append(log_entry)
            st.success("Monitoring data generated")
        
        if st.session_state.pipeline_state['monitoring_logs']:
            st.subheader("Recent Monitoring Metrics")
            logs_df = pd.DataFrame(st.session_state.pipeline_state['monitoring_logs'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", int(logs_df['predictions'].sum()))
            col2.metric("Avg Confidence", f"{logs_df['avg_confidence'].mean():.3f}")
            col3.metric("Avg Latency (ms)", f"{logs_df['latency_ms'].mean():.1f}")
            
            st.dataframe(logs_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "This is a simplified MLOps demo. Production systems should use tools like MLflow, "
    "Kubeflow, or custom orchestration with DVC, Airflow, and monitoring platforms."
)
