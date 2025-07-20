#!/bin/bash

# Activate the correct conda environment
conda activate MLHW

# Run the Streamlit app
streamlit run app.py --server.port 8503

echo "✅ Zero-Click Compass dashboard is running at http://localhost:8503" 