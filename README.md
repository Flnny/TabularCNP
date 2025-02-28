# TabularCNP: Real Estate Valuation using Deep Learning and Conditional Neural Processes

## Overview
TabularCNP is a novel framework designed for real estate valuation. It leverages deep learning models applied to tabular data and Conditional Neural Processes (CNPs) to address the challenges of modeling multifactor interactions and spatiotemporal dependencies in property price prediction. The approach incorporates feature interaction models for capturing complex feature relationships and a CNP-based event logging module to capture spatiotemporal dynamics by conditioning the property value on nearby recent sales.

## Project Structure

```plaintext
TabularCNP/                          # Root directory of the project
│
├── datasets/                       
│   ├── __init__.py            
│   ├── ar_dataloader.py             # Data loader for ARCNP framework 
│   ├── dataloader.py                # CNP data loader and preprocessing for TabularCNP framework
│   └── dataset.py                   # Dataset definitions and management for CNP and ARCNP
│
├── model/                          
│   ├── __init__.py                  
│   ├── base.py                      # Base class for models in the TabularCNP framework
│   ├── cDeepFM.py                   # DeepFM* model in the TabularCNP framework 
│   ├── cDeepFM_at.py                # DeepFM* with Attention as the aggregation function 
│   ├── cDeepFM_deepest.py           # DeepFM* with DeepSet as the aggregation function 
│   ├──cWDL.py                       # CTabNet* model in the TabularCNP framework 
│   └── cWDL.py                      # CWDL* model in the TabularCNP framework 
│
├── tools/                           
│   ├── __init__.py                  
│   ├── data_view.py                 # Visualization utilities to analyze experiment results
│   └── main.py                      # Main script to run and manage experiments
│
└── README.md                        # Project documentation (this file)
```

## Code Updates
The code is available at https://anonymous.4open.science/r/TabularCNP-EE04/. Stay tuned for future updates and improvements!