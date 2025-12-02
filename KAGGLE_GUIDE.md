# Multi-Model Segmentation Pipeline - Kaggle Guide

## Adding Python Files to a Kaggle Notebook Kernel

### Step 1: Add Your .py Files as a Dataset

1. **Prepare Your .py Files:**
   - Download `ann-task1-multi-segmentation-pipeline.py` from this repository
   - Optionally include `models.py` if needed

2. **Create a Dataset on Kaggle:**
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload your `.py` files
   - Name it appropriately (e.g., `ann-segmentation-pipeline`)
   - Set visibility (public or private)
   - Click "Create"

### Step 2: Copy the Path to Your Dataset Folder

1. **Locate Your Dataset:**
   - Go to the "Datasets" section
   - Find the dataset you just created
   - Click on the dataset to open it
   - Look for the dataset path. It usually looks something like:
     ```
     /kaggle/input/ann-segmentation-pipeline
     ```

### Step 3: Add the Dataset to Your Notebook

1. **Open Your Notebook:**
   - Go to your Kaggle notebook (or create a new one)
   - Click "Add data" on the right sidebar
   - Search for your dataset
   - Click "Add" to include it in your notebook

### Step 4: Insert the Path in Your Notebook

1. **Add the Path to Python:**
   In the first cell of your notebook, add:
   ```python
   import sys
   sys.path.insert(1, '/kaggle/input/ann-segmentation-pipeline')
   ```

2. **Import the Pipeline:**
   In the same cell, import the modules:
   ```python
   import sys
   sys.path.insert(1, '/kaggle/input/ann-segmentation-pipeline')
   
   # Import the pipeline
   from ann_task1_multi_segmentation_pipeline import run_pipeline
   ```

### Step 5: Configure and Run the Pipeline

1. **Update Configuration (if needed):**
   The pipeline has default paths for Kaggle. If your data paths differ, modify them:
   ```python
   import sys
   sys.path.insert(1, '/kaggle/input/ann-segmentation-pipeline')
   
   # Import and modify configuration
   import ann_task1_multi_segmentation_pipeline as pipeline
   
   # Update paths if needed
   pipeline.DATA_DIR = "/kaggle/input/your-dataset/training_images"
   pipeline.EXCEL_PATH = "/kaggle/input/your-dataset/training_metadata.xlsx"
   pipeline.OUTPUT_DIR = "/kaggle/working/"
   
   # Update hyperparameters if needed
   pipeline.NUM_EPOCHS_SEGMENTATION = 30  # Reduce for faster testing
   pipeline.BATCH_SIZE = 16
   
   # Run the pipeline
   results = pipeline.run_pipeline()
   ```

2. **Or Run with Default Settings:**
   ```python
   import sys
   sys.path.insert(1, '/kaggle/input/ann-segmentation-pipeline')
   
   from ann_task1_multi_segmentation_pipeline import run_pipeline
   results = run_pipeline()
   ```

### Step 6: View Results

After the pipeline completes:

1. **Results CSV:**
   - Located at `/kaggle/working/segmentation_models_comparison.csv`
   - Contains test Dice and IoU scores for all models

2. **Training Curves:**
   - Saved at `/kaggle/working/training_curves.png`

3. **Model Weights:**
   - Saved in `/kaggle/working/` as `best_<model_name>.pth`

### Example Complete Notebook Cell

```python
# Cell 1: Setup
import sys
sys.path.insert(1, '/kaggle/input/ann-segmentation-pipeline')

# Install required packages
!pip install -q segmentation-models-pytorch

# Cell 2: Run Pipeline
import ann_task1_multi_segmentation_pipeline as pipeline

# Configure (optional)
pipeline.DATA_DIR = "/kaggle/input/tumor-segmentation-ai/training_images/training_images"
pipeline.EXCEL_PATH = "/kaggle/input/tumor-segmentation-ai/training_metadata.xlsx"
pipeline.OUTPUT_DIR = "/kaggle/working/"
pipeline.NUM_EPOCHS_SEGMENTATION = 50
pipeline.BATCH_SIZE = 16

# Run
results = pipeline.run_pipeline()

# Cell 3: Display Results
import pandas as pd
results_df = pd.read_csv("/kaggle/working/segmentation_models_comparison.csv")
display(results_df)

# Show training curves
from IPython.display import Image
Image("/kaggle/working/training_curves.png")
```

### Models Tested

The pipeline tests these segmentation models:
- `BasicUNet` - U-Net from scratch
- `AttentionUNet` - Attention U-Net (custom implementation)
- `SMP_UNet_ResNet50` - U-Net with ResNet50 encoder
- `SMP_MAnet_MiTB0` - MAnet with MiT-B0 encoder
- `SMP_Segformer_MiTB0` - Segformer with MiT-B0 encoder
- `SMP_DeepLabV3Plus_ResNet50` - DeepLabV3+ with ResNet50 encoder

### Requirements

- Python 3.8+
- PyTorch
- segmentation-models-pytorch (`pip install segmentation-models-pytorch`)
- pandas, numpy, matplotlib, PIL, tqdm, sklearn
