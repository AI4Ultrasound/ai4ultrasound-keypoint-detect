# ai4ultrasound-keypoint-detect
Detecting pleural and B-lines via keypoint detection methods. Use keypoints to compute percent pleura.

# Author & Contact Information

*Author:* Alexandre L. Banks Gadbois

*Email:* alexandre_banksgadbois@hms.harvard.edu or abanksga@unb.ca

*Affiliations:* Harvard University (Dr. Tina Kapur Lab)

# Project Repository
The overall project repository should be organized as follows, with this Github repo cloned into the 'Code/' folder.
```
root_directory
+---Code/
|   +---ai4ultrasound-keypoint-detect/
|   |   +---dependencies/
|   |   |   +---mmcv/
|   |   +---mmpose_demo/
|   |   +---src/
|   |       LICENSE
|   |       README.md
|   |       requirements.txt
+---Data/
|   +---Group-001/
|   +---Keypoint_Detect_Data/
|   +---usframecache/
+---Venvs/
|   +---keypointdetect_venv/
```

The 'Group-001' folder contains the raw data and the original data structure tree which is formatted as:
```
+---<Annotator>/
|   +---<Annotator>-<site_id>/
|   |       <clipid>.<Annotator>.json
|   |       <clipid>.dcm
|   |       ...
|   +---<Annotator>-<site_id>/
|   +---<Annotator>-<site_id>/
|   +---<Annotator>-<site_id>/
|   +---<Annotator>-<site_id>/
|   ...
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>/
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>/
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>/
|   ...
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>/
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>_<probe_orientation>/
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>_<probe_orientation>/
|   +---<Annotator>-<site_id>_<patient_id>_<diuretic_time>_<probe_orientation>/
|   ...
+---<Annotator>/
|   ...
...
```

The new data structure (re-organized by preprocessing.py) is in the 'Keypoint_Detect_Data' folder, which has the following structure:
Overview of new data structure:
```
Keypoint_Detect_Data/COCO_Data
+---annotations/
|   +---scanline/
|   |       <Annotator>_<clipid>.json
|   |       <Annotator>_<clipid>.json
|   |       ...
|   +---sector/
|           <Annotator>_<clipid>.json
|           <Annotator>_<clipid>.json
|           ...
+---images/
    +---scanline/
    |       <Annotator>_<clipid>_<framenum>.png
    |       <Annotator>_<clipid>_<framenum>.png
    |       ...
    +---sector/
            <Annotator>_<clipid>_<framenum>.png
            <Annotator>_<clipid>_<framenum>.png
            ...
```

# Requirements
- NVIDIA GPU recommended (required for CUDA acceleration)
  - macOS: No NVIDIA CUDA support — uses CPU or Apple MPS instead
- ~50GB free disk space:
   - ~10GB for models, dependencies and venv
   - ~40GB for data
- Internet connection (for downloading packages and model weights)
  
# Platform Support
| Platform | GPU Acceleration | Notes |
|---|---|---|
| Windows | CUDA (NVIDIA) | Full support |
| Linux | CUDA (NVIDIA) | Full support |
| macOS (Intel) | CPU only | No CUDA, no MPS |
| macOS (Apple Silicon M1/M2/M3/M4) | MPS | Metal Performance Shaders |

# Installation
1. _Visual Studio Code_ (optional, can use other editor)
   - [https://code.visualstudio.com/download?_exp_download=d53503e735](https://code.visualstudio.com/download?_exp_download=d53503e735)
   - Install Python, Python Debugger, Pylance, Python Environment and Jupyter extensions
2. _Install git_
   - [https://git-scm.com/install/](https://git-scm.com/install/)
3. _Install Microsoft C++ Build Tools_
   
   **Windows:**
   - [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select 'Desktop development with C++'
     
   **Linux:**
   ```
   bash
   sudo apt update
   sudo apt install build-essential  # Ubuntu/Debian
   # or
   sudo yum groupinstall "Development Tools"  # CentOS/RHEL
   ```
   **macOS:**
   ```
   xcode-select --install
   ```
5. _UV_ (optional, can simply use pip)
   - [https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)
6. _Virtual Environment & Python Setup_ (required)
   - We use python 3.11.15 => version number is important
   - Create a 'Venvs' folder, create the venv with the correct python version, activate the venv:
     
     **Windows:**
     ```
     cd <root_directory>\Venvs
     uv venv --python 3.11 keypointdetect_venv
     keypointdetect_venv\Scripts\activate
     ```
     **Linux/macOS:**
     ```
     cd <root_directory>/Venvs
     uv venv --python 3.11 keypointdetect_venv
     source keypointdetect_venv/bin/activate
     ```
7. _Verify And Install CUDA If Using NVIDIA GPU_ (required)
   - Run:
     
     **Windows:**
     ```
     nvcc --version
     nvidia-smi
     ```
     **Linux:**
     ```
     nvidia-smi
     sudo apt install nvidia-driver-<version>
     ```
   - If no CUDA toolkit version, proceed with CUDA installation (we have CUDA compilation tools version 13.0):
      - Open 'device manager' on device (if using windows), check under Display Adapters for the GPU
      - Open 'NVIDIA control panel' on device and verify NVIDIA GPU card type and driver version (we are using NVIDIA GeForce RTX 5070 with driver version 581.95)
      - Open 'NVIDIA app' on device, navigate to drivers tab, download latest updates
      - Go to 'NVIDIA CUDA Toolkit download' website [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download CUDA toolkit for system
      - Rerun ``` nvcc --version ``` to verify install (we have CUDA compilation tools version 13.0)
8. _Install Torch_ (required)
   - To install with your CUDA version (previous step), use official website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - We ran (on windows):
     ```
     uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
     ```
   - Fix setuptools (for MMPose compatability):
     ```
     uv pip install "setuptools<70" wheel pip
     ```
9. _Install MMPose_ (required)
   - Install mmengine: ```uv pip install mmengine```
   - Create a dependencies directory (see repo diagram): <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies
   - Clone and build mmcv from source:
     
     **Windows:**
     ```
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies
     deactivate
     git clone https://github.com/open-mmlab/mmcv.git
     cd mmcv
     git checkout v2.1.0
     cd <root_directory>\Venvs
     keypointdetect_venv\Scripts\activate
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies\mmcv
     $env:CL = "/Zc:preprocessor"
     uv pip install -r requirements/optional.txt     
     uv pip install . --no-build-isolation #This can take a long time to build
     ```
     **Linux:**
     ```
     cd <root_directory>/Code/ai4ultrasound-keypoint-detect/dependencies
     deactivate
     git clone https://github.com/open-mmlab/mmcv.git
     cd mmcv
     git checkout v2.1.0
     cd <root_directory>/Venvs
     source keypointdetect_venv/bin/activate
     cd <root_directory>/Code/ai4ultrasound-keypoint-detect/dependencies/mmcv
     uv pip install -r requirements/optional.txt
     uv pip install . --no-build-isolation  # This can take a long time to build
     ```
     **macOS:**
     ```
     cd <root_directory>/Code/ai4ultrasound-keypoint-detect/dependencies
     deactivate
     git clone https://github.com/open-mmlab/mmcv.git
     cd mmcv
     git checkout v2.1.0
     cd <root_directory>/Venvs
     source keypointdetect_venv/bin/activate
     cd <root_directory>/Code/ai4ultrasound-keypoint-detect/dependencies/mmcv
     
     MMCV_WITH_OPS=1 FORCE_CUDA=0 uv pip install -r requirements/optional.txt
     MMCV_WITH_OPS=1 FORCE_CUDA=0 uv pip install . --no-build-isolation  # This can take a long time to build
     ```
   - Install MMDet and MMPose and openmim:
     ```
     uv pip install -U openmim
     uv pip install mmdet==3.2.0
     uv pip install "mmpose>=1.1.0" --no-build-isolation
     ```
10. _Update NumPy, xtcoco tools, and setuptools for MMPose Compatability_ (required)
     ```
     uv pip uninstall numpy
     uv pip install "numpy==1.26.4" --no-deps
     uv pip install xtcocotools --force-reinstall --no-binary xtcocotools --no-build-isolation
     uv pip install "setuptools>=65.0,<70" --force-reinstall
     ```
11. _Install requirements.txt_ (required)
    
     **Windows:**
     ```
     cd  <root_directory>\Code\ai4ultrasound-keypoint-detect
     uv pip install -r requirements.txt
     ```
    **Linux/macOS:**
    ```
    cd <root_directory>/Code/ai4ultrasound-keypoint-detect
    uv pip install -r requirements.txt
    ```
   - **Notes:**
      - Might need to remove '+cu130' from requirements.txt
      - Might need to remove 'mmcv @ file:///C:/Users/Alexandre%20Banks/Documents/Research_Summer2026/Code/ai4ultrasound-keypoint-detect/dependencies/mmcv' from requirements.txt
11. _Fix torch.load function_ (required)
   - Edit mmengine checkpoint file:
      -   Move to '<root_directory>\Venvs\keypointdetect_venv\Lib\site-packages\mmengine\runner\checkpoint.py'
      -   Open the checkpoint.py file
      -   Find the line: 'checkpoint = torch.load(filename, map_location=map_location)' and change it to: 'checkpoint = torch.load(filename, map_location=map_location, weights_only=False)'
12. _Setup mmpose_demo.py_ (optional)
   - Download the keypoint detection models for the mmpose_demo.py script:
     
     **Windows:**
     ```
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\mmpose_demo
     uv run mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
     ```
     **Linux/macOS:**
     ```
     cd <root_directory>/Code/ai4ultrasound-keypoint-detect/mmpose_demo
     uv run mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
     ```
   - Run the mmpose_demo (should produce a demo_result.jpg in the mmpose_demo folder):
     ```
     python mmpose_demo.py
     ```
13. _Run Other Package Checks_ (optional)
     ```
     python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
     python -c "import mmcv; print(mmcv.__file__)" # Should show path inside site-packages
     python -c "import mmcv; print(mmcv.__version__)"
     python -c "import mmengine; print(mmengine.__version__)"
     python -c "import mim; print('mim ok')"   # Must not error
     python -c "import mmpose; print(mmpose.__version__)"     # Must not error
     python -c "import mmdet; print(mmdet.__version__)"    
     python -c "import numpy; print(numpy.__version__)"  # Should show 1.26.4
     python -c "from xtcocotools.coco import COCO; print('xtcocotools ok')"
     python -c "import pkg_resources; print('pkg_resources ok')"
     ```

