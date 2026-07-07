# ai4ultrasound-keypoint-detect
Detecting pleural and B-lines via keypoint detection methods. Use keypoints to compute percent pleura.

# Installation
1. _Visual Studio Code_ (optional, can use other editor)
   - [https://code.visualstudio.com/download?_exp_download=d53503e735](https://code.visualstudio.com/download?_exp_download=d53503e735)
   - Install Python, Python Debugger, Pylance, Python Environment and Jupyter extensions
2. _Install git_
   - [https://git-scm.com/install/](https://git-scm.com/install/)
3. _Install Microsoft C++ Build Tools_
   - [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select 'Desktop development with C++'
4. _UV_ (optional, can simply use pip)
   - [https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)
5. _Virtual Environment & Python Setup_ (required)
   - We use python 3.11.15 => version number is important
   - Create a 'Venvs' folder, create the venv with the correct python version, activate the venv:
     ```
     cd <root_directory>\Venvs
     uv venv --python 3.11 keypointdetect_venv
     keypointdetect_venv\Scripts\activate
     ``` 
6. _Verify And Install CUDA If Using NVIDIA GPU_ (required)
   - Run ``` nvcc --version ```, if no CUDA toolkit version, proceed with CUDA installation (we have CUDA compilation tools version 13.0)
   - Open 'device manager' on device (if using windows), check under Display Adapters for the GPU
   - Open 'NVIDIA control panel' on device and verify NVIDIA GPU card type and driver version (we are using NVIDIA GeForce RTX 5070 with driver version 581.95)
   - Open 'NVIDIA app' on device, navigate to drivers tab, download latest updates
   - Go to 'NVIDIA CUDA Toolkit download' website [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download CUDA toolkit for system
   - Rerun ``` nvcc --version ``` to verify install (we have CUDA compilation tools version 13.0)
7. _Install Torch_ (required)
   - To install with your CUDA version (previous step), use official website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - We ran:
     ```
     uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu13
     ```
   - Fix setuptools (for MMPose compatability):
     ```
     uv pip install "setuptools<70" wheel pip
     ```
8. _Install MMPose_ (required)
   - Install mmengine: ```uv pip install mmengine```
   - Create a dependencies directory (see repo diagram): <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies
   - Clone and build mmcv from source:
     ```
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies
     deactivate
     git clone https://github.com/open-mmlab/mmcv.git
     cd mmcv
     git checkout v2.1.0
     cv <root_directory>\Venvs
     keypointdetect_venv\Scripts\activate
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\dependencies\mmcv
     uv pip install -r requirements/optional.txt
     $env:CL = "/Zc:preprocessor"
     uv pip install -v -e . --no-build-isolation
     ```
   - Install MMDet and MMPose and openmim:
     ```
     uv pip install mmdet==3.2.0
     uv pip install "mmpose>=1.1.0" --no-build-isolation
     uv pip install -U openmim
     ```
9. _Update NumPy, xtcoco tools, and setuptools for MMPose Compatability_ (required)
     ```
     uv pip uninstall numpy
     uv pip install "numpy==1.26.4" --no-deps
     uv pip install xtcocotools --force-reinstall --no-binary xtcocotools --no-build-isolation
     uv pip install "setuptools>=65.0,<70" --force-reinstall
     ```
10. _Install requirements.txt_ (required)
     ```
     cd  <root_directory>\Code\ai4ultrasound-keypoint-detect
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
     ```
     cd <root_directory>\Code\ai4ultrasound-keypoint-detect\mmpose_demo
     uv run mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
     ```
   - Run the mmpose_demo (should produce a demo_result.jpg in the mmpose_demo folder):
     ```
     python .\mmpose_demo.py
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

