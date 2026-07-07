# ai4ultrasound-keypoint-detect
Detecting pleural and B-lines via keypoint detection methods. Use keypoints to compute percent pleura.

# Installation
1. _Visual Studio Code_ (optional, can use other editor)
   - [https://code.visualstudio.com/download?_exp_download=d53503e735](https://code.visualstudio.com/download?_exp_download=d53503e735)
   - Install Python, Python Debugger, Pylance, Python Environment and Jupyter extensions
3. _UV_ (optional, can simply use pip)
   - [https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)
3. _Virtual Environment & Python Setup_ (required)
   - We use python 3.11.15 => version number is important
   - Create a 'Venvs' folder, create the venv with the correct python version, activate the venv:
     ```
     cd ...\Venvs
     uv venv --python 3.11 keypointdetect_venv
     keypointdetect_venv\Scripts\activate
     ``` 
5. _Verify And Install CUDA If Using NVIDIA GPU_ (required)
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
9. _Install MMPose_ (required)
    - 
11. t
12. t
13. t
14. t
15. t
16. 
