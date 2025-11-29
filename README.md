# Python Digital Image Processing Projects

**Voting-Based Document Image Skew Detection**

Implementation of document skew detection and correction using multiple methods with voting system, based on academic research paper.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

## Overview

This project implements document image skew detection algorithms based on the research paper:

**Boiangiu, C.-A., Dinu, O.-A., Popescu, C., Constantin, N., & Petrescu, C. (2020).**  
*[Voting-Based Document Image Skew Detection](https://doi.org/10.3390/app10072236).*  
Applied Sciences, 10(7), 2236.

The implementation combines three detection methods:
- **FFT (Fast Fourier Transform)**: Frequency domain analysis with Hough line detection
- **Projection Profiling**: Connected components analysis with variance optimization  
- **Hough Transform**: Spatial domain line detection with parallel grouping

Results are combined using a confidence-based voting system to select the most reliable skew angle.

## Project Structure

```
Python-Digital-Image-Processing-Projects/
├── notebooks/                     # Original Jupyter implementations
│   ├── PCD-Voting-Based-Documents.ipynb
│   └── project_digital_image_processing.ipynb
├── sample_images/                 # Test dataset (95+ documents)
├── streamlit_app.py              # Web application interface
├── skew_detector.py              # Command-line version
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # MIT license
```

## Usage

### Web Application
```bash
streamlit run streamlit_app.py
```
- Upload document images (JPG, PNG)
- Real-time skew detection and correction
- Visual comparison of results
- Download corrected images

### Command Line
```bash
python skew_detector.py path/to/image.jpg
```
Output displays detection results from all three methods and final voting decision.

## Installation

```bash
git clone https://github.com/arezyhs/Python-Digital-Image-Processing-Projects.git
cd Python-Digital-Image-Processing-Projects
pip install -r requirements.txt
```

## Dependencies

- **Core**: numpy, scipy, matplotlib, scikit-image, opencv-python, Pillow
- **Web App**: streamlit
- **Notebooks**: jupyter

## Authors

- [arezyhs](https://github.com/arezyhs)  
- [SaktiiAJA](https://github.com/SaktiiAJA)  

## Reference

Boiangiu, C.-A., Dinu, O.-A., Popescu, C., Constantin, N., & Petrescu, C. (2020). Voting-Based Document Image Skew Detection. *Applied Sciences*, 10(7), 2236. https://doi.org/10.3390/app10072236

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
