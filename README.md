# VSLR_Pytorch (Vietnamese Sign Language Recognition with PyTorch)

## Tabel of Content

 - [Introduction](#Introduction)
 - [Feature](#Feature)
 - [Project Structure](#Project-Structure)
 - [Installation](#Installation)
 - [How It Works](#How-It-Works)
 - [Contact](#Contact)
    
## Introduction
This project aims to develop a system for recognizing and interpreting Vietnamese Sign Language (VSL) using machine learning and computer vision techniques. Built with PyTorch, the system leverages real-time hand tracking and gesture recognition to facilitate communication between the deaf community and non-signers in Vietnam.

## Features
- **Real-time sign language recognition**: 
  - Processes continuous video input from a webcam or pre-recorded video.  
- **Support for Vietnamese characters**: 
  - Recognizes a wide range of VSL gestures, including special Vietnamese characters (Ă, Â, Ê, Ô, Ơ, Ư).  
- **Smart character filtering**:  
  - Consecutive images with the same character are filtered to keep only the first occurrence, avoiding duplicate characters.  
  - Special characters are detected using two-step gestures (e.g., "A" followed by "dấu mũ" becomes "Â").  
- **Word and sentence formation**:  
  - If no hand is detected for 0.5 to 1 second, a space is added to separate words, forming complete Vietnamese sentences.  
- **User-friendly interface**: 
  - Designed to be accessible for both developers and end-users.

## Project Structure
```bash
VSLR_Pytorch/
├── images/                                # Images for documentation
├── src/                                   # Source code
│   ├── classification.py                  # Script for classification logic
│   ├── collect_data.py                    # Script to collect new data
│   ├── config.py                          # Configuration settings
│   ├── dataset.py                         # Dataset handling
│   ├── hand_tracking.py                   # Hand detection and tracking
│   ├── model.py                           # CNN model definition
│   ├── utils.py                           # Utility functions
├── trained_models/                        # Pre-trained models and parameters
├── inferent.py                            # Inference script
├── app.py                                 # App script
├── train.ipynb                            # Training Jupyter notebook
├── requirements.txt                       # Required Python packages 
├── README.md                              # Project documentation
```

## Installation
1. **Clone the repository**
```bash
git clone https://github.com/HiewNT/VSLR_Pytorch.git
cd VSLR_Pytorch
```
2. **Install the required libraries**
```bash
pip install -r requirements.txt
```
3. **VSL dataset**
You can create your own dataset:
```bash
python src/collect_data.py
```

4. **Train the model (Optional)**
```bash
run train.ipynb
```

5. **Run the recognition system**
```bash
python app.py
```

## How It Works
- **Continuous Frame Processing**:  
  - Images captured by the computer are processed in real-time.  
  - If consecutive frames detect the same character, only the first instance is retained to prevent repetition.  

- **Special Character Detection**:  
  - Vietnamese special characters (e.g., Ă, Â, Ê, Ô, Ơ, Ư) are recognized using two gestures.  
  - For example:  
    - "A" followed by "dấu mũ" (hat sign) becomes "Â".  
    - "O" followed by "dấu râu" (hook sign) becomes "Ơ".  

- **Sentence Formation**:  
  - If no hand is detected for 0.5 to 1 second, a space is inserted after the current word.  
  - This allows the system to build complete Vietnamese sentences.  

- **Some other functions**:
  - Delete the last character of the recognized sentence
  - Delete all recognized sentences
  - Save the recognized sentence to file

## Contact

For any questions or issues, please contact me at nthieu.1703@gmail.com.
