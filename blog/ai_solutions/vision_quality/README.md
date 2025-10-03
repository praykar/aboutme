# Vision Quality Control Demo (Streamlit)

This directory contains a minimal computer vision quality control demo showing image inspection with dummy pass/fail classification based on brightness, blur detection, and contrast analysis.

## Quick start

* **Python 3.9+**
* **Install deps:**

```bash
pip install -U streamlit numpy pillow
streamlit run app.py
```

## Architecture

```
Image Upload --> Image Processing --> Quality Checks --> Pass/Fail Decision
     |                  |                    |                |
     v                  v                    v                v
 [File/Sample]  [Brightness/Blur]   [Threshold Compare]  [Report Display]
```

* **Image metrics**: Brightness, blur (Laplacian variance), contrast (std dev)
* **QC checks**: Configurable thresholds for each metric
* **Results**: Visual pass/fail indicators with detailed metrics

## Technical Summary

### QC Metrics

1. **Brightness**: Average pixel intensity (0-255 range)
2. **Blur Detection**: Laplacian variance for sharpness
3. **Contrast**: Standard deviation of pixel intensities

### Decision Logic

* Each metric compared against user-defined thresholds
* Overall pass requires ALL checks to pass
* Visual indicators (✅/❌) for each metric

## Files

* **app.py**: Streamlit app with image upload, QC analysis, and results display
* **README.md**: This file

## Demo Behavior

1. Upload an image or generate sample image
2. Adjust QC thresholds in sidebar
3. View overall pass/fail status
4. See detailed metrics for each quality check
5. Inspect detailed check results with target ranges

## References

* **Computer Vision**: Szeliski, "Computer Vision: Algorithms and Applications". [http://szeliski.org/Book/](http://szeliski.org/Book/)
* **OpenCV**: [https://opencv.org/](https://opencv.org/)
* **PIL/Pillow**: [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)
* **Industrial Vision**: Cognex. "Machine Vision Fundamentals". [https://www.cognex.com/](https://www.cognex.com/)

## Notes for Hugging Face Space Integration

* **Add requirements.txt**: streamlit, numpy, pillow
* **Lightweight**: CPU-only, no deep learning models
* **Multi-demo integration**: Expose as tab in unified launcher
* **Production alternatives**: Use CNN models (ResNet, EfficientNet) for defect classification, YOLO for object detection, U-Net for segmentation
