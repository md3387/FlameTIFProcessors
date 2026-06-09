# FlameTIFProcessors
tif file image processors intended to determine flame area from a 2D image - written in Python
*All codes intended to work with ".tif" NOT ".tiff" files, even if comments and filenames suggest otherwise.

# python codes included
- mp4_from_tifffile.py - converts .tif files into a movie file for viewing on any windows system.   Not a necessary precursor to running any of the additional processing codes in this repository.  It's just useful for quick viewing of the images, which are not otherwise viewable with standard windows programs.
- tfb_flamefront_01_preprocessing.py - Determines the conversion from pixels to inches using an image of a standard ruler, or any other object where the vertical distance between two points is known. Default known distance is 3 inches. The burner nozzle should also be visible in the image so that the user can select the location of the nozzle exit and centerline.  Typically, we set the ruler on top of the burner nozzle and use that image.
- tfb_flamefront_02_processing_MDH.py - Interactive flame area calculation from .tif file.  Requires values from tfb_flamefront_01_preprocessing.py. This is only useful for images using visible light.  It is not written to process OH* cross-sections.  Key features:
      1) calculates the contour of a VISIBLE wavelength flame image using a sobel transform
      2) post-processing of sobel transform to fill gaps and remove nonphysical spikes from contour
      3) interactive user review to include or exclude individual images from the average based on user-identified noise
      4) saves 17 key features of image selection to a .csv file ("ml_training_data.csv") that can be used in ML training so that eventually the user review can be replaced by machine review.
      5) outputs averaged flame data : area(mm^2) avg and stdev, max flame diameter, average flame height
- FlameClassifierTraining.py - creates and trains an AI model to classify images automatically.  It reads "ml_training_data.csv," adjusts for "include vs "exclude" class imbalance, cross-validates by training the model 5 times on 5 separate chunks of the data, creates one final Random Forest classifier using all labled frames, then determines which of the 17 key features were most useful for making correct predictions.  Finally, it reports how many frames it classified differently from the user.
        Outputs:
        1) the model "frame_classifier.joblib"
        2) a bar plot showing relative importance of the 17 key features
        3) performance metrics in the command window.  Example below.
  

# Workflow:
1) Before experiments, capture a spatial calibration image (single image .tif of a ruler on top of burner)\\
2) Conduct flame image capture experiments and save files as .tif - transfer to postprocessing computer in folder defined by "filepath" in ...01 and ...02 processors.
3) Open tfb_flamefront_01_preprocessing.py, change "filename" to the spatial calibration image filename.  Save.
4) Run tfb_flamefront_01_preprocessing.py - and record values for "spatial_cal_mm_pix", "jet_exit_plane_pix", and "jet_exit_centerline_pix"
5) Open tfb_flamefront_02_processing_MDH.py, change "filename" to the tif file with flame images you want to postprocess. Save.
6) Run tfb_flamefront_02_processing_MDH.py - and include/exclude images based on noise. More conservative inclusion leads to higher accuracy average flame area, AND it trains the ML to be more conservative (i.e. more trustworthy).  Upon completion, this will create a file called "ml_training_data.csv" in the same directory where your .tif image files are stored.  If the file has already been created, the code will apppend training data to it.  Nothing needs to be done with that file until you have processed a large (~2000) number of images 
7) Record the outputs from  tfb_flamefront_02_processing_MDH.py into your flame database.
8) Repeat 1) through 7) for many representative flame cases. Data processing and ML training are simultaneous in these steps
9)  Run FlameClassifierTraining.py when you believe you have produced a large enough training data set (~>2000images).



The idea is that your workload shrinks over time as the model improves:
Early sessions:   review all 250 frames  →  save labels  →  accumulate CSV
After ~2000 frames: train first model
Later sessions:   model auto-labels 200+ frames  →  you review only ~30 uncertain ones
Eventually:       you only review edge cases; model handles the rest






# FlameClassifierTraining.py example output
Loaded 7750 labeled frames
  Included: 3983  (51.4%)
  Excluded: 3767  (48.6%)
After dropping NaN rows: 7711 frames remaining

Running 5-fold cross-validation...
  F1 score per fold: [0.953 0.939 0.947 0.945 0.94 ]
  Mean F1: 0.945 +/- 0.005
  (F1=1.0 is perfect, F1=0.5 is roughly random for balanced classes)

Training final model...

Feature importances:
redge_std         0.218
ledge_std         0.119
area_clean        0.090
area_raw          0.090
diameter_std      0.077
center_std        0.062
area_change       0.053
redge_nan_frac    0.048
ledge_nan_frac    0.047
img_p90           0.034
grad_mean         0.030
grad_std          0.029
img_std           0.029
img_mean          0.026
img_max           0.020
grad_max          0.020
img_p10           0.008
Confusion matrix (training set — optimistic):
[[3710   18]
 [  16 3967]]
              precision    recall  f1-score   support
     exclude       1.00      1.00      1.00      3728
     include       1.00      1.00      1.00      3983
    accuracy                           1.00      7711
   macro avg       1.00      1.00      1.00      7711
weighted avg       1.00      1.00      1.00      7711
Model saved to: D:\Downloads\AE471_CY2026_Spring\frame_classifier.joblib

Interpretation:
----------------
Confusion matrix (training set — optimistic):
[[3710   18]
 [  16 3967]]

 Means 
             Predicted exclude    Predicted include
Actual exclude        3710                  18
Actual include          16                3967

Out of 7711 frames the model made only 34 mistakes total — 18 frames it called "include" that I had labeled "exclude", and 16 frames it called "exclude" that I had labeled "include". That's a 99.6% accuracy on the training data.
The script prints "optimistic" because the model was tested on the same data it was trained on
The honest performance estimate is the 5-fold cross-validation F1 score printed earlier "F1 score per fold: [0.953 0.939 0.947 0.945 0.94 ]"  Values in the 90s are pretty good.
--------------------

