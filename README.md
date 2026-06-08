# FlameTIFProcessors
tif file image processors intended to determine flame area from a 2D image - written in Python
*All codes intended to work with ".tif" NOT ".tiff" files, even if comments and filenames suggest otherwise.

#python codes included
- mp4_from_tifffile.py - converts .tif files into a movie file for viewing on any windows system.   Not a necessary precursor to running any of the additional processing codes in this repository.  It's just useful for quick viewing of the images, which are not otherwise viewable with standard windows programs.
- tfb_flamefront_01_preprocessing.py - Determines the conversion from pixels to inches using an image of a standard ruler, or any other object where the vertical distance between two points is known. Default known distance is 3 inches. The burner nozzle should also be visible in the image so that the user can select the location of the nozzle exit and centerline.  Typically, we set the ruler on top of the burner nozzle and use that image.
- tfb_flamefront_02_processing_MDH.py - Interactive flame area calculation from .tif file.  Requires values from tfb_flamefront_01_preprocessing.py. This is only useful for images using visible light.  It is not written to process OH* cross-sections.  Key features:
      1) calculates the contour of a VISIBLE wavelength flame image using a sobel transform
      2) post-processing of sobel transform to fill gaps and remove nonphysical spikes from contour
      3) interactive user review to include or exclude individual images from the average based on user-identified noise
      4) saves key features of image selection to a .csv file that can be used in ML training so that eventually the user review can be replaced by machine review.
      5) outputs averaged flame data : area(mm^2) avg and stdev, max flame diameter, average flame height
- FlameClassifierTraining.py -
