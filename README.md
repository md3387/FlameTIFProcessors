# FlameTIFProcessors
tif file image processors intended to determine flame area from a 2D image - written in Python
*All codes intended to work with ".tif" NOT ".tiff" files, even if comments and filenames suggest otherwise.

#python codes included
- mp4_from_tifffile.py - converts .tif files into a movie file for viewing on any windows system.   Not a necessary precursor to running any of the additional processing codes in this repository.  It's just useful for quick viewing of the images, which are not otherwise viewable with standard windows programs.
- tfb_flamefront_01_preprocessing.py - Determines the conversion from pixels to inches using an image of a standard ruler, or any other object where the vertical distance between two points is known. Default known distance is 3 inches. The burner nozzle should also be visible in the image so that the user can select the location of the nozzle exit and centerline.  Typically, we set the ruler on top of the burner nozzle and use that image.
-  
