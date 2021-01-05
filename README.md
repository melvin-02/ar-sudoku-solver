# AUGMENTED REALITY SUDOKU SOLVER  
The AR sudoku solver looks for a sudoku puzzle in live video, solves the puzzle and displays the solution onto the live video in realtime.
Works even with skewness and upto 45 degree rotation.  
  
## Working  
Download the repo, create a new python environment and install the requirments.txt
```
conda create -n sudoku python==3.8
conda activate sudoku
pip install -r requirements.txt
```
Run the application.py file
```
python application.py
```
  
## Example
![Demo video](images/demo_.gif)  

## Limitations
-Webcam solver cannot detect when a new puzzle has entered the frame, will try to warp the solution of the first puzzle it sees onto any subsequent puzzles
-Cannot solve puzzles that don't have a distinguishable four-point outer border
