import cv2
import visualize
# made by www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
# initialize the list of reference points
refPt = []
image = []

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates
		refPt.append((x, y))
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (255, 255, 255), 2)
		cv2.imshow("image", image)
		
 
#Er komt een afbeelding van de radiograph en je moet klikken en slepen om een kader te maken.
#Als deze kader de gevraagde tand goed afbakend, klik je op C (correct)
#Als er teveel kaders op staan en je wil beginnen met een nieuwe afbeelding, klik op R (reset)
#Return waarde is linker-boven-hoek en rechter-onder-hoek van de gekozen kader
def askForEstimate(radiograph):
    global refPt, image
    
    m,n = radiograph.shape
    image = cv2.resize(radiograph,(1000,500))
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    # keep looping until the 'c' key is pressed
    while True:
   	# display the image and wait for a keypress
   	cv2.imshow("image", image)
   	key = cv2.waitKey(1) & 0xFF
    
   	# if the 'r' key is pressed, reset the cropping region
   	if key == ord("r"):
  		image = clone.copy()
    
   	# if the 'c' key is pressed, break from the loop
   	elif key == ord("c"):
  		break
    
    # close all open windows
    cv2.destroyAllWindows()
    
    # herschaal de waarden zodat ze passen op de originele image
    [(a,b),(c,d)] = refPt
    x1 = int(a * n / 1000)
    y1 = int(b * m / 500)
    
    x2 = int(c * n / 1000)
    y2 = int(d * m / 500)
    
    # zet x1 en y1 op de kleinste waarden, zodat 1 altijd links-boven en 2 rechts-boven is
    if x1 > x2:
        temp = x1
        x1 = x2
        x2 = temp
    if y1 > y2:
        temp = y1
        y1 = y2
        y2 = temp
    
    # reset de globale waarden
    refPt = []
    image = []
    
    return [(x1,y1),(x2,y2)]