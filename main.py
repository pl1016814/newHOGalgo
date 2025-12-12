import cv2, numpy as np

hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)
#window size - block size - block movement - cell size - num of angle groups
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#opencv only takes 8-bit image files
edges = lambda proFullImage: hog.compute(cv2.resize(cv2.GaussianBlur(cv2.equalizeHist(proFullImage), (5,5), 0), (128, 128)).astype(np.uint8))
#very small lambda function to clean up image and make it easy for hog
#Gaussian Blur softens noise through blur and Equalize Hist fixes lighting differences.
target = cv2.imread("target.png", cv2.IMREAD_GRAYSCALE)
target = edges(target)

fullImage = cv2.resize(cv2.imread("CROWDIDSNEY.jpg"), None, fx=2, fy=2) #resized by scale of 2
proFullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(proFullImage, 1.25, 1, minSize=(40,40))
#scans the image for anything that looks like a face that is of reasonable size
match = None
similarityScore = 1000

for x, y, w, h in faces: #checks each face
	#the lines minus each other below
	similarity = np.linalg.norm(target - edges(proFullImage[y:y+h, x:x+w])) #how similar; lower the difference, the more similar
	if similarity < similarityScore:
		match = (x,y,w,h)
		similarityScore = similarity
if match:
	x, y, w, h = match #what faces really is is a list of rectangles; there are 4 points on a rectangle
	cv2.rectangle(fullImage, (x, y), (x+w, y+h), (0,255,0), 2) #2 pixels thick, 0, 255, 0 means BGR

cv2.imshow("This Works", fullImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

