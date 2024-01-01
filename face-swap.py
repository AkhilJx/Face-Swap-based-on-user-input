import os
import cv2
import numpy as np

from common import media_utils

width = 640
height = 480
result_height = 700
result_width = 700


# Function to display image with text and wait for user input
def display_image_with_text(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 700))


    # Add text to the image
    text = "PRESS 1 FOR MALE CELEBRITIES !!"
    text2 = "PRESS 2 FOR FEMALE CELEBRITIES !!"

    cv2.putText(img, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Image', img)

    # Wait for user input (1 or 2)
    key = 0
    while key not in [ord('1'), ord('2')]:
        key = cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()

    return chr(key)


# Replace 'your_image_path.jpg' with the actual path to your image file
image_path = 'BACKGROUND.png'


# Display image with text and get user input
user_choice = display_image_with_text(image_path)

# print(type(user_choice))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("videos/human_face_video.mp4")
cap.set(3, width)
cap.set(4, height)

if user_choice == '1':
    # For static images:
    src_image_paths = [
        "male/MESSI.jpg",
        "male/ELON_MUSK.jpg",
        "male/RONALDO.jpg",
        "male/TOM_CRUISE.jpg",
        "male/WILL_SMITH.jpg",
        "male/VIRAT_KOHLI.jpeg",
        "male/DWAYNE_JOHNSON.jpg",
        "male/DAVID_BECKHAM.jpg",
        "male/LEONARDO_DICAPRIO.jpg",
        "male/PUTIN.jpg",
    ]

if user_choice == '2':
    # For static images:
    src_image_paths = [
        "female/ANGELINA_JOLIE.jpeg",
        "female/EMA_WATSON.jpg",
        "female/SHAKIRA.jpg",
        "female/KRISTEN_STEWART.jpg",
        "female/SERENA_WILLIAMS.jpg",
        "female/PRIYANKA_CHOPRA.jpg",
        "female/JENNIFER_LOPEZ.jpg",
        "female/KATE_WINSLET.jpg",
        "female/BEYONCE.jpg",
        "female/SELENA_GOMEZ.jpg",
    ]

print(src_image_paths)
# src_image_paths = ["images/psy.jpg"]

src_images = []
for image_path in src_image_paths:
    image = cv2.imread(image_path)
    src_images.append(image)


def set_src_image(image):
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
    src_image = image
    src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    src_mask = np.zeros_like(src_image_gray)

    src_landmark_points = media_utils.get_landmark_points(src_image)
    src_np_points = np.array(src_landmark_points)
    src_convexHull = cv2.convexHull(src_np_points)
    cv2.fillConvexPoly(src_mask, src_convexHull, 255)

    indexes_triangles = media_utils.get_triangles(convexhull=src_convexHull,
                                                  landmarks_points=src_landmark_points,
                                                  np_points=src_np_points)

current_src_image_index = 0
set_src_image(src_images[0])
while True:
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles

    _, dest_image = cap.read()
    dest_image = cv2.resize(dest_image, (width, height))

    dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
    dest_mask = np.zeros_like(dest_image_gray)

    dest_landmark_points = media_utils.get_landmark_points(dest_image)
    if dest_landmark_points is None:
        continue
    dest_np_points = np.array(dest_landmark_points)
    dest_convexHull = cv2.convexHull(dest_np_points)

    height, width, channels = dest_image.shape
    new_face = np.zeros((height, width, channels), np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        points, src_cropped_triangle, cropped_triangle_mask, _ = media_utils.triangulation(
            triangle_index=triangle_index,
            landmark_points=src_landmark_points,
            img=src_image)

        # Triangulation of second face
        points2, _, dest_cropped_triangle_mask, rect = media_utils.triangulation(triangle_index=triangle_index,
                                                                                 landmark_points=dest_landmark_points)

        # Warp triangles
        warped_triangle = media_utils.warp_triangle(rect=rect, points1=points, points2=points2,
                                                    src_cropped_triangle=src_cropped_triangle,
                                                    dest_cropped_triangle_mask=dest_cropped_triangle_mask)

        # Reconstructing destination face
        media_utils.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

    # Face swapped (putting 1st face into 2nd face)
    # new_face = cv2.medianBlur(new_face, 3)
    result = media_utils.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                       dest_convexHull=dest_convexHull, new_face=new_face)

    result = cv2.medianBlur(result, 3)
    h, w, _ = src_image.shape
    rate = width / w

    t = src_image_paths[current_src_image_index]
    t = os.path.splitext(os.path.basename(t))[0]
    t = str(current_src_image_index)+". "+t

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 0, 0)  # White color

    cv2.putText(src_image, t, (1, 50), font, font_scale, text_color, font_thickness)
    cv2.imshow("Source image", cv2.resize(src_image, (result_width, result_height)))

    # Add text to the result image
    text = "Press"+' "ESC" '+"to Quit!!"
    text2 = "Press"+' "(0 to 9)" '+"to change the personalities!!"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = .9
    font_thickness = 2
    text_color = (0, 0, 255)  # White color
    text_color2 = (91, 109, 0)

    result = cv2.resize(result, (result_width, result_height))

    cv2.putText(result, text, (1, 50), font, font_scale, text_color, font_thickness)
    cv2.putText(result, text2, (1, 90), font, font_scale, text_color2, font_thickness)

    # cv2.imshow("New face", new_face)
    cv2.imshow("Result", result)

    # Keyboard input
    key = cv2.waitKey(3)
    # ESC
    if cv2.waitKey(30) & 0xFF == 27:
        quit()
    # Source image change
    if ord("0") <= key <= ord("9"):
        num = int(chr(key))
        if num < len(src_images):
            current_src_image_index = num
            set_src_image(src_images[num])

cv2.destroyAllWindows()
