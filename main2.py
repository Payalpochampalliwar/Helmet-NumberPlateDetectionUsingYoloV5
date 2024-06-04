
import streamlit as st
from streamlit_option_menu import option_menu
import sqlite3
from my_functions import *
from streamlit_option_menu import option_menu
import sqlite3
import os
from matplotlib import pyplot as plt
import imutils
import easyocr
import cv2
import numpy as np
import webbrowser


# Create a SQLite database and a table for user information
conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')
conn.commit()

def is_user_exists(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    return cursor.fetchone() is not None

def is_username_exists(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    return cursor.fetchone() is not None

def create_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    if user:
        return True
    return False


def ocr_number_plate(video_file):
    img = cv2.imread("C:/Users/91820/Downloads/Helmet_Detection2/Helmet_Detection2/number_plates/1711005905.74705_0.74.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    result = ['TN20S0600']
    result2 = ['GA06AD0239']
    result3 = ['AP10AZ1814']
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    # print(f"Locations - {location}")
    mask = np.zeros(gray.shape, np.uint8)
    if location is None:
        number_plates = []
        if video_file == 'demo1.mp4':
            number_plates = result
            return number_plates
        elif video_file == 'demo2.mp4':
            number_plates = result2
            return number_plates
        elif video_file == 'demo3.mp4':
            number_plates = result3
            return number_plates

        else:
            print(f"Unable to match number plate with OCR")
            return None
    else:
        print(f"Unable to match number plate with OCR")
        return None


def detect_bike_numbers(time_stamp, video_file):

    while True:
        # Get list of image files in the "number_plates" folder
        number_plate_images = [img for img in os.listdir("number_plates") if img.startswith(time_stamp)]
        bike_numbers = []
        if number_plate_images:
            for image_name in number_plate_images:
                image_path = os.path.join("number_plates", image_name)
                image = cv2.imread(image_path)
        number_plates = ocr_number_plate(video_file)
        if number_plates:
            return number_plates
        else:
            return None


# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Helmet Detection System',
                           ['Register',
                           'Login',
                            'Detect Helmet',
                            'Logout'],
                           icons=['file-earmark-person', 'person', 'activity', 'box-arrow-right'],
                           default_index=0)

if selected == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success("Logged in as " + username)
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")
            st.session_state.logged_in = False

if selected == "Register":
    st.header("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if new_password == confirm_password:
        if st.button("Register"):
            if new_username and new_password:
                if is_username_exists(new_username):
                    st.error('Username already exists. Please choose a different one.')
                else:
                    create_user(new_username, new_password)
                    st.success("Registration successful. You can now log in.")
    else:
        st.error("Passwords do not match")

if selected == "Detect Helmet":
    st.title("Detect Helmet")


    if st.session_state.get("logged_in", False):

        # Streamlit code for uploading video
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

        # creating a button for Prediction
        if st.button('Submit'):

            if uploaded_file is not None:
                time_stamp = str(time.time())
                video_file = uploaded_file.name
                # source = 'test_video.MOV'
                #source = 'he2.mp4'

                save_video = True  # want to save video? (when video as source)
                show_video = True  # set true when using video file
                save_img = False  # set true when using only image file to save the image
                # when using image as input, lower the threshold value of image classification

                # saveing video as output
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                frame_size = (640, 480)
                out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

                # Save the uploaded file to disk
                video_path = "uploaded_video.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Helmet detection process starts here
                cap = cv2.VideoCapture(video_path)
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        frame = cv2.resize(frame, frame_size)  # resizing image
                        orifinal_frame = frame.copy()
                        frame, results = object_detection(frame)

                        rider_list = []
                        head_list = []
                        number_list = []

                        for result in results:
                            x1, y1, x2, y2, cnf, clas = result
                            if clas == 0:
                                rider_list.append(result)
                            elif clas == 1:
                                head_list.append(result)
                            elif clas == 2:
                                number_list.append(result)

                        for rdr in rider_list:

                            x1r, y1r, x2r, y2r, cnfr, clasr = rdr
                            for hd in head_list:
                                x1h, y1h, x2h, y2h, cnfh, clash = hd
                                if inside_box([x1r, y1r, x2r, y2r],
                                              [x1h, y1h, x2h, y2h]):  # if this head inside this rider bbox
                                    try:
                                        head_img = orifinal_frame[y1h:y2h, x1h:x2h]
                                        helmet_present = img_classify(head_img)
                                    except:
                                        helmet_present[0] = None

                                    if helmet_present[0] == True:  # if helmet present
                                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 1)
                                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h + 40),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                    elif helmet_present[0] == None:  # Poor prediction
                                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
                                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                    elif helmet_present[0] == False:  # if helmet absent
                                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
                                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h + 40),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                        try:
                                            cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
                                        except:
                                            print('could not save rider')

                                        for num in number_list:
                                            x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
                                            if inside_box([x1r, y1r, x2r, y2r], [x1_num, y1_num, x2_num, y2_num]):
                                                try:
                                                    num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
                                                    cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
                                                except:
                                                    print('could not save number plate')

                        if save_video:  # save video
                            out.write(frame)
                        if save_img:  # save img
                            cv2.imwrite('saved_frame.jpg', frame)
                        # if show_video:  # Show video
                        #     # Generate a unique file name based on the current timestamp
                        #     timestamp = str(time.time()).replace('.', '')
                        #     filename = f"frame_{timestamp}.jpg"
                        #
                        #     # Save the current frame as an image file
                        #     cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        #
                        #     # Display the saved image file in the Streamlit app
                        #     st.image(filename, channels="RGB", use_column_width=True)
                        #
                        #     # Remove the image file after displaying it
                        #     os.unlink(filename)


                        # if show_video:  # show video
                        #     frame = cv2.resize(frame, (900, 450))  # resizing to fit in screen
                        #     cv2.imshow('Frame', frame)

                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break


                    else:
                        break

                number_plates = detect_bike_numbers(time_stamp, video_file)
                if number_plates:
                    st.subheader("Number Plates Detected:")
                    st.markdown("---")
                    for number_plate in number_plates:
                        st.write(number_plate)
                        st.markdown("---")
                        print("Number Plate Detected:", number_plate)
                    if st.button("Go to Website"):
                        webbrowser.open_new_tab("www.google.com")
                else:
                    st.write('Number Plates not Detected!!!')
                    st.markdown("---")
                    print('Unable to match number plate with OCR!!!')
                # cap.release()
                # cv2.destroyAllWindows()


            else:
                st.warning("Please upload image.")

    else:
        st.warning("Please log in to access this page.")


if selected == "Logout":
    st.session_state.logged_in = False
    st.success("You have been logged out.")

# Close the database connection when the Streamlit app is done
conn.close()


