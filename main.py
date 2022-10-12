import argparse
import tensorflow as tf
import numpy as np

import cv2
import Score_Sheet
import cv_utils


print('Đọc dữ liệu từ model tensorflow model...')
predict_model = tf.keras.models.load_model('./models/model_tensorflow')
predict_model.summary()


img_path = './input/2.jpg'

num_rows = 19


parser = argparse.ArgumentParser()
parser.add_argument("--num_rows", help="set num rows ")
parser.add_argument("--img_path", help="specify path to input image")
parser.add_argument("--debug", help="specify debug to stop at show_window")

args = parser.parse_args()

if args.num_rows:
    num_rows = int(args.num_rows)

if args.debug:
    cv_utils.set_debug(bool(args.debug))

if args.img_path:
    img_path = args.img_path

print("Đọc hình ảnh theo đường dẫn", img_path)
input_img = cv2.imread(img_path)

img_sheet, img_binary_sheet, img_binary_only_numbers, cells_bounding_rects = Score_Sheet.generate_sheet(input_img, num_rows_in_grid=num_rows)
# Debugging step
img_cells = img_sheet.copy()
cv_utils.draw_bounding_rects(img_cells,cells_bounding_rects)
cv_utils.show_window('img_cells', img_cells)


digit_contours = cv_utils.get_external_contours(img_binary_only_numbers)

#gọi hàm vẽ Contourts cho các con số
for i, cnt in enumerate(digit_contours):

    digit_bounding_rect = cv2.boundingRect(cnt)
    x, y, w, h = digit_bounding_rect


    cell = Score_Sheet.validate_and_find_cell(cells_bounding_rects, digit_bounding_rect)
    if cell is None:
        continue

    # Chuyển thành ảnh nhị phân trắng đen roi
    roi = img_binary_sheet[y:y+h, x:x+w]
    roi_fit_20x20 = 20 / max(roi.shape[0], roi.shape[1])
    # Resize với INTER_NEAREST
    roi = cv2.resize(roi, None, fx=roi_fit_20x20, fy=roi_fit_20x20, interpolation=cv2.INTER_NEAREST)

    roi_background = np.zeros((28, 28), dtype=roi.dtype)
    roi_background[4:4+roi.shape[0], 4:4+roi.shape[1]] = roi

    # Lưu roi
    cv2.imwrite("./output/roi/original/roi_" +
                str(i) + ".png", roi_background)

    delta_x, delta_y = cv_utils.get_com_shift(roi_background)
    roi_background = cv_utils.shift_by(roi_background, delta_x, delta_y)
    cv2.imwrite("./output/roi/shifted/roi_" +
                str(i) + ".png", roi_background)
    roi_background = roi_background - 127.5
    roi_background /= 127.5
    prediction = predict_model(np.reshape(roi_background, (1, 28, 28, 1)))
    predicted_digit = np.argmax(prediction)
    cv2.rectangle(img_sheet, (x, y),
                  (x+w, y+h), (100, 10, 100), 1)
    cv2.putText(img_sheet, str(predicted_digit), (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
imgKQ=cv2.imread(img_path)
resized = cv2.resize(imgKQ,(1000,1000), interpolation = cv2.INTER_AREA)
cv2.imshow("KQ",resized)
cv_utils.show_window('img_sheet', img_sheet)
cv2.waitKey(0)