import cv2


def get_selective_search():
    return cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def get_rects(img, st):
    gs = get_selective_search()
    config(gs, img, st)
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]
    return rects[0:200]


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)
    if strategy == 's':
        gs.switchToSingleStrategy()
    elif strategy == 'f':
        gs.switchToSelectiveSearchFast()
    elif strategy == 'q':
        gs.switchToSelectiveSearchQuality()


def rect_img(img, rects, bndboxs):
    for x1, y1, x2, y2 in rects[0:50]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=1)
    for x1, y1, x2, y2 in bndboxs:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 235, 255), thickness=1)
    cv2.imshow("rect", img)
    cv2.waitKey()


# 156,97,351,270

def is_cross(rect1, rect2):
    cx1 = (rect1[0] + rect1[2]) / 2
    cy1 = (rect1[1] + rect1[3]) / 2
    cx2 = (rect2[0] + rect2[2]) / 2
    cy2 = (rect2[1] + rect2[3]) / 2
    cw = abs(rect1[0] - rect1[2]) / 2 + abs(rect2[0] - rect2[2]) / 2
    ch = abs(rect1[1] - rect1[3]) / 2 + abs(rect2[1] - rect2[3]) / 2
    if abs(cx2 - cx1) < cw and abs(cy1 - cy2) < ch:
        return True
    return False


def IoU(bndboxs, rects, ts):
    max_area = 0
    print("bndboxs",bndboxs)
    for xmin, ymin, xmax, ymax in bndboxs:
        b_area = abs(xmax - xmin) * abs(ymax - ymin)
        if max_area < b_area:
            max_area = b_area

    x = []
    y = []
    flag = False
    netaive_box = list()
    for rect in rects:
        for bndbox in bndboxs:
            x = []
            y = []
            xmin, ymin, xmax, ymax = bndbox
            x.append(xmin)
            x.append(xmax)
            x.append(rect[0])
            x.append(rect[2])
            y.append(ymin)
            y.append(ymax)
            y.append(rect[1])
            y.append(rect[3])
            x.sort()
            y.sort()

            aArea = abs(xmax - xmin) * abs(ymax - ymin)
            bArea = (abs(rect[2] - rect[0])) * (abs(rect[3] - rect[1]))

            jiao = 0
            if is_cross(bndbox, rect):
                jiao = (abs(x[2] - x[1])) * (abs(y[2] - y[1]))
            iou = jiao / (aArea + bArea - jiao)
            # if iou>0:
            #     print(iou)
            # print(rect)
            # print(jiao)
            # print(x)
            # print(y)
            # 6079
            if iou <= 0.3 and bArea > max_area / 5.0:
                flag = True
        if flag:
            netaive_box.append(rect)
        flag = False
    # print(len(netaive_box))
    return netaive_box
