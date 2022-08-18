import os
import cv2

from markline.boltdetect import Bolt_State
from markline.cablejointdetect import CableJoint_State
from markline.crossboltdetect import CrossBolt_State
from markline.fasteningdetect import Fastening_State
from markline.groundterminaldetect import GroundTerminal_State
from markline.pipeclampdetect import PipeClamp_State
from markline.unionnutdetect import UnionNut_State

root_path = r"E:\04-Data\02-robot\04-Xingqiao0512\Markline"

def Bolt_main():



    bolt_path = os.path.join(root_path, "Bolt")

    for i in os.listdir(bolt_path):
        if i == ".DS_Store":
            os.remove(os.path.join(bolt_path, i))

    bolt_file = os.listdir(bolt_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in bolt_file:

        image = cv2.imread(os.path.join(bolt_path, i))

        state, old_img = Bolt_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/bolt_result/39/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/bolt_result/39/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/bolt_result/39/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/bolt_result/39/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def CableJoint_main():

    root_path = "/Users/kerwinji/Desktop/Markline"

    cj_path = os.path.join(root_path, "CableJoint")

    for i in os.listdir(cj_path):
        if i == ".DS_Store":
            os.remove(os.path.join(cj_path, i))

    cj_file = os.listdir(cj_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in cj_file:

        image = cv2.imread(os.path.join(cj_path, i))

        state, old_img = CableJoint_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/cablejoint_result/59/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/cablejoint_result/59/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/cablejoint_result/59/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/cablejoint_result/59/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def CrossBolt_main():

    # root_path = "/Users/kerwinji/Desktop/Markline"

    cb_path = os.path.join(root_path, "CrossBolt")

    for i in os.listdir(cb_path):
        if i == ".DS_Store":
            os.remove(os.path.join(cb_path, i))

    cb_file = os.listdir(cb_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in cb_file:

        image = cv2.imread(os.path.join(cb_path, i))

        state, old_img = CrossBolt_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/crossbolt_result/513/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("//Users/kerwinji/Desktop/tagline_detect/crossbolt_result/513/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/crossbolt_result/513/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/crossbolt_result/513/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def Fastening_main():

    # root_path = "/Users/kerwinji/Desktop/Markline"

    f_path = os.path.join(root_path, "Fastening")

    for i in os.listdir(f_path):
        if i == ".DS_Store":
            os.remove(os.path.join(f_path, i))

    f_file = os.listdir(f_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in f_file:

        image = cv2.imread(os.path.join(f_path, i))

        state, old_img = Fastening_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/fastening_result/59/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/fastening_result/59/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/fastening_result/59/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/fastening_result/59/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def GroundTerminal_main():

    # root_path = "/Users/kerwinji/Desktop/Markline"

    gt_path = os.path.join(root_path, "GroundTerminal")

    for i in os.listdir(gt_path):
        if i == ".DS_Store":
            os.remove(os.path.join(gt_path, i))

    gt_file = os.listdir(gt_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in gt_file:

        image = cv2.imread(os.path.join(gt_path, i))

        state, old_img = GroundTerminal_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/groundterminal_result/59/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/groundterminal_result/59/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/groundterminal_result/59/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/groundterminal_result/59/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def PipeClamp_main():

    # root_path = "/Users/kerwinji/Desktop/Markline"
    # root_path = "/Users/kerwinji/Desktop/tagline_detect/pipeclamp_result/511"
    pc_path = os.path.join(root_path, "sd")

    for i in os.listdir(pc_path):
        if i == ".DS_Store":
            os.remove(os.path.join(pc_path, i))

    pc_file = os.listdir(pc_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in pc_file:

        image = cv2.imread(os.path.join(pc_path, i))

        state, old_img = PipeClamp_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/pipeclamp_result/511/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/pipeclamp_result/511/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/pipeclamp_result/511/sd1", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/pipeclamp_result/511/zc", i), image)
        #     print("state =", state, "放入zc中了", )


def UnionNut_main():

    # root_path = "/Users/kerwinji/Desktop/Markline"

    un_path = os.path.join(root_path, "UnionNut")

    for i in os.listdir(un_path):
        if i == ".DS_Store":
            os.remove(os.path.join(un_path, i))

    un_file = os.listdir(un_path)
    # a_file.sort(key = lambda x: int (x[:3]))

    for i in un_file:

        image = cv2.imread(os.path.join(un_path, i))

        state, old_img = UnionNut_State(image)
        print(state)

        # if state == -3:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/unionnut_result/511/wu", i), image)
        #     print("state =", state, "放入wu中了")
        #
        # elif state == -2:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/unionnut_result/511/wx", i), image)
        #     print("state =", state, "放入wx中了")
        #
        # elif state == 0:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/unionnut_result/511/sd", i), image)
        #     print("state =", state, "放入sd中了")
        #
        # else:
        #     cv2.imwrite(os.path.join("/Users/kerwinji/Desktop/tagline_detect/unionnut_result/511/zc", i), image)
        #     print("state =", state, "放入zc中了", )


if __name__ == "__main__":

    '''
    -3 无有效轮廓线
    1 无松动
    0 松动
    
    '''
    a = int(input("输入你想检测的项点类型对应的数字:1:Bolt, 2:CableJoint, 3:CrossBolt, 4:Fastening, 5:GroundTerminal, 6:PipeClamp, 7:UnionNut:"))
    print(type(a))
    if a == 1:
        Bolt_main()
    elif a == 2:
        CableJoint_main()
    elif a == 3:
        CrossBolt_main()
    elif a == 4:
        Fastening_main()
    elif a == 5:
        GroundTerminal_main()
    elif a == 6:
        PipeClamp_main()
    elif a == 7:
        UnionNut_main()
    else:
        print("输入有误，请输入数字1到数字7")