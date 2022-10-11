from re import X
import cv2
import numpy as np
import open3d as o3d


points = []

class Projection(object):

    #self.data_member :image,height,width,channels
    def __init__(self, image_path, points):  
        """
            :param points: Selected pixels on top view(BEV) image
        """
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)  #save img data
        self.height, self.width, self.channels = self.image.shape
        self.bev_uv=points

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=np.pi/2):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        ### TODO ###
        f=0.5*512/(np.tan(fov/2)) #caculate focus length

        #turn piexl coordinate to 3D cooridnate             
        z_bev=1.5
        CORBEV=[]
        for no in self.bev_uv:
            CORBEV.append([z_bev*(no[0]-256)/f,
                            z_bev*(no[1]-256)/f,
                            z_bev,
                            1])                     #CORBEV 4x4

        #trans 3D_bev_coordinate to 3D_front_coordinate
        rot=np.pi/2
        transform=np.array([[1,0,0,0],
                            [0,np.cos(rot),-np.sin(rot),0],
                            [0,np.sin(rot),np.cos(rot),-1.5],
                            [0,0,0,1]])
        cor_bev=[]
        for no in CORBEV:
            cor_bev.append(np.dot(transform,no))

        #trans 3D_forn to 2D_image
        new_pixels=[]
        for no in cor_bev:
            new_pixels.append([int(-f*no[0]/no[2]+256),int(f*no[1]/no[2]+256)]) #CORBEV 4x4
        # print(new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)
        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('bev_image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('bev_image', img)


if __name__ == "__main__":

    pitch_ang = -90
    front_rgb = "front_view_path.png"
    top_rgb = "top_view_path.png"

    # click the pixels on windo/w
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('bev_image', img)
    cv2.setMouseCallback('bev_image', click_event)
    cv2.waitKey(0)

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    print(new_pixels)
    projection.show_image(new_pixels)
    cv2.destroyAllWindows()