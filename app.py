import streamlit as st
from  PIL import Image
import numpy as np
#import tensorflow as tf
from tensorflow import Graph as Graph
from tensorflow import import_graph_def
from tensorflow.compat.v1 import GraphDef as GraphDef
from tensorflow.compat.v1 import Session as Session
from tensorflow.io.gfile import GFile as GFile
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


# What model to download.
MODEL_NAME = 'E:\AIML-\Diabetic-Ratinopathy-master\optic_disc_macula_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = 'resnet-inference-graph.pb'
NUM_CLASSES = 2

detection_graph = Graph()
with detection_graph.as_default():
    od_graph_def = GraphDef()
    with GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


labelmap = {1: {'id': 1, 'name': 'optic_disease'}, 2: {'id': 2, 'name': 'macula'}}
dmp =[]

def pred(img):
    with detection_graph.as_default():
        with Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # for image_path in img:
            # image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(img)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            dmp.append([boxes, scores, classes, num])
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                # category_index,
                labelmap,
                use_normalized_coordinates=True,
                line_thickness=40)
            # plt.figure(figsize=(24,16))
            # x = image_path.split("\\")
            # x = list(map(lambda x: x.replace('tst_img', 'res_img'), x))
            # fn = '//'.join(x)
            # plt.imsave(fn,image_np)
            # plt.imshow(image_np)
            # plt.imsave(fn,image_np)
    return(image_np)



#User Interface---------------------------------------------------------

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

pred_flag = False
def main():
    st.label_visibility='collapse'
    st.title("diabetic ratinopathy Prediction")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<p style="text-align: center;"><label>Image : </label></p>',unsafe_allow_html=True)
        st.image(image,width=500)
    if st.button("Predict"):
        x = pred(image)
        st.markdown('<p style="text-align: center;"><label>Prediction : </label></p>',unsafe_allow_html=True)
        st.image(x,width=900)
        # result =''
        # st.success('The output is {}'.format(result))
if __name__ == '__main__': #
    main()

