import ast
import io
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from PIL import Image


class CaptionBot:
    def __init__(self, image):
        self.__wordtoindexpath = "wordToIndex.txt"
        self.__indextowordpath = "indexToWord.txt"
        self.__model = load_model("model_99.h5")
        self.__image = image
        self.__resnet_model = ResNet50(weights = "imagenet", input_shape=(224,224,3))
        self.__newmodel = Model(self.__resnet_model.input,self.__resnet_model.layers[-2].output)
        self.__indextoword = {}
        self.__wordtoindex = {}
        self.__featurevector = None
        with open(self.__wordtoindexpath) as f1:
            with open(self.__indextowordpath) as f2:
                self.__wordtoindex = f1.read();
                self.__wordtoindex = ast.literal_eval(self.__wordtoindex)
                self.__indextoword = f2.read();
                self.__indextoword = ast.literal_eval(self.__indextoword)

    def preprocess_image(self):
        self.__image = Image.open(io.BytesIO(self.__image))
        self.__image = self.__image.resize((224, 224))
        self.__image = np.array(self.__image)
        self.__image = np.expand_dims(self.__image,axis=0)
        self.__image = preprocess_input(self.__image)
        return self.__image

    def encode_image(self):
        self.__image = self.preprocess_image()
        featurevector = self.__newmodel.predict(self.__image)
        featurevector = featurevector.reshape((-1,))
        return featurevector
    
    def predict_caption(self):
        max_len = 74
        in_text = "startSeq"
        for i in range(max_len):
            sequence = [self.__wordtoindex[w] for w in in_text.split() if w in self.__wordtoindex]
            sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
            ypred =  self.__model.predict([self.__image,sequence])
            ypred = ypred.argmax()
            word = self.__indextoword[ypred]
            in_text+= ' ' +word
            if word =='endSeq':
                break
        final_caption =  in_text.split()
        final_caption = final_caption[1:-1]
        final_caption = ' '.join(final_caption)
        return final_caption

    def run(self):
        self.__featurevector = self.encode_image();
        self.__image = self.__featurevector.reshape((1,2048))
        caption = self.predict_caption();
        return caption
    
